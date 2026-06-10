"""Whisper Phase-3-bundle end-to-end validation (GPU).

Validates the cjm-torch-plugin-utils adoption (release_model + cuda_oom +
resolve_torch_device), the Shape-1 heartbeat around the urllib/Azure-CDN model
download, and the Track 19 WORKER_ENV migration (incl. the de-quirked
XDG_CACHE_HOME=${CJM_MODELS_DIR}) live, mirroring the Voxtral-HF Phase 3 pattern.

Run from the whisper repo root after:
  1. `cjm-ctl --cjm-config cjm.yaml setup-runtime`
  2. `cjm-ctl --cjm-config cjm.yaml install-all --plugins plugins_test.yaml`
     (whisper + ffmpeg + cjm-system-monitor-nvidia)
  3. A short speech clip at test_files/short_test_audio.mp3

Then:
  conda run -n cjm-transcription-plugin-whisper --no-capture-output \\
    python tests_manual/validate_whisper_e2e.py
"""
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
)
log = logging.getLogger("whisper-e2e")

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_AUDIO = REPO_ROOT / "test_files" / "short_test_audio.mp3"
MANIFESTS_DIR = REPO_ROOT / ".cjm" / "manifests"
EMPIRICAL_DB = REPO_ROOT / ".cjm" / "empirical_resources.db"

PLUGIN_NAME = "cjm-transcription-plugin-whisper"
SYSMON_NAME = "cjm-system-monitor-nvidia"
FFMPEG_NAME = "cjm-media-plugin-ffmpeg"


def check_prereqs() -> None:
    assert TEST_AUDIO.exists(), f"Missing test audio: {TEST_AUDIO}"
    assert MANIFESTS_DIR.exists(), (
        f"Missing manifests dir: {MANIFESTS_DIR} — run cjm-ctl setup-runtime + install-all first"
    )
    for name in (PLUGIN_NAME, SYSMON_NAME, FFMPEG_NAME):
        assert (MANIFESTS_DIR / f"{name}.json").exists(), f"Missing manifest: {name}.json"
    log.info("Prereqs OK: test audio + whisper + nvidia-monitor + ffmpeg manifests present")


def assert_manifest_shape() -> None:
    manifest = json.loads((MANIFESTS_DIR / f"{PLUGIN_NAME}.json").read_text())
    assert manifest["format_version"] == "2.0", manifest["format_version"]
    code = manifest["code"]

    desc = code.get("description") or manifest.get("description") or ""
    assert desc.strip(), "manifest description is empty (T24 regression)"
    log.info(f"Manifest T24 description: {desc!r}")

    tax = code["taxonomy"]
    assert tax["domain"] == "transcription" and tax["role"] == "TranscriptionPlugin", tax
    assert code["resources"]["requires_gpu"] is True, code["resources"]
    for stale in ("min_gpu_vram_mb", "recommended_gpu_vram_mb", "min_system_ram_mb"):
        assert stale not in code["resources"], f"stale resource field present: {stale}"
    log.info(f"Manifest CR-1/Phase-5a: taxonomy={tax}, resources={code['resources']}")

    # Track 19: static CUDA_VISIBLE_DEVICES + OMP_NUM_THREADS + templated XDG_CACHE_HOME.
    worker_env = code.get("worker_env", [])
    by_name = {e["name"]: e for e in worker_env}
    assert {"CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "XDG_CACHE_HOME"} <= set(by_name), (
        f"Track 19 WORKER_ENV missing expected vars: {sorted(by_name)}"
    )
    xdg_default = by_name["XDG_CACHE_HOME"].get("default", "")
    assert xdg_default == "${CJM_MODELS_DIR}", f"XDG_CACHE_HOME default not templated/de-quirked: {xdg_default!r}"
    install_env = manifest.get("install", {}).get("env_vars", {})
    assert not install_env, f"install.env_vars should be empty post-migration: {install_env}"
    log.info(f"Manifest Track 19 worker_env: {sorted(by_name)} | XDG_CACHE_HOME default={xdg_default!r}; install.env_vars empty")


def run_e2e() -> None:
    import asyncio

    from cjm_plugin_system.core.manager import PluginManager
    from cjm_plugin_system.core.config import get_config
    from cjm_plugin_system.core.queue import JobQueue
    from cjm_plugin_system.core.ports import Composition, CompositionNode, NodeState, OutputRef

    cfg = get_config()
    log.info(f"data_dir={cfg.data_dir}, models_dir={cfg.models_dir}")

    pm = PluginManager(search_paths=[MANIFESTS_DIR], sysmon_plugin_name=SYSMON_NAME)
    pm.discover_manifests()
    log.info(f"Discovered: {[m.name for m in pm.discovered]}")

    pm.load_plugin(next(m for m in pm.discovered if m.name == SYSMON_NAME))
    pm.load_plugin(next(m for m in pm.discovered if m.name == FFMPEG_NAME))
    whisper_meta = next(m for m in pm.discovered if m.name == PLUGIN_NAME)
    db_path = whisper_meta.manifest.get("db_path")
    # tiny model keeps the download + inference quick while still exercising the full flow.
    ok = pm.load_plugin(whisper_meta, config={"model": "tiny"})
    assert ok, f"Failed to load {PLUGIN_NAME}"
    whisper_id = whisper_meta.name
    log.info(f"Loaded {SYSMON_NAME} + {FFMPEG_NAME} + {PLUGIN_NAME} (model=tiny); db_path={db_path}")

    # CR-4 prefetch: urllib/Azure-CDN model download (cold cache) wrapped by the heartbeat.
    log.info("Calling prefetch() to download + load the Whisper model...")
    t0 = time.time()
    pm.get_plugin(whisper_id).prefetch()
    log.info(f"prefetch() returned in {time.time() - t0:.1f}s")

    # CR-16 (stage 3): the composition binds the consumer's input to ffmpeg's
    # ACTUAL hashed cache_dir_for_config output path at execution time via
    # OutputRef — the predict-the-path pattern is retired.
    async def run_composition() -> Any:
        queue = JobQueue(deps=pm, sysmon_plugin_name=SYSMON_NAME)
        await queue.start()
        try:
            comp_id = await queue.submit_composition(Composition(nodes=[
                CompositionNode("convert", FFMPEG_NAME, {
                    "action": "convert", "input_path": str(TEST_AUDIO),
                    "output_format": "wav", "sample_rate": 16000, "channels": 1,
                }),
                CompositionNode("transcribe", whisper_id,
                                {"audio": OutputRef("convert", "output_path")}),
            ]))
            log.info(f"Submitted composition {comp_id}: ffmpeg.convert -> whisper.execute")
            run = await queue.wait_for_composition(comp_id)
            if run.status != NodeState.completed:
                raise RuntimeError(f"Composition {comp_id} status={run.status}; nodes={run.node_runs}")
            return run.results_by_node()["transcribe"]
        finally:
            await queue.stop()

    log.info(f"Submitting composition for {TEST_AUDIO.name}...")
    t0 = time.time()
    result = asyncio.run(run_composition())
    from cjm_transcription_plugin_system.core import TranscriptionResult  # noqa: F401 — registers the wire kind (typed decode)
    text = result.text  # typed TranscriptionResult (stage-2 wire layer)
    log.info(f"Composition completed in {time.time() - t0:.1f}s: text={text[:120]!r}")
    assert text and text.strip(), f"Empty transcription; raw result={result!r}"

    # Plugin DB: confirm the transcription row persisted.
    if db_path and Path(db_path).exists():
        con = sqlite3.connect(db_path)
        try:
            for t in [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
                n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                log.info(f"plugin DB {t}: {n} rows")
        finally:
            con.close()

    # Empirical store: GPU plugin -> assert a NON-ZERO gpu peak (real subtree attribution).
    assert EMPIRICAL_DB.exists(), f"empirical store not created: {EMPIRICAL_DB}"
    con = sqlite3.connect(EMPIRICAL_DB)
    gpu_peak = 0.0
    try:
        for t in [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]:
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})").fetchall()]
            if "gpu_memory_mb_peak_max" not in cols:
                continue
            for r in con.execute(
                f"SELECT * FROM {t} WHERE plugin_name=? OR instance_id=? OR instance_id LIKE ?",
                (PLUGIN_NAME, whisper_id, f"{PLUGIN_NAME}%"),
            ).fetchall():
                row = dict(zip(cols, r))
                log.info(f"  empirical {t}: {row}")
                gpu_peak = max(gpu_peak, float(row.get("gpu_memory_mb_peak_max") or 0.0))
    finally:
        con.close()
    assert gpu_peak > 0.0, "empirical gpu_memory_mb_peak is 0 — subtree GPU attribution failed"
    log.info(f"GPU attribution VERIFIED: whisper gpu_memory_mb_peak_max={gpu_peak:.1f} MB")

    pm.unload_plugin(whisper_id)
    pm.unload_plugin(FFMPEG_NAME)
    pm.unload_plugin(SYSMON_NAME)
    log.info("Unloaded plugins; validation done.")


def main() -> int:
    check_prereqs()
    assert_manifest_shape()
    run_e2e()
    log.info("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
