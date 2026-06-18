# Tombstone — `test_reconfigure.py` (RETIRED 2026-06-18, stage 9)

**Origin:** `cjm-transcription-plugin-whisper/tests_manual/test_reconfigure.py` (2026-05-25).
**Retired because:** pre-overhaul per-tool instance of the **substrate CR-4 reconfigure contract**; the cohort is retired, not patched. (Canonical framing: `cjm-media-plugin-silero-vad/tests_manual/test_reconfigure.md` — this is a substrate-behavior test, reimplement against `cjm-substrate`, not per tool.)

**What it validated (contract-level, fake model — no real GPU load):**
- `reconfigure(model` flip`)` → `RELOAD_TRIGGER` → `_release_model` + `_apply_config`.
- non-trigger field (`temperature`) → must NOT release the model.
- `device` change → also a `RELOAD_TRIGGER`.
- `on_disable` → releases the model (CR-2).

**Coverage status:** UNIQUE (substrate reconfigure delta path; not in the cores' happy-path). **Reimplementation target:** a single `cjm-substrate` reconfigure test with a `_Fake*` tool covering trigger vs non-trigger deltas + on_disable release — supersedes all per-tool copies.
