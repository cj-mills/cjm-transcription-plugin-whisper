# Tombstone — `validate_whisper_e2e.py` (RETIRED 2026-06-18, stage 9)

**Origin:** `cjm-transcription-plugin-whisper/tests_manual/validate_whisper_e2e.py` (Phase-3-bundle era).
**Retired because:** imported `TranscriptionResult` from the now-dissolved `cjm-transcription-plugin-system.core` shim (GitHub-archived 2026-06-18; the DTO now lives in `cjm-capability-primitives`). Per the stage-9 decision the pre-overhaul `tests_manual` cohort is **retired, not patched**.

**What it validated:** the cjm-torch-plugin-utils adoption (`release_model` / `cuda_oom` / `resolve_torch_device`), the heartbeat around the model download, the WORKER_ENV migration (templated `XDG_CACHE_HOME=${CJM_MODELS_DIR}`), the v2.0 manifest shape, and a real transcription with empirical-DB GPU-peak assertions.

**Coverage status:** SUPERSEDED — `cjm-transcription-core`'s standing both-transcriber e2e validates whisper end-to-end on the real corpus; install-all schema-v2 validation covers the manifest shape; admission/empirical behavior is exercised by the cores.

**Reimplementation target:** none required (cores are the standing harness). If per-tool manifest/empirical micro-assertions are still wanted, fold them into the core loop-back, not a per-tool script.
