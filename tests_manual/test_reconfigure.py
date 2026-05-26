"""CR-4 reconfigure-lifecycle validation for the Whisper plugin.

Contract-level (no real model load — the Whisper model is large/GPU; the wiring
is what we validate). Exercises the substrate's reconfigure delta path in-process
with a fake model object:

  1. reconfigure(model flip) -> RELEASE the model (RELOAD_TRIGGER -> _release_model)
     + RE-APPLY config (_apply_config) -- the behavior CR-4 wired into
     PluginManager.update_plugin_config
  2. a non-trigger field change (temperature) must NOT release the model
  3. a device change is also a RELOAD_TRIGGER
  4. on_disable releases the model (CR-2)

Requires the substrate version with the two-phase reconfigure (CR-4). Run from the
repo root in the plugin's env:

    conda run -n cjm-transcription-plugin-whisper --no-capture-output python tests_manual/test_reconfigure.py

Becomes a pytest under Track 17. A real-model + real-audio variant belongs in a
GPU-marked pytest (test_files/).
"""
import sys

CPU = {"model": "base", "device": "cpu"}


def main() -> int:
    from cjm_transcription_plugin_whisper.plugin import WhisperLocalPlugin

    p = WhisperLocalPlugin()
    p._apply_config(CPU)
    assert p.config.model == "base" and p.device == "cpu"

    # 1) model trigger: release + re-apply
    p.model = object()  # simulate a loaded model
    p.reconfigure(CPU, {"model": "small", "device": "cpu"})
    assert p.model is None, "RELOAD_TRIGGER model must fire _release_model"
    assert p.config.model == "small", "reconfigure must re-apply the new config (CR-4)"
    print("[1] reconfigure model base->small: released + applied  OK")

    # 2) non-trigger field (temperature) retains the model
    p.model = object()
    p.reconfigure({"model": "small", "device": "cpu", "temperature": 0.0},
                  {"model": "small", "device": "cpu", "temperature": 0.5})
    assert p.model is not None, "non-trigger (temperature) change must NOT release the model"
    assert p.config.temperature == 0.5, "config still applied on non-trigger change"
    print("[2] temperature change (non-trigger): model retained + applied  OK")

    # 3) device is also a trigger
    p.model = object()
    p.reconfigure({"model": "small", "device": "cpu"}, {"model": "small", "device": "auto"})
    assert p.model is None, "device change must release the model"
    print("[3] reconfigure device cpu->auto: released  OK")

    # 4) on_disable releases (CR-2)
    p.model = object()
    p.on_disable()
    assert p.model is None, "on_disable must release the model"
    print("[4] on_disable: model released  OK")

    print("RECONFIGURE VALIDATION: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
