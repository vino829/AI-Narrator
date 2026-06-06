"""
Phase 2.3 - Emotion + cfg_value experiment.
Tests reference cloning with different emotions and cfg_value settings.
Run AFTER seed audios have been generated and selected.

Usage:
    python experiments/emotion_cfg_test.py [role] [--cfg 1.5,2.0,2.5]

    If no role specified, tests all roles with seed_audio in voices.json.
"""

import json
import sys
import time
from pathlib import Path

import soundfile as sf

SAMPLE_RATE = 48000

EMOTIONS = ["平靜", "開心", "憤怒", "悲傷", "驚訝"]
DEFAULT_CFG_VALUES = [1.5, 2.0, 2.5]

TEST_TEXT = "這句話用來測試不同情緒和參數下的語音效果，希望能找到最佳的設定。"


def main():
    base = Path(__file__).resolve().parent.parent.parent
    narrator_dir = base / "Narrator"
    voices_path = narrator_dir / "voices" / "voices.json"
    model_path = base / "models" / "VoxCPM2"
    output_dir = narrator_dir / "experiments" / "emotion_cfg_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse args
    target_role = None
    cfg_values = DEFAULT_CFG_VALUES

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--cfg":
            cfg_values = [float(x) for x in args[i + 1].split(",")]
            i += 2
        else:
            target_role = args[i]
            i += 1

    # Load voices
    with open(voices_path, "r", encoding="utf-8") as f:
        voices = json.load(f)

    # Filter roles with seed_audio
    roles = []
    for role, profile in voices.items():
        seed = profile.get("seed_audio")
        if not seed:
            continue
        if target_role and role != target_role:
            continue
        # Resolve relative path
        seed_path = Path(seed)
        if not seed_path.is_absolute():
            seed_path = (narrator_dir / seed_path).resolve()
        if not seed_path.exists():
            print(f"[WARN] Seed audio not found for {role}: {seed_path}")
            continue
        roles.append((role, str(seed_path)))

    if not roles:
        print("No roles with seed_audio found. Run seed_generator.py first.")
        return

    # Load model
    from voxcpm import VoxCPM

    print(f"Loading model from {model_path} ...")
    model = VoxCPM.from_pretrained(str(model_path), load_denoiser=False)
    print("Model loaded.\n")

    # Run experiments
    for role, seed_path in roles:
        print(f"{'=' * 50}")
        print(f"Role: {role}")
        print(f"Seed: {seed_path}")
        print(f"{'=' * 50}")

        for cfg in cfg_values:
            for emotion in EMOTIONS:
                prompt = f"({emotion}){TEST_TEXT}"
                label = f"{role}_cfg{cfg}_{emotion}"

                print(f"  [{label}] generating ...", end="", flush=True)
                t0 = time.time()
                wav = model.generate(
                    text=prompt,
                    reference_wav_path=seed_path,
                    cfg_value=cfg,
                    inference_timesteps=10,
                )
                elapsed = time.time() - t0
                duration = len(wav) / SAMPLE_RATE

                out_path = output_dir / f"{label}.wav"
                sf.write(str(out_path), wav, SAMPLE_RATE)
                print(f" {duration:.1f}s audio, {elapsed:.1f}s gen")

        # Also generate a neutral baseline (no emotion prefix)
        for cfg in cfg_values:
            label = f"{role}_cfg{cfg}_neutral_no_prefix"
            print(f"  [{label}] generating ...", end="", flush=True)
            t0 = time.time()
            wav = model.generate(
                text=TEST_TEXT,
                reference_wav_path=seed_path,
                cfg_value=cfg,
                inference_timesteps=10,
            )
            elapsed = time.time() - t0
            duration = len(wav) / SAMPLE_RATE

            out_path = output_dir / f"{label}.wav"
            sf.write(str(out_path), wav, SAMPLE_RATE)
            print(f" {duration:.1f}s audio, {elapsed:.1f}s gen")

        print()

    print(f"\nAll results saved to: {output_dir}")
    print(f"Total files: {len(list(output_dir.glob('*.wav')))}")
    print()
    print("Listen and compare:")
    print("  1. Same cfg, different emotions -> does emotion change?")
    print("  2. Same emotion, different cfg -> does cfg affect emotion strength?")
    print("  3. Compare with neutral_no_prefix -> is prefix doing anything?")
    print()
    print("Record findings in Narrator/notes/phase2_emotion_results.md")


if __name__ == "__main__":
    main()
