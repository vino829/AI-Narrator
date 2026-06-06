"""
AI Narrator - Seed audio generator.
Generates candidate seed audios for each character using Voice Design mode,
then lets the user select the best candidate and registers it in voices.json.
"""

import json
import logging
import sys
import time
from pathlib import Path

import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SAMPLE_RATE = 48000
NUM_CANDIDATES = 3

# Audition texts designed to showcase each character's vocal traits.
# Narrators get a descriptive passage; dialogue characters get a mix of
# declarative, interrogative, and exclamatory sentences (~15-30s of audio).
AUDITION_TEXTS = {
    "narrator": (
        "夜幕低垂，城市的喧囂逐漸褪去。街角的路燈投下昏黃的光暈，"
        "照亮了被雨水打濕的石板路。遠處傳來幾聲犬吠，"
        "隨後一切又歸於沉寂。在這樣的夜裡，總有些不為人知的故事，"
        "悄悄地在某個角落上演著。"
    ),
    "李明": (
        "我一直在想，這一切到底值不值得。那些年我們走過的路，"
        "真的有意義嗎？也許吧，至少我不後悔。"
        "可是為什麼每次想起來，心裡還是會隱隱作痛呢？"
        "算了，不想了！明天的事情明天再說吧。"
    ),
    "王芳": (
        "你聽我說，這件事情沒有你想的那麼簡單！"
        "我昨天去問過了，他們根本不同意我們的方案。"
        "那怎麼辦？難道就這樣放棄嗎？"
        "不行，我絕對不會認輸的！我們再想想別的辦法。"
    ),
    "老陳": (
        "年輕人，有些事情急不來的。我在這一行幹了三十多年，"
        "什麼大風大浪沒見過？你以為光靠一腔熱血就能成事嗎？"
        "不是那麼回事。慢慢來，把基礎打好，機會自然會來的。"
        "唉，要是當年有人跟我說這些，我也不至於走那麼多彎路啊。"
    ),
}

# Fallback audition text for roles not in the above dict.
DEFAULT_AUDITION_TEXT = (
    "這是一段語音測試。今天天氣真不錯，你覺得呢？"
    "我想說的是，無論發生什麼事，我們都要堅持下去！"
    "畢竟，生活總是充滿了各種各樣的可能性。"
)


def load_model(model_path: str):
    """Load VoxCPM2 model."""
    from voxcpm import VoxCPM

    log.info("Loading model from %s ...", model_path)
    t0 = time.time()
    model = VoxCPM.from_pretrained(model_path, load_denoiser=False)
    log.info("Model loaded in %.1fs", time.time() - t0)
    return model


def generate_candidates(
    model,
    voices_path: str,
    output_dir: str,
    model_path: str = "./models/VoxCPM2",
    num_candidates: int = NUM_CANDIDATES,
):
    """Generate candidate seed audios for roles missing seed_audio."""
    voices_path = Path(voices_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(voices_path, "r", encoding="utf-8") as f:
        voices = json.load(f)

    # Find roles that need seeds
    roles_to_generate = [
        (role, profile)
        for role, profile in voices.items()
        if not profile.get("seed_audio")
    ]

    if not roles_to_generate:
        log.info("All roles already have seed_audio. Nothing to generate.")
        return

    log.info(
        "Generating candidates for %d role(s): %s",
        len(roles_to_generate),
        ", ".join(r for r, _ in roles_to_generate),
    )

    for role, profile in roles_to_generate:
        desc = profile.get("description", "")
        text_body = AUDITION_TEXTS.get(role, DEFAULT_AUDITION_TEXT)
        prompt = f"({desc}){text_body}" if desc else text_body

        log.info("--- Role: %s ---", role)
        log.info("  Description: %s", desc)
        log.info("  Text: %s", text_body[:60] + "...")

        for i in range(num_candidates):
            log.info("  Generating candidate %d/%d ...", i + 1, num_candidates)
            t0 = time.time()
            wav = model.generate(
                text=prompt,
                cfg_value=2.0,
                inference_timesteps=10,
            )
            elapsed = time.time() - t0
            duration = len(wav) / SAMPLE_RATE

            out_path = output_dir / f"{role}_candidate_{i}.wav"
            sf.write(str(out_path), wav, SAMPLE_RATE)
            log.info(
                "  Saved: %s (%.1fs audio, %.1fs gen, RTF=%.2f)",
                out_path.name,
                duration,
                elapsed,
                elapsed / duration if duration > 0 else 0,
            )

    print()
    print("=" * 50)
    print("Candidate generation complete!")
    print(f"Output directory: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Listen to the candidate files")
    print("  2. Run this script with --select to choose the best candidate")
    print(f"     python {Path(__file__).name} --select")
    print("=" * 50)


def select_seeds(voices_path: str, candidates_dir: str):
    """Interactive CLI to select the best candidate for each role."""
    voices_path = Path(voices_path)
    candidates_dir = Path(candidates_dir)

    with open(voices_path, "r", encoding="utf-8") as f:
        voices = json.load(f)

    roles_to_select = [
        role for role, profile in voices.items() if not profile.get("seed_audio")
    ]

    if not roles_to_select:
        print("All roles already have seed_audio assigned.")
        return

    print()
    print("=" * 50)
    print("Seed Audio Selection")
    print("=" * 50)

    updated = False

    for role in roles_to_select:
        # Find candidate files for this role
        candidates = sorted(candidates_dir.glob(f"{role}_candidate_*.wav"))
        if not candidates:
            print(f"\n[{role}] No candidate files found, skipping.")
            continue

        print(f"\n--- {role} ---")
        print(f"  Description: {voices[role].get('description', 'N/A')}")
        print(f"  Candidates:")
        for j, c in enumerate(candidates):
            # Show file size as a rough indicator
            size_kb = c.stat().st_size / 1024
            print(f"    {j}: {c.name} ({size_kb:.0f} KB)")

        print()
        while True:
            choice = input(
                f"  Select best candidate for [{role}] (0-{len(candidates)-1}), "
                f"or 's' to skip: "
            ).strip()

            if choice.lower() == "s":
                print(f"  Skipped {role}.")
                break

            try:
                idx = int(choice)
                if 0 <= idx < len(candidates):
                    # Copy candidate to final seed file
                    seed_path = candidates_dir / f"{role}.wav"
                    import shutil

                    shutil.copy2(candidates[idx], seed_path)

                    # Update voices.json with relative path
                    rel_path = f"voices/{role}.wav"
                    voices[role]["seed_audio"] = rel_path

                    print(f"  Selected candidate {idx} -> {seed_path.name}")
                    print(f"  Updated voices.json: seed_audio = \"{rel_path}\"")
                    updated = True
                    break
                else:
                    print(f"  Invalid index. Enter 0-{len(candidates)-1} or 's'.")
            except ValueError:
                print(f"  Invalid input. Enter 0-{len(candidates)-1} or 's'.")

    if updated:
        with open(voices_path, "w", encoding="utf-8") as f:
            json.dump(voices, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"\nvoices.json updated: {voices_path}")
    else:
        print("\nNo changes made.")

    print()


def main():
    base = Path(__file__).resolve().parent.parent
    narrator_dir = base / "Narrator"
    voices_path = narrator_dir / "voices" / "voices.json"
    voices_dir = narrator_dir / "voices"
    model_path = base / "models" / "VoxCPM2"

    if "--select" in sys.argv:
        select_seeds(str(voices_path), str(voices_dir))
    else:
        model = load_model(str(model_path))
        generate_candidates(
            model,
            voices_path=str(voices_path),
            output_dir=str(voices_dir),
            model_path=str(model_path),
        )


if __name__ == "__main__":
    main()
