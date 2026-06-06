"""
AI Narrator - Core synthesis script.
Reads annotated JSON input and voice profiles, generates per-segment audio,
and concatenates them into a full chapter audiobook.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SAMPLE_RATE = 48000

# Silence durations in milliseconds
SILENCE_SAME_ROLE = 300
SILENCE_ROLE_SWITCH = 500
SILENCE_NARRATION_SWITCH = 600
SILENCE_PARAGRAPH_END = 800


def load_model(model_path: str):
    """Load VoxCPM2 model."""
    from voxcpm import VoxCPM

    log.info("Loading model from %s ...", model_path)
    t0 = time.time()
    model = VoxCPM.from_pretrained(model_path, load_denoiser=False)
    log.info("Model loaded in %.1fs", time.time() - t0)
    return model


def load_input(input_path: str) -> dict:
    """Load and validate novel input JSON."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "segments" not in data:
        raise ValueError(f"Input JSON missing 'segments' key: {input_path}")

    for i, seg in enumerate(data["segments"]):
        if "role" not in seg:
            raise ValueError(f"Segment {i} missing 'role'")
        if "text" not in seg:
            raise ValueError(f"Segment {i} missing 'text'")

    log.info(
        "Loaded input: %s - %s (%d segments)",
        data.get("title", "untitled"),
        data.get("chapter", ""),
        len(data["segments"]),
    )
    return data


def load_voices(voices_path: str) -> dict:
    """Load voice profiles. Resolves relative seed_audio paths against
    the parent directory of the voices file (i.e. Narrator/)."""
    voices_file = Path(voices_path)
    base_dir = voices_file.parent.parent  # voices.json is in voices/, base is Narrator/

    with open(voices_path, "r", encoding="utf-8") as f:
        voices = json.load(f)

    # Resolve relative seed_audio paths to absolute
    for role, profile in voices.items():
        seed = profile.get("seed_audio")
        if seed:
            seed_path = Path(seed)
            if not seed_path.is_absolute():
                profile["seed_audio"] = str((base_dir / seed_path).resolve())

    log.info("Loaded %d voice profiles", len(voices))
    return voices


def build_text(segment: dict, voice: dict | None) -> str:
    """Build the text prompt for TTS generation.

    - Voice Design mode (no seed_audio): prepend voice description + emotion
    - Controllable Cloning mode (has seed_audio): prepend emotion only
    """
    text = segment["text"]
    emotion = segment.get("emotion", "")
    has_seed = voice and voice.get("seed_audio")

    if has_seed:
        # Cloning mode: emotion prefix only
        if emotion:
            return f"({emotion}){text}"
        return text
    else:
        # Voice Design mode: description + emotion
        desc = voice["description"] if voice else ""
        if emotion:
            return f"({desc}，{emotion}){text}"
        if desc:
            return f"({desc}){text}"
        return text


def generate_segment(model, segment: dict, voice: dict | None) -> np.ndarray:
    """Generate audio for a single segment."""
    prompt = build_text(segment, voice)
    seed_audio = voice.get("seed_audio") if voice else None
    cfg = voice.get("cfg_value", 2.0) if voice else 2.0

    kwargs = {
        "text": prompt,
        "cfg_value": cfg,
        "inference_timesteps": 10,
    }
    if seed_audio:
        kwargs["reference_wav_path"] = seed_audio

    return model.generate(**kwargs)


def get_silence_duration(current: dict, next_seg: dict | None) -> int:
    """Determine silence duration (ms) between current and next segment."""
    if next_seg is None:
        return 0

    curr_role = current["role"]
    next_role = next_seg["role"]
    curr_is_narrator = curr_role == "narrator"
    next_is_narrator = next_role == "narrator"

    # Paragraph end: current ends with period and next is narrator
    curr_text = current["text"].strip()
    if curr_text.endswith(("。", ".", "」")) and next_is_narrator:
        return SILENCE_PARAGRAPH_END

    # Same role continues
    if curr_role == next_role:
        return SILENCE_SAME_ROLE

    # Narration <-> dialogue switch
    if curr_is_narrator or next_is_narrator:
        return SILENCE_NARRATION_SWITCH

    # Dialogue to dialogue (different roles)
    return SILENCE_ROLE_SWITCH


def make_silence(duration_ms: int) -> np.ndarray:
    """Create a silence array of given duration."""
    num_samples = int(SAMPLE_RATE * duration_ms / 1000)
    return np.zeros(num_samples, dtype=np.float32)


def synthesize(
    input_path: str,
    voices_path: str,
    output_dir: str,
    model_path: str = "./models/VoxCPM2",
):
    """Main synthesis pipeline."""
    output_dir = Path(output_dir)
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    # Load everything
    model = load_model(model_path)
    data = load_input(input_path)
    voices = load_voices(voices_path)
    segments = data["segments"]

    total = len(segments)
    success = 0
    failed = 0
    skipped = 0
    audio_chunks: list[np.ndarray] = []
    total_start = time.time()

    for i, seg in enumerate(segments):
        role = seg["role"]
        text = seg["text"].strip()
        idx = f"{i + 1:03d}"

        # Skip empty text
        if not text:
            log.warning("Segment %s (%s): empty text, skipping", idx, role)
            skipped += 1
            continue

        # Resolve voice profile
        voice = voices.get(role)
        if voice is None:
            log.warning(
                "Segment %s: role '%s' not in voices.json, using text-only generation",
                idx,
                role,
            )

        # Generate audio
        log.info("Segment %s/%03d [%s]: %s", idx, total, role, text[:40])
        t0 = time.time()
        try:
            wav = generate_segment(model, seg, voice)
            elapsed = time.time() - t0
            duration = len(wav) / SAMPLE_RATE
            log.info(
                "  -> %.1fs audio, generated in %.1fs (RTF=%.2f)",
                duration,
                elapsed,
                elapsed / duration if duration > 0 else 0,
            )
        except Exception:
            log.error("Segment %s (%s) generation failed:", idx, role, exc_info=True)
            # Insert 1s silence as placeholder
            wav = make_silence(1000)
            failed += 1
        else:
            success += 1

        # Save segment file
        seg_filename = f"{idx}_{role}.wav"
        seg_path = segments_dir / seg_filename
        sf.write(str(seg_path), wav, SAMPLE_RATE)

        # Accumulate audio
        audio_chunks.append(wav)

        # Insert silence before next segment
        next_seg = segments[i + 1] if i + 1 < total else None
        silence_ms = get_silence_duration(seg, next_seg)
        if silence_ms > 0:
            audio_chunks.append(make_silence(silence_ms))

    total_elapsed = time.time() - total_start

    # Concatenate and write full chapter
    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        title = data.get("title", "output")
        chapter = data.get("chapter", "")
        full_filename = f"{title}_{chapter}.wav" if chapter else f"{title}.wav"
        full_path = output_dir / full_filename
        sf.write(str(full_path), full_audio, SAMPLE_RATE)
        total_duration = len(full_audio) / SAMPLE_RATE
    else:
        total_duration = 0.0

    # Summary report
    minutes, secs = divmod(total_duration, 60)
    e_min, e_sec = divmod(total_elapsed, 60)
    rtf = total_elapsed / total_duration if total_duration > 0 else 0

    log.info("=" * 50)
    log.info("Generation complete")
    log.info("  Total segments : %d", total)
    log.info("  Success        : %d", success)
    log.info("  Failed         : %d", failed)
    log.info("  Skipped (empty): %d", skipped)
    log.info("  Audio duration : %d:%04.1f", minutes, secs)
    log.info("  Wall time      : %d:%04.1f", e_min, e_sec)
    log.info("  RTF            : %.2f", rtf)
    log.info("  Output         : %s", output_dir)
    log.info("=" * 50)


if __name__ == "__main__":
    import sys

    base = Path(__file__).resolve().parent.parent

    input_file = sys.argv[1] if len(sys.argv) > 1 else str(base / "Narrator" / "input" / "example.json")
    voices_file = sys.argv[2] if len(sys.argv) > 2 else str(base / "Narrator" / "voices" / "voices.json")
    output_path = sys.argv[3] if len(sys.argv) > 3 else str(base / "Narrator" / "output")
    model_dir = sys.argv[4] if len(sys.argv) > 4 else str(base / "models" / "VoxCPM2")

    synthesize(input_file, voices_file, output_path, model_dir)
