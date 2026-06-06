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

# Defaults (overridden by config.yaml)
DEFAULT_CONFIG = {
    "generation": {
        "cfg_value": 2.0,
        "inference_timesteps": 10,
    },
    "silence": {
        "same_role": 300,
        "role_switch": 500,
        "narration_switch": 600,
        "paragraph_end": 800,
    },
    "output": {
        "sample_rate": 48000,
    },
}


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


def generate_segment(model, segment: dict, voice: dict | None, config: dict) -> np.ndarray:
    """Generate audio for a single segment."""
    prompt = build_text(segment, voice)
    seed_audio = voice.get("seed_audio") if voice else None
    cfg = voice.get("cfg_value", config["generation"]["cfg_value"]) if voice else config["generation"]["cfg_value"]

    kwargs = {
        "text": prompt,
        "cfg_value": cfg,
        "inference_timesteps": config["generation"]["inference_timesteps"],
    }
    if seed_audio:
        kwargs["reference_wav_path"] = seed_audio

    return model.generate(**kwargs)


def get_silence_duration(current: dict, next_seg: dict | None, config: dict) -> int:
    """Determine silence duration (ms) between current and next segment."""
    if next_seg is None:
        return 0

    sil = config["silence"]
    curr_role = current["role"]
    next_role = next_seg["role"]
    curr_is_narrator = curr_role == "narrator"
    next_is_narrator = next_role == "narrator"

    # Paragraph end: current ends with period and next is narrator
    curr_text = current["text"].strip()
    if curr_text.endswith(("。", ".", "」")) and next_is_narrator:
        return sil["paragraph_end"]

    # Same role continues
    if curr_role == next_role:
        return sil["same_role"]

    # Narration <-> dialogue switch
    if curr_is_narrator or next_is_narrator:
        return sil["narration_switch"]

    # Dialogue to dialogue (different roles)
    return sil["role_switch"]


def make_silence(duration_ms: int, sample_rate: int) -> np.ndarray:
    """Create a silence array of given duration."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(num_samples, dtype=np.float32)


def _seg_filename(index: int, role: str) -> str:
    """Build segment filename: 001_narrator.wav, etc."""
    return f"{index + 1:03d}_{role}.wav"


def _find_existing_segments(segments_dir: Path, total: int, segments: list[dict]) -> dict[int, Path]:
    """Scan segments_dir for already-generated segment files.
    Returns a dict mapping segment index -> file path."""
    existing = {}
    for i, seg in enumerate(segments):
        fname = _seg_filename(i, seg["role"])
        fpath = segments_dir / fname
        if fpath.exists():
            existing[i] = fpath
    return existing


def synthesize(
    input_path: str,
    voices_path: str,
    output_dir: str,
    model_path: str = "./models/VoxCPM2",
    config: dict | None = None,
    force: bool = False,
    output_format: str = "both",
):
    """Main synthesis pipeline.

    Args:
        input_path: Path to novel input JSON.
        voices_path: Path to voices.json.
        output_dir: Output directory.
        model_path: Path to VoxCPM2 model.
        config: Config dict (from config.yaml). Uses DEFAULT_CONFIG if None.
        force: If True, regenerate all segments even if files exist.
        output_format: "wav" (full only), "segments" (segments only), "both".
    """
    if config is None:
        config = DEFAULT_CONFIG

    sample_rate = config["output"]["sample_rate"]
    output_dir = Path(output_dir)
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    # Load everything
    model = load_model(model_path)
    data = load_input(input_path)
    voices = load_voices(voices_path)
    segments = data["segments"]

    total = len(segments)

    # Check for existing segments (resume support)
    if force:
        existing = {}
    else:
        existing = _find_existing_segments(segments_dir, total, segments)
        if existing:
            log.info("Found %d existing segment(s), will skip (use --force to regenerate)", len(existing))

    success = 0
    failed = 0
    skipped_empty = 0
    skipped_existing = 0
    audio_chunks: list[np.ndarray] = []
    total_start = time.time()

    # Try to use tqdm for progress display
    try:
        from tqdm import tqdm
        progress = tqdm(enumerate(segments), total=total, desc="Generating", unit="seg")
    except ImportError:
        log.warning("tqdm not installed, progress bar disabled (pip install tqdm)")
        progress = enumerate(segments)

    for i, seg in progress:
        role = seg["role"]
        text = seg["text"].strip()
        idx = f"{i + 1:03d}"

        # Update progress bar description
        if hasattr(progress, "set_postfix"):
            progress.set_postfix_str(role, refresh=True)

        # Skip empty text
        if not text:
            log.warning("Segment %s (%s): empty text, skipping", idx, role)
            skipped_empty += 1
            continue

        # Resume: load existing segment instead of regenerating
        if i in existing:
            log.info("Segment %s/%03d [%s]: exists, loading", idx, total, role)
            wav, sr = sf.read(str(existing[i]), dtype="float32")
            audio_chunks.append(wav)
            skipped_existing += 1

            # Insert silence before next segment
            next_seg = segments[i + 1] if i + 1 < total else None
            silence_ms = get_silence_duration(seg, next_seg, config)
            if silence_ms > 0:
                audio_chunks.append(make_silence(silence_ms, sample_rate))
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
            wav = generate_segment(model, seg, voice, config)
            elapsed = time.time() - t0
            duration = len(wav) / sample_rate
            log.info(
                "  -> %.1fs audio, generated in %.1fs (RTF=%.2f)",
                duration,
                elapsed,
                elapsed / duration if duration > 0 else 0,
            )
        except Exception:
            log.error("Segment %s (%s) generation failed:", idx, role, exc_info=True)
            # Insert 1s silence as placeholder
            wav = make_silence(1000, sample_rate)
            failed += 1
        else:
            success += 1

        # Save segment file
        seg_filename = _seg_filename(i, role)
        seg_path = segments_dir / seg_filename
        sf.write(str(seg_path), wav, sample_rate)

        # Accumulate audio
        audio_chunks.append(wav)

        # Insert silence before next segment
        next_seg = segments[i + 1] if i + 1 < total else None
        silence_ms = get_silence_duration(seg, next_seg, config)
        if silence_ms > 0:
            audio_chunks.append(make_silence(silence_ms, sample_rate))

    total_elapsed = time.time() - total_start

    # Concatenate and write full chapter
    full_path = None
    total_duration = 0.0
    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        total_duration = len(full_audio) / sample_rate

        if output_format in ("wav", "both"):
            title = data.get("title", "output")
            chapter = data.get("chapter", "")
            full_filename = f"{title}_{chapter}.wav" if chapter else f"{title}.wav"
            full_path = output_dir / full_filename
            sf.write(str(full_path), full_audio, sample_rate)

    # Clean up segments if format is "wav" only
    if output_format == "wav" and segments_dir.exists():
        import shutil
        shutil.rmtree(segments_dir)

    # Clean up full file if format is "segments" only
    # (full file was not written in this case, nothing to clean)

    # Summary report
    minutes, secs = divmod(total_duration, 60)
    e_min, e_sec = divmod(total_elapsed, 60)
    rtf = total_elapsed / total_duration if total_duration > 0 else 0

    log.info("=" * 50)
    log.info("生成完成")
    log.info("  總段數         : %d", total)
    log.info("  成功           : %d", success)
    log.info("  失敗           : %d", failed)
    log.info("  跳過（空文本）  : %d", skipped_empty)
    log.info("  跳過（已存在）  : %d", skipped_existing)
    log.info("  總音頻時長      : %d:%04.1f", minutes, secs)
    log.info("  總生成耗時      : %d:%04.1f", e_min, e_sec)
    log.info("  RTF            : %.2f", rtf)
    if full_path:
        log.info("  輸出           : %s", full_path)
    else:
        log.info("  輸出           : %s", segments_dir)
    log.info("=" * 50)


if __name__ == "__main__":
    import sys

    base = Path(__file__).resolve().parent.parent

    input_file = sys.argv[1] if len(sys.argv) > 1 else str(base / "Narrator" / "input" / "example.json")
    voices_file = sys.argv[2] if len(sys.argv) > 2 else str(base / "Narrator" / "voices" / "voices.json")
    output_path = sys.argv[3] if len(sys.argv) > 3 else str(base / "Narrator" / "output")
    model_dir = sys.argv[4] if len(sys.argv) > 4 else str(base / "models" / "VoxCPM2")

    synthesize(input_file, voices_file, output_path, model_dir)
