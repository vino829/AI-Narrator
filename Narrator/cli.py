"""
AI Narrator - CLI entry point.
Provides subcommands: generate, seed, test.
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_config(config_path: str | None = None) -> dict:
    """Load config.yaml and return as dict. Falls back to defaults if not found."""
    from narrator import DEFAULT_CONFIG

    if config_path is None:
        config_path = str(Path(__file__).parent / "config.yaml")

    path = Path(config_path)
    if not path.exists():
        return DEFAULT_CONFIG

    with open(path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}

    # Deep-merge user config over defaults
    merged = {}
    for section, defaults in DEFAULT_CONFIG.items():
        merged[section] = {**defaults, **user_cfg.get(section, {})}

    # Carry over top-level keys not in DEFAULT_CONFIG (e.g. "model")
    for key in user_cfg:
        if key not in merged:
            merged[key] = user_cfg[key]

    return merged


def resolve_paths(args, config: dict) -> dict:
    """Resolve default paths relative to Narrator/ directory."""
    narrator_dir = Path(__file__).resolve().parent
    base_dir = narrator_dir.parent

    if not args.voices:
        args.voices = str(narrator_dir / "voices" / "voices.json")
    if hasattr(args, "output_dir") and not args.output_dir:
        args.output_dir = str(narrator_dir / "output")

    # Model path from config or default
    model_path = config.get("model", {}).get("path", "./models/VoxCPM2")
    if not Path(model_path).is_absolute():
        model_path = str((base_dir / model_path).resolve())

    return model_path


# -- Subcommand handlers --

def cmd_generate(args):
    """Handle the 'generate' subcommand."""
    config = load_config(args.config)
    model_path = resolve_paths(args, config)

    # CLI overrides
    if args.format:
        config.setdefault("output", {})["format"] = args.format
    output_format = config.get("output", {}).get("format", "both")

    from narrator import synthesize

    synthesize(
        input_path=args.input,
        voices_path=args.voices,
        output_dir=args.output_dir,
        model_path=model_path,
        config=config,
        force=args.force,
        output_format=output_format,
    )


def cmd_seed(args):
    """Handle the 'seed' subcommand."""
    config = load_config(args.config)
    model_path = resolve_paths(args, config)

    narrator_dir = Path(__file__).resolve().parent

    if not args.output_dir:
        args.output_dir = str(narrator_dir / "voices")

    from seed_generator import generate_candidates, select_seeds, load_model

    if args.select:
        select_seeds(args.voices, args.output_dir)
    else:
        model = load_model(model_path)
        generate_candidates(
            model,
            voices_path=args.voices,
            output_dir=args.output_dir,
            model_path=model_path,
            num_candidates=args.candidates,
        )


def cmd_test(args):
    """Handle the 'test' subcommand."""
    config = load_config(args.config)
    narrator_dir = Path(__file__).resolve().parent
    base_dir = narrator_dir.parent

    model_path = config.get("model", {}).get("path", "./models/VoxCPM2")
    if not Path(model_path).is_absolute():
        model_path = str((base_dir / model_path).resolve())

    from narrator import load_model

    model = load_model(model_path)

    # Build text prompt
    text = args.text
    if args.voice_desc:
        text = f"({args.voice_desc}){text}"

    kwargs = {
        "text": text,
        "cfg_value": config["generation"]["cfg_value"],
        "inference_timesteps": config["generation"]["inference_timesteps"],
    }
    if args.reference:
        kwargs["reference_wav_path"] = args.reference

    import time
    import soundfile as sf

    print(f"Generating: {text[:60]}...")
    t0 = time.time()
    wav = model.generate(**kwargs)
    elapsed = time.time() - t0

    sample_rate = config["output"]["sample_rate"]
    duration = len(wav) / sample_rate

    output_path = args.output or "test.wav"
    sf.write(output_path, wav, sample_rate)
    print(f"Saved: {output_path} ({duration:.1f}s audio, {elapsed:.1f}s gen, RTF={elapsed/duration:.2f})")


# -- Argument parser --

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="AI Narrator - VoxCPM2-based multi-character audiobook generator",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")

    sub = parser.add_subparsers(dest="command", required=True)

    # -- generate --
    p_gen = sub.add_parser("generate", help="Generate audiobook from annotated JSON")
    p_gen.add_argument("-i", "--input", required=True, help="Input JSON file path")
    p_gen.add_argument("-v", "--voices", default="", help="voices.json path (default: voices/voices.json)")
    p_gen.add_argument("-o", "--output-dir", default="", help="Output directory (default: output/)")
    p_gen.add_argument(
        "-f", "--format",
        choices=["wav", "segments", "both"],
        default=None,
        help="Output format (default: from config.yaml)",
    )
    p_gen.add_argument("--force", action="store_true", help="Force regenerate all segments")
    p_gen.set_defaults(func=cmd_generate)

    # -- seed --
    p_seed = sub.add_parser("seed", help="Generate or select seed audios for characters")
    p_seed.add_argument("-v", "--voices", default="", help="voices.json path")
    p_seed.add_argument("-o", "--output-dir", default="", help="Output directory for seed files")
    p_seed.add_argument("-n", "--candidates", type=int, default=3, help="Number of candidates per role (default: 3)")
    p_seed.add_argument("--select", action="store_true", help="Enter interactive selection mode")
    p_seed.set_defaults(func=cmd_seed)

    # -- test --
    p_test = sub.add_parser("test", help="Quick single-sentence TTS test")
    p_test.add_argument("-t", "--text", required=True, help="Text to synthesize")
    p_test.add_argument("-d", "--voice-desc", default=None, help="Voice description, e.g. '年輕女性'")
    p_test.add_argument("-r", "--reference", default=None, help="Reference audio path for cloning")
    p_test.add_argument("-o", "--output", default=None, help="Output file path (default: test.wav)")
    p_test.set_defaults(func=cmd_test)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
