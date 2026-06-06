"""
Phase 3 automated tests.
Mocks VoxCPM model so tests run without GPU or model weights.

Usage:
    cd Narrator/
    python -m pytest test_phase3.py -v
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
import yaml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_RATE = 48000


def _make_fake_wav(duration_s: float = 0.5) -> np.ndarray:
    """Generate a short sine wave for testing."""
    t = np.linspace(0, duration_s, int(SAMPLE_RATE * duration_s), dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def fake_model():
    """Return a mock VoxCPM model whose .generate() returns a short sine wave."""
    model = MagicMock()
    model.generate.side_effect = lambda **kw: _make_fake_wav(0.5)
    model.tts_model = MagicMock()
    model.tts_model.sample_rate = SAMPLE_RATE
    return model


@pytest.fixture
def tmp_workspace(tmp_path):
    """Create a temporary workspace with input JSON, voices.json, and seed wavs."""
    # voices dir
    voices_dir = tmp_path / "voices"
    voices_dir.mkdir()

    # Write a seed wav for narrator
    seed_wav = voices_dir / "narrator.wav"
    sf.write(str(seed_wav), _make_fake_wav(1.0), SAMPLE_RATE)

    # voices.json — narrator has seed, 角色A does not
    voices = {
        "narrator": {
            "description": "成熟男性",
            "seed_audio": f"voices/narrator.wav",
        },
        "角色A": {
            "description": "年輕女性",
            "seed_audio": None,
        },
    }
    voices_path = voices_dir / "voices.json"
    voices_path.write_text(json.dumps(voices, ensure_ascii=False, indent=2), encoding="utf-8")

    # input JSON
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    input_data = {
        "title": "測試書",
        "chapter": "第一章",
        "segments": [
            {"role": "narrator", "text": "旁白第一句。"},
            {"role": "角色A", "emotion": "happy", "text": "角色A的台詞！"},
            {"role": "narrator", "text": "旁白第二句。"},
            {"role": "角色A", "text": ""},  # empty — should be skipped
            {"role": "角色A", "text": "角色A第二句。"},
        ],
    }
    input_path = input_dir / "test_input.json"
    input_path.write_text(json.dumps(input_data, ensure_ascii=False, indent=2), encoding="utf-8")

    # output dir
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    # config.yaml
    config = {
        "model": {"path": "./models/VoxCPM2", "load_denoiser": False},
        "generation": {"cfg_value": 2.0, "inference_timesteps": 10},
        "silence": {
            "same_role": 300,
            "role_switch": 500,
            "narration_switch": 600,
            "paragraph_end": 800,
        },
        "output": {"format": "both", "sample_rate": 48000},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config, allow_unicode=True), encoding="utf-8")

    return tmp_path


# ---------------------------------------------------------------------------
# 1. Config loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_load_config_from_yaml(self, tmp_workspace):
        """config.yaml values are correctly loaded and merged with defaults."""
        sys.path.insert(0, str(Path(__file__).parent))
        from cli import load_config
        from narrator import DEFAULT_CONFIG

        config = load_config(str(tmp_workspace / "config.yaml"))

        assert config["generation"]["cfg_value"] == 2.0
        assert config["generation"]["inference_timesteps"] == 10
        assert config["silence"]["same_role"] == 300
        assert config["output"]["sample_rate"] == 48000
        # model section carried over (not in DEFAULT_CONFIG)
        assert config["model"]["path"] == "./models/VoxCPM2"

    def test_load_config_missing_file(self):
        """Falls back to DEFAULT_CONFIG when file doesn't exist."""
        from cli import load_config
        from narrator import DEFAULT_CONFIG

        config = load_config("/nonexistent/config.yaml")
        assert config == DEFAULT_CONFIG

    def test_partial_config_merged(self, tmp_path):
        """Partial config.yaml merges with defaults for missing keys."""
        partial = {"generation": {"cfg_value": 3.5}}
        cfg_path = tmp_path / "partial.yaml"
        cfg_path.write_text(yaml.dump(partial), encoding="utf-8")

        from cli import load_config

        config = load_config(str(cfg_path))
        assert config["generation"]["cfg_value"] == 3.5
        # inference_timesteps should come from defaults
        assert config["generation"]["inference_timesteps"] == 10
        # silence section should come from defaults entirely
        assert config["silence"]["same_role"] == 300


# ---------------------------------------------------------------------------
# 2. CLI argument parsing
# ---------------------------------------------------------------------------

class TestCLIParsing:
    def test_generate_args(self):
        from cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "generate", "-i", "input.json", "-v", "voices.json",
            "-o", "out/", "-f", "segments", "--force",
        ])
        assert args.command == "generate"
        assert args.input == "input.json"
        assert args.voices == "voices.json"
        assert args.output_dir == "out/"
        assert args.format == "segments"
        assert args.force is True

    def test_seed_args(self):
        from cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["seed", "-n", "5", "--select"])
        assert args.command == "seed"
        assert args.candidates == 5
        assert args.select is True

    def test_test_args(self):
        from cli import build_parser

        parser = build_parser()
        args = parser.parse_args([
            "test", "-t", "你好", "-d", "年輕女性", "-r", "ref.wav", "-o", "out.wav",
        ])
        assert args.command == "test"
        assert args.text == "你好"
        assert args.voice_desc == "年輕女性"
        assert args.reference == "ref.wav"
        assert args.output == "out.wav"

    def test_missing_subcommand_fails(self):
        from cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_generate_missing_input_fails(self):
        from cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["generate"])

    def test_format_invalid_choice_fails(self):
        from cli import build_parser

        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["generate", "-i", "x.json", "-f", "mp3"])


# ---------------------------------------------------------------------------
# 3. build_text logic
# ---------------------------------------------------------------------------

class TestBuildText:
    def test_cloning_mode_with_emotion(self):
        from narrator import build_text

        seg = {"text": "你好", "emotion": "angry"}
        voice = {"description": "年輕女性", "seed_audio": "ref.wav"}
        assert build_text(seg, voice) == "(angry)你好"

    def test_cloning_mode_no_emotion(self):
        from narrator import build_text

        seg = {"text": "你好"}
        voice = {"description": "年輕女性", "seed_audio": "ref.wav"}
        assert build_text(seg, voice) == "你好"

    def test_voice_design_with_emotion(self):
        from narrator import build_text

        seg = {"text": "你好", "emotion": "happy"}
        voice = {"description": "年輕女性", "seed_audio": None}
        assert build_text(seg, voice) == "(年輕女性，happy)你好"

    def test_voice_design_no_emotion(self):
        from narrator import build_text

        seg = {"text": "你好"}
        voice = {"description": "年輕女性", "seed_audio": None}
        assert build_text(seg, voice) == "(年輕女性)你好"

    def test_no_voice_profile(self):
        from narrator import build_text

        seg = {"text": "你好"}
        assert build_text(seg, None) == "你好"


# ---------------------------------------------------------------------------
# 4. Silence duration logic
# ---------------------------------------------------------------------------

class TestSilenceDuration:
    def setup_method(self):
        from narrator import DEFAULT_CONFIG
        self.config = DEFAULT_CONFIG

    def test_no_next_segment(self):
        from narrator import get_silence_duration

        seg = {"role": "narrator", "text": "句子。"}
        assert get_silence_duration(seg, None, self.config) == 0

    def test_same_role(self):
        from narrator import get_silence_duration

        seg = {"role": "角色A", "text": "句子"}
        nxt = {"role": "角色A", "text": "下一句"}
        assert get_silence_duration(seg, nxt, self.config) == 300

    def test_role_switch_dialogue(self):
        from narrator import get_silence_duration

        seg = {"role": "角色A", "text": "句子"}
        nxt = {"role": "角色B", "text": "句子"}
        assert get_silence_duration(seg, nxt, self.config) == 500

    def test_narration_switch(self):
        from narrator import get_silence_duration

        seg = {"role": "narrator", "text": "句子"}
        nxt = {"role": "角色A", "text": "句子"}
        assert get_silence_duration(seg, nxt, self.config) == 600

    def test_paragraph_end(self):
        from narrator import get_silence_duration

        seg = {"role": "角色A", "text": "結尾。"}
        nxt = {"role": "narrator", "text": "下一段"}
        assert get_silence_duration(seg, nxt, self.config) == 800

    def test_paragraph_end_with_quote(self):
        from narrator import get_silence_duration

        seg = {"role": "角色A", "text": "說完了」"}
        nxt = {"role": "narrator", "text": "下一段"}
        assert get_silence_duration(seg, nxt, self.config) == 800

    def test_custom_config_values(self):
        from narrator import get_silence_duration

        custom = {**self.config, "silence": {**self.config["silence"], "same_role": 999}}
        seg = {"role": "A", "text": "句子"}
        nxt = {"role": "A", "text": "句子"}
        assert get_silence_duration(seg, nxt, custom) == 999


# ---------------------------------------------------------------------------
# 5. make_silence
# ---------------------------------------------------------------------------

class TestMakeSilence:
    def test_duration_correct(self):
        from narrator import make_silence

        wav = make_silence(500, 48000)
        assert len(wav) == 24000  # 0.5s * 48000
        assert np.all(wav == 0)

    def test_zero_duration(self):
        from narrator import make_silence

        wav = make_silence(0, 48000)
        assert len(wav) == 0


# ---------------------------------------------------------------------------
# 6. Segment filename
# ---------------------------------------------------------------------------

class TestSegFilename:
    def test_basic(self):
        from narrator import _seg_filename

        assert _seg_filename(0, "narrator") == "001_narrator.wav"
        assert _seg_filename(9, "角色A") == "010_角色A.wav"
        assert _seg_filename(99, "x") == "100_x.wav"


# ---------------------------------------------------------------------------
# 7. Resume (checkpoint) logic
# ---------------------------------------------------------------------------

class TestResume:
    def test_find_existing_segments(self, tmp_path):
        from narrator import _find_existing_segments

        seg_dir = tmp_path / "segments"
        seg_dir.mkdir()

        segments = [
            {"role": "narrator", "text": "a"},
            {"role": "角色A", "text": "b"},
            {"role": "narrator", "text": "c"},
        ]

        # Only create file for segment 0
        sf.write(str(seg_dir / "001_narrator.wav"), _make_fake_wav(0.3), SAMPLE_RATE)

        existing = _find_existing_segments(seg_dir, 3, segments)
        assert 0 in existing
        assert 1 not in existing
        assert 2 not in existing

    def test_find_no_existing(self, tmp_path):
        from narrator import _find_existing_segments

        seg_dir = tmp_path / "segments"
        seg_dir.mkdir()
        segments = [{"role": "narrator", "text": "a"}]
        existing = _find_existing_segments(seg_dir, 1, segments)
        assert existing == {}


# ---------------------------------------------------------------------------
# 8. Full synthesize pipeline (mocked model)
# ---------------------------------------------------------------------------

class TestSynthesizePipeline:
    @patch("narrator.load_model")
    def test_basic_generation(self, mock_load_model, fake_model, tmp_workspace):
        """Full pipeline produces segment files and concatenated output."""
        mock_load_model.return_value = fake_model
        from narrator import synthesize, DEFAULT_CONFIG

        input_path = str(tmp_workspace / "input" / "test_input.json")
        voices_path = str(tmp_workspace / "voices" / "voices.json")
        output_dir = str(tmp_workspace / "output")

        synthesize(
            input_path=input_path,
            voices_path=voices_path,
            output_dir=output_dir,
            model_path="dummy",
            config=DEFAULT_CONFIG,
        )

        seg_dir = tmp_workspace / "output" / "segments"
        out_dir = tmp_workspace / "output"

        # 5 segments total, 1 empty → 4 segment files
        seg_files = sorted(seg_dir.glob("*.wav"))
        assert len(seg_files) == 4
        assert seg_files[0].name == "001_narrator.wav"
        assert seg_files[1].name == "002_角色A.wav"
        assert seg_files[2].name == "003_narrator.wav"
        # 004 is skipped (empty text)
        assert seg_files[3].name == "005_角色A.wav"

        # Full output file
        full_file = out_dir / "測試書_第一章.wav"
        assert full_file.exists()

        # Model was called 4 times (4 non-empty segments)
        assert fake_model.generate.call_count == 4

    @patch("narrator.load_model")
    def test_resume_skips_existing(self, mock_load_model, fake_model, tmp_workspace):
        """Segments that already exist on disk are skipped."""
        mock_load_model.return_value = fake_model
        from narrator import synthesize, DEFAULT_CONFIG

        input_path = str(tmp_workspace / "input" / "test_input.json")
        voices_path = str(tmp_workspace / "voices" / "voices.json")
        output_dir = str(tmp_workspace / "output")
        seg_dir = tmp_workspace / "output" / "segments"
        seg_dir.mkdir(parents=True, exist_ok=True)

        # Pre-create segment 0 and 1
        sf.write(str(seg_dir / "001_narrator.wav"), _make_fake_wav(0.5), SAMPLE_RATE)
        sf.write(str(seg_dir / "002_角色A.wav"), _make_fake_wav(0.5), SAMPLE_RATE)

        synthesize(
            input_path=input_path,
            voices_path=voices_path,
            output_dir=output_dir,
            model_path="dummy",
            config=DEFAULT_CONFIG,
            force=False,
        )

        # Only 2 new segments generated (seg 2 and 4, since seg 3 is empty)
        assert fake_model.generate.call_count == 2

    @patch("narrator.load_model")
    def test_force_regenerates_all(self, mock_load_model, fake_model, tmp_workspace):
        """--force ignores existing files and regenerates everything."""
        mock_load_model.return_value = fake_model
        from narrator import synthesize, DEFAULT_CONFIG

        input_path = str(tmp_workspace / "input" / "test_input.json")
        voices_path = str(tmp_workspace / "voices" / "voices.json")
        output_dir = str(tmp_workspace / "output")
        seg_dir = tmp_workspace / "output" / "segments"
        seg_dir.mkdir(parents=True, exist_ok=True)

        # Pre-create all segments
        sf.write(str(seg_dir / "001_narrator.wav"), _make_fake_wav(0.5), SAMPLE_RATE)
        sf.write(str(seg_dir / "002_角色A.wav"), _make_fake_wav(0.5), SAMPLE_RATE)
        sf.write(str(seg_dir / "003_narrator.wav"), _make_fake_wav(0.5), SAMPLE_RATE)
        sf.write(str(seg_dir / "005_角色A.wav"), _make_fake_wav(0.5), SAMPLE_RATE)

        synthesize(
            input_path=input_path,
            voices_path=voices_path,
            output_dir=output_dir,
            model_path="dummy",
            config=DEFAULT_CONFIG,
            force=True,
        )

        # All 4 non-empty segments regenerated
        assert fake_model.generate.call_count == 4

    @patch("narrator.load_model")
    def test_format_segments_only(self, mock_load_model, fake_model, tmp_workspace):
        """output_format='segments' produces segments but no full wav."""
        mock_load_model.return_value = fake_model
        from narrator import synthesize, DEFAULT_CONFIG

        input_path = str(tmp_workspace / "input" / "test_input.json")
        voices_path = str(tmp_workspace / "voices" / "voices.json")
        output_dir = str(tmp_workspace / "output")

        synthesize(
            input_path=input_path,
            voices_path=voices_path,
            output_dir=output_dir,
            model_path="dummy",
            config=DEFAULT_CONFIG,
            output_format="segments",
        )

        seg_dir = tmp_workspace / "output" / "segments"
        out_dir = tmp_workspace / "output"

        assert len(list(seg_dir.glob("*.wav"))) == 4
        assert not (out_dir / "測試書_第一章.wav").exists()

    @patch("narrator.load_model")
    def test_format_wav_only(self, mock_load_model, fake_model, tmp_workspace):
        """output_format='wav' produces full wav and removes segments dir."""
        mock_load_model.return_value = fake_model
        from narrator import synthesize, DEFAULT_CONFIG

        input_path = str(tmp_workspace / "input" / "test_input.json")
        voices_path = str(tmp_workspace / "voices" / "voices.json")
        output_dir = str(tmp_workspace / "output")

        synthesize(
            input_path=input_path,
            voices_path=voices_path,
            output_dir=output_dir,
            model_path="dummy",
            config=DEFAULT_CONFIG,
            output_format="wav",
        )

        out_dir = tmp_workspace / "output"
        seg_dir = out_dir / "segments"

        assert (out_dir / "測試書_第一章.wav").exists()
        assert not seg_dir.exists()

    @patch("narrator.load_model")
    def test_empty_text_skipped(self, mock_load_model, fake_model, tmp_workspace):
        """Segments with empty text are skipped without error."""
        mock_load_model.return_value = fake_model
        from narrator import synthesize, DEFAULT_CONFIG

        input_path = str(tmp_workspace / "input" / "test_input.json")
        voices_path = str(tmp_workspace / "voices" / "voices.json")
        output_dir = str(tmp_workspace / "output")

        # Should not raise
        synthesize(
            input_path=input_path,
            voices_path=voices_path,
            output_dir=output_dir,
            model_path="dummy",
            config=DEFAULT_CONFIG,
        )

        seg_dir = tmp_workspace / "output" / "segments"
        # Segment 004 (empty) should not exist
        assert not (seg_dir / "004_角色A.wav").exists()

    @patch("narrator.load_model")
    def test_generation_failure_inserts_silence(self, mock_load_model, tmp_workspace):
        """When model.generate() throws, a silence placeholder is inserted."""
        model = MagicMock()
        call_count = 0

        def side_effect(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail on second segment
                raise RuntimeError("GPU exploded")
            return _make_fake_wav(0.5)

        model.generate.side_effect = side_effect
        mock_load_model.return_value = model
        from narrator import synthesize, DEFAULT_CONFIG

        input_path = str(tmp_workspace / "input" / "test_input.json")
        voices_path = str(tmp_workspace / "voices" / "voices.json")
        output_dir = str(tmp_workspace / "output")

        # Should not raise
        synthesize(
            input_path=input_path,
            voices_path=voices_path,
            output_dir=output_dir,
            model_path="dummy",
            config=DEFAULT_CONFIG,
        )

        seg_dir = tmp_workspace / "output" / "segments"
        # Failed segment still has a file (silence placeholder)
        assert (seg_dir / "002_角色A.wav").exists()
        # Full output still created
        assert (tmp_workspace / "output" / "測試書_第一章.wav").exists()


# ---------------------------------------------------------------------------
# 9. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_missing_segments_key(self, tmp_path):
        from narrator import load_input

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"title": "x"}), encoding="utf-8")
        with pytest.raises(ValueError, match="missing 'segments'"):
            load_input(str(bad))

    def test_missing_role(self, tmp_path):
        from narrator import load_input

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"segments": [{"text": "hi"}]}), encoding="utf-8")
        with pytest.raises(ValueError, match="missing 'role'"):
            load_input(str(bad))

    def test_missing_text(self, tmp_path):
        from narrator import load_input

        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"segments": [{"role": "x"}]}), encoding="utf-8")
        with pytest.raises(ValueError, match="missing 'text'"):
            load_input(str(bad))

    def test_valid_input(self, tmp_path):
        from narrator import load_input

        good = tmp_path / "good.json"
        data = {"title": "T", "segments": [{"role": "narrator", "text": "hi"}]}
        good.write_text(json.dumps(data), encoding="utf-8")
        result = load_input(str(good))
        assert result["title"] == "T"
        assert len(result["segments"]) == 1


# ---------------------------------------------------------------------------
# 10. Voices loading
# ---------------------------------------------------------------------------

class TestVoicesLoading:
    def test_resolve_relative_seed_audio(self, tmp_workspace):
        from narrator import load_voices

        voices_path = str(tmp_workspace / "voices" / "voices.json")
        voices = load_voices(voices_path)

        # narrator has seed_audio resolved to absolute
        narrator_seed = voices["narrator"]["seed_audio"]
        assert Path(narrator_seed).is_absolute()
        assert Path(narrator_seed).exists()

        # 角色A has no seed_audio
        assert voices["角色A"]["seed_audio"] is None
