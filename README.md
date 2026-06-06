# AI Narrator

基於 [VoxCPM2](https://github.com/openbmb/VoxCPM) 的中文小說多角色有聲書生成器。

將標註好角色的小說文本自動合成為多角色有聲書，每個角色擁有獨立且一致的聲音。

## Features

- 多角色語音合成，每個角色音色獨立
- 基於 Voice Design 生成種子音頻，再用 Controllable Cloning 保持一致性
- 情緒控制（開心、憤怒、悲傷等）
- 智能靜音間隔（角色切換、段落結束等場景）
- 斷點續傳，中斷後可繼續生成

## Requirements

- Python 3.12.9
- NVIDIA GPU with ≥8GB VRAM
- ~5GB disk space for model weights

## Installation

```bash
# Clone with submodule
git clone --recursive https://github.com/vino829/AI-Narrator.git
cd AI-Narrator

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install VoxCPM2
pip install -e ./VoxCPM

# Install dependencies
pip install -r requirements.txt

# Download model weights
python -c "
from huggingface_hub import snapshot_download
snapshot_download('openbmb/VoxCPM2', local_dir='./models/VoxCPM2')
"
```

## Usage

> 開發中，以下為目標用法。

```bash
# Generate audiobook from annotated JSON
python Narrator/cli.py generate --input novel.json --voices voices.json

# Generate seed audio candidates for characters
python Narrator/cli.py seed --voices voices.json

# Quick single-sentence test
python Narrator/cli.py test --text "測試文本" --voice-desc "(年輕女性)"
```

## Input Format

小說輸入為 JSON 格式：

```json
{
  "title": "書名",
  "chapter": "第一章",
  "segments": [
    { "role": "narrator", "text": "他看著窗外的雨。" },
    { "role": "李明", "emotion": "melancholic", "text": "也許我不該來。" }
  ]
}
```

## Project Structure

```
AI Narrator/
  VoxCPM/           # Upstream TTS engine (submodule)
  Narrator/         # Project source code
  models/           # Model weights (not in git)
```

## License

MIT License. See [LICENSE](LICENSE).

VoxCPM2 is licensed under Apache-2.0.
