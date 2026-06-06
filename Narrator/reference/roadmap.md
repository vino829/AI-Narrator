# AI Narrator Roadmap

基於 VoxCPM2 的中文小說多角色有聲書生成器。

---

## Phase 0：準備工作

### 0.1 Python 環境

VoxCPM 要求 Python >=3.10, <3.13（目前 3.14.0 不相容）。

- [x]  安裝 Python 3.12.10
- [x]  建立 venv
  ```bash
  cd "C:/Users/anf28/Desktop/AI Narrator"
  python3.12 -m venv .venv
  .venv\Scripts\activate    # Windows
  ```
- [x]  安裝 VoxCPM 及相依套件
  ```bash
  pip install -e ./VoxCPM
  ```

  若 editable install 失敗，改用 `pip install ./VoxCPM`。
- [x]  安裝額外依賴
  ```bash
  pip install soundfile numpy
  ```

**驗證方法：**

```python
import voxcpm; print(voxcpm.__version__)
```

無報錯即通過。

### 0.2 模型下載

- [x]  下載 VoxCPM2 模型權重
  ```python
  from huggingface_hub import snapshot_download
  snapshot_download("openbmb/VoxCPM2", local_dir="./models/VoxCPM2")
  ```

  模型約 4-5GB，存放於 `AI Narrator/models/VoxCPM2/`。

**驗證方法：**
確認 `models/VoxCPM2/config.json` 存在且 `"architecture": "voxcpm2"`。

### 0.3 推理冒煙測試

- [x]  執行最小 TTS 測試
  ```python
  from voxcpm import VoxCPM
  import soundfile as sf

  model = VoxCPM.from_pretrained("./models/VoxCPM2", load_denoiser=False)
  wav = model.generate(
      text="這是一段測試語音，用來驗證模型是否正常運作。",
      cfg_value=2.0,
      inference_timesteps=10,
  )
  sf.write("test_output.wav", wav, model.tts_model.sample_rate)
  ```
- [x]  測試 Voice Design 模式
  ```python
  wav = model.generate(
      text="(年輕女性，溫柔甜美的聲音)你好，歡迎收聽有聲小說。",
      cfg_value=2.0,
      inference_timesteps=10,
  )
  sf.write("test_voice_design.wav", wav, model.tts_model.sample_rate)
  ```
- [x]  測試 Controllable Cloning 模式
  ```python
  # 用上一步生成的音頻作為 reference
  wav = model.generate(
      text="這是用參考音頻克隆的語音。",
      reference_wav_path="test_voice_design.wav",
      cfg_value=2.0,
      inference_timesteps=10,
  )
  sf.write("test_clone.wav", wav, model.tts_model.sample_rate)
  ```

**驗證方法：**

- 三個 wav 檔皆可正常播放
- `test_clone.wav` 的音色與 `test_voice_design.wav` 相近
- 記錄單句生成耗時（RTX 5060 Laptop 預期 RTF < 0.5）

### 0.4 Voice Design 一致性實驗

這步很關鍵——測試同一描述多次生成的聲音差異程度。

- [ ]  用同一描述生成 5 次，比較一致性
  ```python
  desc = "(中年男性，沉穩的朗讀聲音)"
  for i in range(5):
      wav = model.generate(
          text=f"{desc}夜色籠罩了整座城市，街道上行人稀少。",
          cfg_value=2.0,
          inference_timesteps=10,
      )
      sf.write(f"consistency_test_{i}.wav", wav, model.tts_model.sample_rate)
  ```
- [ ]  人耳比較：5 個檔案的音色是否一致？
- [ ]  結論記錄：Voice Design 是否可靠到直接用於每句生成？
  - 若否（預期結果）：確認 Phase 2 的 reference cloning 策略是正確方向

**驗證方法：**
主觀聽感。記錄結論到 `Narrator/notes/` 供後續參考。

---

## Phase 1：基礎管線

核心目標：一個 Python 腳本，讀取標註好的 JSON，逐段生成音頻，輸出完整有聲書。

### 1.1 定義輸入格式

- [ ]  設計 JSON schema

**小說輸入檔** (`input.json`)：

```json
{
  "title": "測試章節",
  "chapter": "第一章",
  "segments": [
    {
      "role": "narrator",
      "text": "他看著窗外的雨，嘆了口氣。"
    },
    {
      "role": "李明",
      "emotion": "melancholic",
      "text": "也許我不該來這裡。"
    },
    {
      "role": "王芳",
      "emotion": "angry",
      "text": "你現在才知道嗎？"
    },
    {
      "role": "narrator",
      "text": "房間陷入了沉默。"
    }
  ]
}
```

欄位說明：

- `role`：角色識別名（必填）
- `text`：台詞/旁白文字（必填）
- `emotion`：情緒描述，可選，會與角色 voice profile 的描述合併

- [ ]  建立 1-2 個測試用 JSON（500-1000 字，3-5 個角色）

**驗證方法：** JSON 可被 `json.load()` 正確讀取，schema 欄位完整。

### 1.2 角色聲音設定

- [ ]  設計 voice profile 格式

**角色設定檔** (`voices.json`)：

```json
{
  "narrator": {
    "description": "成熟男性，沉穩的播音腔，語速適中",
    "seed_audio": null
  },
  "李明": {
    "description": "二十多歲的年輕男性，聲音溫和略帶憂鬱",
    "seed_audio": null
  },
  "王芳": {
    "description": "二十多歲的年輕女性，聲音清脆明亮",
    "seed_audio": null
  }
}
```

`seed_audio` 初始為 null，Phase 2 的聲音管理器會填入路徑。

### 1.3 核心合成腳本

- [ ]  實作 `narrator.py`，核心邏輯：

```
載入模型
載入 input.json + voices.json
for each segment:
    組合 voice description + emotion + text
    if seed_audio 存在:
        用 Controllable Cloning 模式
    else:
        用 Voice Design 模式
    生成 wav，存為分段檔案
拼接所有分段（插入靜音間隔）
輸出完整音頻
```

- [ ]  靜音間隔策略：

  - 同角色連續句：300ms
  - 角色切換（對白→對白）：500ms
  - 旁白→對白 或 對白→旁白：600ms
  - 段落結束（句尾為句號且下一段是旁白）：800ms
- [ ]  分段檔案命名：`output/segments/001_narrator.wav`, `002_liMing.wav`, ...
- [ ]  完整輸出：`output/full_chapter.wav`

**驗證方法：**

- 分段檔案與完整音頻皆可播放
- 角色聲音可區分
- 靜音間隔聽感自然
- 完整流程無報錯

### 1.4 基礎錯誤處理

- [ ]  空文本跳過（log warning）
- [ ]  單段生成失敗時：log error，插入靜音佔位，繼續處理剩餘段落
- [ ]  生成完成後輸出摘要：總段數、成功/失敗數、總時長、總耗時

**驗證方法：** 故意在 JSON 中放入一段空文本，確認不會中斷整個流程。

---

## Phase 2：聲音管理

核心目標：為每個角色生成並儲存種子音頻，後續用 reference cloning 保證一致性。

### 2.1 種子音頻生成工具

- [ ]  實作 `seed_generator.py`：

  ```
  讀取 voices.json
  for 每個 seed_audio 為 null 的角色:
      用 Voice Design 生成 3 個候選音頻
      存為 voices/{role}_candidate_0.wav ~ _2.wav
      輸出提示：「請試聽並選擇最佳候選」
  ```
- [ ]  候選音頻的文本內容需能展現角色特色：

  - 旁白：用一段典型的敘事文
  - 角色：用一段包含多種語氣的對白（平述 + 疑問 + 感嘆）
  - 長度建議 15-30 秒（太短音色資訊不足，太長浪費生成時間）

**驗證方法：** 候選音頻可播放，同角色的 3 個候選音色大致相近（Voice Design 的變異度可接受）。

### 2.2 種子音頻確認與註冊

- [ ]  實作選擇流程（CLI 互動或直接手動複製檔案）：
  ```bash
  # 手動：試聽後將最佳候選複製為正式種子
  cp voices/narrator_candidate_1.wav voices/narrator.wav
  ```
- [ ]  更新 `voices.json` 中對應角色的 `seed_audio` 路徑
  ```json
  "narrator": {
    "description": "成熟男性，沉穩的播音腔",
    "seed_audio": "voices/narrator.wav"
  }
  ```
- [ ]  核心合成腳本 (`narrator.py`) 檢測到 `seed_audio` 後自動切換為 Controllable Cloning 模式

**驗證方法：**

- 用同一角色的種子音頻生成 10 句不同文本
- 10 句的音色應高度一致（明顯優於 Phase 0.4 的純 Voice Design 結果）

### 2.3 情緒疊加驗證

- [ ]  測試 reference cloning + 情緒控制的搭配效果：
  ```python
  # 保持音色，改變情緒
  for emotion in ["平靜", "開心", "憤怒", "悲傷", "驚訝"]:
      wav = model.generate(
          text=f"({emotion})這句話用來測試情緒控制的效果。",
          reference_wav_path="voices/narrator.wav",
          cfg_value=2.0,
      )
  ```
- [ ]  記錄 `cfg_value` 對情緒表現力的影響：
  - cfg=1.5：更自由，情緒更明顯，但音色可能偏離
  - cfg=2.0：平衡點（預設值）
  - cfg=2.5：更貼合 reference，但情緒可能被壓制
- [ ]  確定每個角色的最佳 `cfg_value`，記錄到 voices.json

**驗證方法：** 不同情緒的音頻聽感上有情緒差異，但音色保持一致。找到 cfg_value 的甜蜜點。

### 2.4 聲音管理目錄結構

- [ ]  最終結構：
  ```
  Narrator/
    voices/
      narrator.wav
      narrator_candidate_0.wav  (可選保留)
      narrator_candidate_1.wav
      narrator_candidate_2.wav
      李明.wav
      王芳.wav
      ...
    voices.json
  ```

---

## Phase 3：CLI 工具化

核心目標：將 Phase 1-2 的腳本整合為一個易用的命令列工具。

### 3.1 CLI 入口

- [ ]  實作 `cli.py`，支持以下子命令：

```bash
# 生成有聲書（主要功能）
python cli.py generate \
    --input novel.json \
    --voices voices.json \
    --output-dir output/ \
    --format both           # wav / segments / both

# 生成種子音頻候選
python cli.py seed \
    --voices voices.json \
    --output-dir voices/ \
    --candidates 3

# 單句快速測試
python cli.py test \
    --text "測試文本" \
    --voice-desc "(年輕女性)" \
    --output test.wav
```

### 3.2 生成流程完善

- [ ]  進度顯示：tqdm 進度條，顯示 `段落 X/Y, 角色: XXX`
- [ ]  斷點續傳：
  - 生成前檢查 `output/segments/` 中已有的分段檔案
  - 跳過已存在的分段（依檔名序號判斷）
  - 加 `--force` 旗標可強制全部重新生成
- [ ]  生成結束後輸出報告：
  ```
  生成完成
  總段數: 142
  成功: 140, 失敗: 2
  總音頻時長: 48:32
  總生成耗時: 15:24
  RTF: 0.32
  輸出: output/full_chapter.wav
  ```

### 3.3 輸出格式

- [ ]  分段輸出：`output/segments/001_narrator.wav`, ...
- [ ]  完整輸出：`output/{title}_{chapter}.wav`
- [ ]  可選 MP3 輸出（依賴 ffmpeg 或 pydub）

### 3.4 設定檔

- [ ]  支持 YAML 全局設定 (`config.yaml`)：
  ```yaml
  model:
    path: ./models/VoxCPM2
    device: auto
    load_denoiser: false

  generation:
    cfg_value: 2.0
    inference_timesteps: 10

  silence:
    same_role: 300        # ms
    role_switch: 500
    narration_switch: 600
    paragraph_end: 800

  output:
    format: both          # wav / segments / both
    sample_rate: 48000
  ```
- [ ]  CLI 參數覆蓋設定檔（CLI 優先）

**驗證方法：**

- `generate` 子命令能從 JSON 生成完整有聲書
- 中斷後重新執行，已完成的段落不會重複生成
- `--force` 可強制重新生成
- 進度條正確顯示

---

## Phase 4：未來目標（暫不實作）

以下為未來擴展方向，記錄想法供日後參考。

### 4.1 LLM 自動角色解析（原 Phase 2）

- 接入 Claude API 或其他 LLM，自動從原始小說文本解析角色對白/旁白
- 輸入：純文本小說
- 輸出：結構化 JSON（同 Phase 1 的輸入格式）
- 挑戰：省略引號的對白、隱含說話者、多人對話場景

### 4.2 Web UI

- 基於 Gradio 或獨立前端
- 小說文本編輯器 + 角色管理面板
- 章節進度條、分章生成/重新生成
- 音頻預覽與下載

### 4.3 LoRA 微調特定角色

- 適用場景：固定角色反覆出現（如系列小說的主角）
- 需要 5-10 分鐘該角色的高品質音頻（可從 Phase 2 的 reference cloning 成果中擷取）
- 訓練設定：使用現有 `conf/voxcpm_v2/voxcpm_finetune_lora.yaml`
- LoRA 權重熱切換：`model.load_lora()` / `model.unload_lora()`
- VRAM 需求：LoRA 微調峰值約 16-24GB（RTX 5060 Laptop 8GB 需要大幅降低 batch_size 或改用雲端 GPU）

### 4.4 多章節批次處理

- 輸入一整本小說（多章節 JSON 或目錄結構）
- 自動分章生成，輸出按章節組織的音頻檔
- 全書拼接（含章節間停頓 / 章節提示音）

### 4.5 音頻後處理

- 背景音樂混合
- 音量正規化（loudness normalization）
- 淡入/淡出效果

---

## 目錄結構規劃

```
AI Narrator/
  VoxCPM/                   # 上游開源專案（不修改）
  Narrator/
    reference/
      roadmap.md              # 本文件
    cli.py                  # CLI 入口
    narrator.py             # 核心合成邏輯
    seed_generator.py       # 種子音頻生成工具
    config.yaml             # 全局設定
    voices/                 # 角色種子音頻
      voices.json           # 角色聲音設定檔
    input/                  # 小說輸入檔
      example.json
    output/                 # 生成輸出
      segments/             # 分段音頻
    notes/                  # 實驗筆記
  models/                   # 模型權重
    VoxCPM2/
  .venv/                    # Python 虛擬環境
  README.md
  REQUIREMENTS.md
```
