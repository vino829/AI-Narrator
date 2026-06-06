# AI Narrator CLI 使用指南

## 快速開始

所有指令從 `Narrator/` 目錄執行：

```bash
cd "AI Narrator/Narrator"
```

## 子命令

### `generate` — 生成有聲書

從標註好的 JSON 輸入檔生成多角色有聲書。

```bash
python cli.py generate --input input/example.json
```

**參數：**

| 參數 | 縮寫 | 說明 | 預設值 |
|------|------|------|--------|
| `--input` | `-i` | 輸入 JSON 檔路徑 | （必填） |
| `--voices` | `-v` | voices.json 路徑 | `voices/voices.json` |
| `--output-dir` | `-o` | 輸出目錄 | `output/` |
| `--format` | `-f` | 輸出格式：`wav` / `segments` / `both` | `both`（config.yaml） |
| `--force` | | 強制重新生成所有段落 | `false` |
| `--config` | | config.yaml 路徑 | `config.yaml` |

**輸出格式說明：**

- `both`：分段檔 + 完整檔（預設）
- `wav`：僅完整檔，不保留分段
- `segments`：僅分段檔，不拼接完整檔

**斷點續傳：**

程式會自動偵測 `output/segments/` 中已存在的分段檔案，跳過已生成的段落。使用 `--force` 可強制全部重新生成。

**範例：**

```bash
# 基本用法
python cli.py generate -i input/example.json

# 指定輸出目錄和格式
python cli.py generate -i input/novel_ch1.json -o output/ch1 -f wav

# 強制重新生成
python cli.py generate -i input/example.json --force

# 使用自訂 config
python cli.py generate -i input/example.json --config my_config.yaml
```

---

### `seed` — 生成種子音頻

為 `voices.json` 中尚未指定 `seed_audio` 的角色生成候選種子音頻。

```bash
# 生成候選
python cli.py seed

# 指定候選數量
python cli.py seed --candidates 5

# 進入選擇模式（互動式 CLI）
python cli.py seed --select
```

**參數：**

| 參數 | 縮寫 | 說明 | 預設值 |
|------|------|------|--------|
| `--voices` | `-v` | voices.json 路徑 | `voices/voices.json` |
| `--output-dir` | `-o` | 種子音頻輸出目錄 | `voices/` |
| `--candidates` | `-n` | 每個角色的候選數量 | `3` |
| `--select` | | 進入互動選擇模式 | `false` |

**工作流程：**

1. 執行 `python cli.py seed` 生成候選音頻
2. 手動試聽候選檔案（如 `voices/narrator_candidate_0.wav`）
3. 執行 `python cli.py seed --select` 選擇最佳候選
4. 選定後 `voices.json` 會自動更新 `seed_audio` 路徑

---

### `test` — 單句快速測試

快速生成單句語音，用於測試模型或調整參數。

```bash
python cli.py test --text "你好，這是一段測試語音。"
```

**參數：**

| 參數 | 縮寫 | 說明 | 預設值 |
|------|------|------|--------|
| `--text` | `-t` | 要合成的文本 | （必填） |
| `--voice-desc` | `-d` | 聲音描述（Voice Design 模式） | 無 |
| `--reference` | `-r` | 參考音頻路徑（Cloning 模式） | 無 |
| `--output` | `-o` | 輸出檔路徑 | `test.wav` |

**範例：**

```bash
# Voice Design 模式
python cli.py test -t "今天天氣真好。" -d "年輕女性，溫柔甜美"

# Controllable Cloning 模式
python cli.py test -t "今天天氣真好。" -r voices/narrator.wav

# 指定輸出路徑
python cli.py test -t "測試" -o output/my_test.wav
```

---

## 設定檔 (`config.yaml`)

全局設定檔位於 `Narrator/config.yaml`，CLI 參數會覆蓋設定檔中的值。

```yaml
model:
  path: ./models/VoxCPM2      # 模型路徑（相對於專案根目錄）
  load_denoiser: false         # 是否載入降噪器

generation:
  cfg_value: 2.0               # CFG 值，越高越貼合 reference
  inference_timesteps: 10      # 擴散步數，越多品質越好但越慢

silence:                       # 段落間靜音時長（毫秒）
  same_role: 300               # 同角色連續句
  role_switch: 500             # 角色切換（對白↔對白）
  narration_switch: 600        # 旁白↔對白
  paragraph_end: 800           # 段落結束

output:
  format: both                 # 預設輸出格式
  sample_rate: 48000           # 取樣率
```

---

## 輸入 JSON 格式

```json
{
  "title": "書名或章節標題",
  "chapter": "第一章",
  "segments": [
    {
      "role": "narrator",
      "text": "旁白文字。"
    },
    {
      "role": "角色名",
      "emotion": "情緒描述（可選）",
      "text": "角色台詞。"
    }
  ]
}
```

**欄位說明：**

- `title`：標題，用於完整輸出檔命名
- `chapter`：章節名（可選）
- `segments[].role`：角色名，必須與 `voices.json` 中的 key 對應
- `segments[].text`：台詞或旁白文字
- `segments[].emotion`：情緒描述（可選），如 `melancholic`、`angry`、`surprised`

---

## 輸出結構

```
output/
  segments/
    001_narrator.wav
    002_李明.wav
    003_narrator.wav
    ...
  書名_第一章.wav          # 完整拼接檔
```
