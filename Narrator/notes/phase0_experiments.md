# Phase 0 Experiment Results

Date: 2026-06-06
GPU: RTX 5060 Laptop (8GB VRAM)
Model: VoxCPM2 v2.0.3

---

## 0.3 Smoke Tests

All three generation modes work correctly:

| Mode | RTF | Notes |
|------|-----|-------|
| Basic TTS | 0.86 | No reference, no description |
| Voice Design | 0.84 | Description in parentheses |
| Controllable Cloning | 5.44 | With reference_wav_path |

Cloning RTF significantly higher than other modes.

## 0.4 Voice Design Consistency

**Result: Completely inconsistent.** Same description generates entirely different
voices across 5 runs. One sample even had degraded audio quality. Confirms that
Voice Design cannot be used directly for per-sentence generation — reference
cloning (Phase 2) is the correct strategy.

## Additional Experiments

### Controllable Cloning Consistency (Test A)

Using the same reference audio with 5 different sentences: voice identity is
generally preserved, but occasional pitch drift and artifacts occur (1 out of 5
had audible issues). Acceptable for production with retry capability.

### Long Text (Test B)

90-character paragraph generated correctly. No truncation, repetition, or
quality degradation. 19.0s audio in 17.0s (RTF=0.89).

### Emotion Control (Tests C, 6-9)

**Key findings:**

1. **Voice Design mode: emotion works but breaks voice identity.**
   Adding emotion to the description changes timbre — each emotion
   produces a different-sounding speaker. Cannot rely on description alone
   for consistent character voice + emotion.

2. **Controllable Cloning mode: emotion has minimal effect.**
   Emotion prefixes like `(angry)` or `(sad)` in the text produce almost
   no audible difference when using reference_wav_path. Only slight speed
   changes observed. This holds true for:
   - Chinese vs English descriptions
   - Short vs verbose descriptions
   - cfg_value 1.5 vs 2.0 vs 2.5

3. **cfg_value findings:**
   - cfg < 1.5: significant quality degradation (slurred speech, pitch issues)
   - cfg 1.5-2.5: no meaningful emotion difference in cloning mode
   - Recommended range: 1.5-2.5, default 2.0

4. **Emotion gradient (Voice Design only):**
   - sad/happy: mild/medium/strong gradient works reasonably well
   - angry: highly unstable — same description produces wildly different
     intensity levels across runs
   - All gradients suffer from voice identity inconsistency

5. **Emotion-specific seed + Cloning (Test 5, 8):**
   - Using a seed generated with an emotional description (e.g., angry voice)
     then cloning from it: the cloned output does NOT reliably carry the
     emotion from the seed. Cloning primarily transfers timbre, not affect.

6. **Angry emotion is particularly unreliable** across all modes and strategies.

### Conclusions for Architecture

- **Voice identity** must come from reference cloning (seed audio), not descriptions
- **Emotion control via text prefix is ineffective** in Cloning mode
- **Best current strategy:** one neutral seed per character, accept limited emotion
- **Future goal:** fine-tune emotion-specific seeds or LoRA per character to get
  reliable emotional expression while maintaining voice identity
- **Regeneration capability is essential** — artifacts and quality issues occur
  randomly; users need to be able to retry specific segments
