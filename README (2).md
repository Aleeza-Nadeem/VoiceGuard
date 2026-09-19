# VoiceGuard: An Investigation into Audio Deepfake Detection Generalization

**TL;DR:** I built an AI voice deepfake detector, achieved 91% accuracy on standard benchmarks — then discovered it performed at *chance level* on real-world audio. Rather than treating that as a dead end, I ran four independent, controlled experiments to diagnose exactly why, and isolated the actual cause: a class-imbalanced noise bias in training data that taught the model the wrong signal entirely. This repo documents that full investigation.

---

## Table of Contents
- [Motivation](#motivation)
- [Architecture](#architecture)
- [Dataset Sources](#dataset-sources)
- [Version History & Results](#version-history--results)
- [The Generalization Gap](#the-generalization-gap)
- [Root Cause Analysis](#root-cause-analysis)
- [Key Takeaways](#key-takeaways)
- [What's Next](#whats-next)
- [Tech Stack](#tech-stack)

---

## Motivation

I'd previously built an AI voice dubbing/cloning pipeline (Demucs for source separation, WhisperX for transcription/alignment, speaker diarization, and XTTS-v2 for voice synthesis). Having worked hands-on with how convincing synthetic voices are generated, I wanted to approach the *detection* side of the same problem — could I use what I understood about voice generation to help identify when a voice had been synthesized or cloned?

That question turned into VoiceGuard.

---

## Architecture

```
Audio clip (.wav)
      │
      ▼
Wav2Vec2 / WavLM (pretrained, frozen)
      │  → 768-dim embedding per clip (mean-pooled over time)
      ▼
Random Forest Classifier
      │
      ▼
Prediction: Real (0) or Fake (1)
```

**Why this architecture:** pretrained speech models like Wav2Vec2 already encode rich acoustic structure from large-scale self-supervised training, making them strong feature extractors without needing to train a representation model from scratch. A Random Forest on top is fast, interpretable, and well-suited to fixed-length embedding vectors — practical given development on Google Colab's free tier.

---

## Dataset Sources

| Source | Content | Purpose |
|---|---|---|
| LibriSpeech | Clean audiobook narration | Real-voice training data |
| AUDETER (10 TTS engines: Bark, ChatTTS, CosyVoice, F5-TTS, Fish Speech, SparkTTS, VITS, XTTS, YourTTS, Zonos) | 1,000 clips per tool | Fake-voice training data, high generation-method diversity |
| FLEURS, VoxPopuli | Multilingual / parliamentary real speech | Real-voice diversity |
| ASVspoof2019 LA (systems A04–A06) | Voice-conversion-based fakes | Non-TTS fake-generation diversity |
| Myrtle/CAIMAN-ASR-BackgroundNoise | Ambient noise clips | Noise augmentation |
| **In-The-Wild** (Müller et al., 2022) | Real & synthetic speech of 58 public figures from online video | **Held-out generalization test** — never used in training |

The In-The-Wild dataset (150 real + 150 fake clips, sampled once and never touched during any training run) served as the project's core evaluation tool — a genuinely out-of-distribution test, as opposed to a random split of the training data.

---

## Version History & Results

| Version | Change | Standard Test Accuracy | In-The-Wild Holdout Accuracy |
|---|---|---|---|
| v1 | Baseline: 3k real clips + 10 TTS tools | ~91% | 49% (chance) |
| v2 | Increased real-voice volume to 8k clips | ~91% | 52% (chance) |
| v3 | Added real-voice source diversity (FLEURS, VoxPopuli, ITW-train slice) | 85% | 58% |
| v4 | Added noise-augmented real clips + voice-conversion fakes | 84% | 56% |
| WavLM pilot | Swapped embedding backbone to a noise-robust pretrained model | 100%* | 50% (chance) |

*The pilot's perfect score was a red flag rather than a success — see [Root Cause Analysis](#root-cause-analysis).

**The pattern:** standard test accuracy stayed consistently high (84–91%) across every version. It was never the informative number. The In-The-Wild holdout accuracy — the true test of generalization — never rose meaningfully above chance, across four structurally different interventions.

---

## The Generalization Gap

Standard machine learning practice often evaluates models on a held-out split of the *same* dataset they were trained on. This measures memorization within a distribution, not real-world robustness. VoiceGuard's standard test numbers (84–91%) looked strong by that measure — but the In-The-Wild holdout, drawn from an entirely separate, real-world-sourced dataset, told a very different story: the model was performing at or near random-guessing levels (49–58%) throughout most of the project.

This is a well-documented, actively researched problem in the audio anti-spoofing field — see Müller et al.'s *"Does Audio Deepfake Detection Generalize?"* (2022), the paper that introduced the In-The-Wild dataset specifically to expose this gap in published detection systems.

---

## Root Cause Analysis

Four independent hypotheses were tested, in order:

1. **Insufficient real-voice data volume** → ruled out. More real-voice clips (v1→v2) produced negligible improvement (49%→52%).
2. **Insufficient fake-generation-method diversity** → ruled out. Adding voice-conversion-based fakes (v4) alongside existing TTS-based fakes produced no improvement (56%, comparable to v3's 58%).
3. **Embedding backbone not robust to noise** → ruled out. Swapping Wav2Vec2 for WavLM (pretrained with noise-robustness in mind) did not help — it dropped back to pure chance (50%), with the model's confusion matrix showing it defaulting almost entirely to predicting "real."
4. **Class-asymmetric noise bias** → **identified as the actual root cause.** In v4, background noise augmentation was applied only to real-class clips, in an attempt to make the model robust to noisy real-world audio. This likely taught the model a spurious shortcut: *"noisy audio = real, clean audio = fake."* This is the opposite of reality — a genuine voice-cloning attack (e.g., a scam phone call) is transmitted through noisy, compressed, real-world channels too. A detector that equates cleanliness with fakeness would be systematically fooled by exactly the kind of attack it's meant to catch.

This is a textbook instance of **shortcut learning** (also called spurious correlation): a model exploiting an easy, unintended pattern that correlates with the training labels rather than learning the actual underlying signal.

---

## Key Takeaways

- **High accuracy on a standard test split can be actively misleading** if the test data isn't distributionally distinct from training data. This project's headline number (91%) was, in hindsight, close to meaningless on its own.
- **A held-out, genuinely out-of-distribution test set is essential**, not optional, for any detection/security-adjacent ML system, and should be re-run after *every* change, not periodically.
- **Data augmentation applied asymmetrically across classes can introduce new, harder-to-detect biases** rather than fixing the intended problem — the noise-augmentation experiment (v4) is a clear example.
- **A perfect or near-perfect score should trigger scrutiny, not confidence** — it's frequently a sign of an exploitable shortcut rather than genuine signal, as seen in the WavLM pilot's misleading 100%.

---

## What's Next

The next, untested experiment: apply matched noise augmentation to **both** real and fake clips (rather than real-only, as in v4), removing noise-presence as a usable shortcut and forcing the model to rely on actual voice-authenticity signals. This was identified as the logical next step; the project was deliberately paused here to consolidate and document findings rather than continue open-ended iteration.

---

## Tech Stack

- **Feature extraction:** Wav2Vec2 (`facebook/wav2vec2-base-960h`), WavLM (`microsoft/wavlm-base-plus`)
- **Classifier:** scikit-learn Random Forest
- **Datasets:** LibriSpeech, AUDETER, FLEURS, VoxPopuli, ASVspoof2019 LA, In-The-Wild, Myrtle/CAIMAN-ASR-BackgroundNoise
- **Environment:** Google Colab (free tier), Google Drive for persistence
- **Libraries:** `transformers`, `datasets`, `torch`, `torchaudio`, `scikit-learn`, `numpy`, `pandas`
