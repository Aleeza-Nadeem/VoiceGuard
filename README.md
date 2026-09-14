# 🎙️ VoiceGuard — AI Voice Deepfake Detector

A machine learning tool that detects whether an audio clip is a real human voice or AI-generated (voice cloning / text-to-speech), built by someone who also built a full AI voice-cloning pipeline — using that generation-side understanding to inform the detection approach.

---

## The Problem

Voice cloning scams are a real, growing threat — calls that convincingly sound like a family member or executive, used to request money or sensitive information. Existing detection tools are largely closed, black-box, and built by teams who understand detection but not generation. Most public detector projects are also trained on outdated benchmarks (e.g. ASVspoof 2019) that don't reflect how modern TTS systems actually generate voices.

## What Makes This Different

This project was built alongside a separate, complete AI dubbing pipeline (Demucs → WhisperX → speaker diarization → LLM translation → XTTS-v2 voice cloning). That means the detector was designed from direct, hands-on understanding of how a modern voice clone is actually generated — not from guessing at "AI voices sound robotic" folklore. The core hypothesis: a convincing clone reproduces the big, easily-measured characteristics of a voice (pitch, tone, accent) but tends to smooth away the tiny, involuntary imperfections (breathing irregularity, pitch jitter, natural vocal inconsistency) that a real human vocal tract produces.

## How It Works

1. **Pretrained embeddings** — audio is passed through Wav2Vec2, a model pretrained on millions of hours of real human speech. It has no built-in concept of "fake" — it simply describes any audio it's given using deeply learned real-speech patterns. When fed a fake clip, the resulting embedding subtly reflects how that clip deviates from real speech, even though the model was never told to look for deviation.
2. **Classification** — a Random Forest classifier, trained on these embeddings, learns to read that deviation as the real/fake signal.

```
Audio clip → Wav2Vec2 (pretrained, real-speech-only) → 768-dim embedding → Random Forest → Real / Fake
```

## Dataset

- **Fake audio:** [AUDETER](https://github.com/mueller91) (KDD 2026) — 10 diverse, modern TTS/voice-cloning tools (bark, chattts, cosyvoice, f5_tts, fish_speech, sparktts, vits, xtts, yourtts, zonos), 1,000 clips each. Chosen over older benchmarks (ASVspoof, Fake-or-Real) after confirming via recent research that detectors trained on outdated data do not reliably generalize to newer TTS architectures.
- **Real audio:** LibriSpeech (clean audiobook narration), 8,000 clips, randomly sampled with shuffling for diversity.

## Results

### Standard test set (held-out 20% of training distribution)
| Class | Precision | Recall | F1 |
|---|---|---|---|
| Real (0) | 0.94 | 0.85 | 0.90 |
| Fake (1) | 0.88 | 0.95 | 0.91 |

**Accuracy: 91%**

### Generalization test 1 — unseen TTS tool (fish_speech, held out entirely from training)
Recall on unseen tool: **77-80%** (vs. random chance of 50%) — the model partially generalizes to a new tool within the same generation family (TTS).

### Generalization test 2 — real-world benchmark (In-The-Wild dataset, Müller et al. 2022)
Accuracy: **~50%** (chance level) — tested against real celebrity/political speech (noisy, real-world recording conditions) and older-era, non-TTS spoofing techniques.

**Honest interpretation:** performance degrades in two independent, diagnosable ways — (1) real-audio distribution mismatch (clean LibriSpeech narration vs. noisy real-world recordings), and (2) fake-generation-method mismatch (modern TTS vs. older/different spoofing families). Crossing *one* of these boundaries (fish_speech) causes moderate degradation; crossing *both simultaneously* (In-The-Wild) drops performance to chance level. This is a genuine, evidence-based finding about the limits of training-distribution generalization — not a hidden flaw, but a clearly characterized boundary of the current model.

## Known Limitations

- Trained real-audio data (LibriSpeech) is clean studio-style narration, not representative of noisy real-world audio (phone calls, social media, video).
- Trained fake-audio data covers modern TTS systems only — does not include voice conversion or older-generation spoofing techniques.
- As with any supervised classifier, cannot recognize a genuinely novel attack pattern absent from training data.

## Future Improvements

- Add noisy, real-world real-speech data (e.g. Common Voice) to close the real-audio distribution gap.
- Add non-TTS spoofing/voice-conversion techniques to training data to close the fake-generation-method gap.
- Re-test against In-The-Wild after both additions to measure concrete improvement.
- Explore explicit acoustic features (pitch jitter, breathing pattern via librosa/Praat) alongside embeddings for improved interpretability.
- Test on a larger, more diverse held-out set to confirm generalization findings are consistent, not sample-specific.

## Project Structure

```
voiceguard/
├── app.py                      # Streamlit web app
├── requirements.txt            # Python dependencies
├── voiceguard_model_8k.pkl     # Trained Random Forest classifier
└── README.md
```

## Tech Stack

Python · PyTorch · Transformers (Wav2Vec2) · scikit-learn · librosa · Streamlit · Hugging Face Datasets

---

*This tool provides a probability-based signal, not a certainty verdict. It should be used as one input among several when assessing audio authenticity, not as a sole source of truth.*
