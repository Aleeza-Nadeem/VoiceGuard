import streamlit as st
import numpy as np
import joblib
import librosa
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

st.set_page_config(page_title="VoiceGuard", page_icon="🎙️", layout="centered")

st.title("🎙️ VoiceGuard — AI Voice Detector")
st.caption(
    "Upload an audio clip to check whether it's likely a real human voice "
    "or AI-generated (voice cloning / TTS)."
)

@st.cache_resource
def load_models():
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    wav2vec_model.eval()
    classifier = joblib.load("voiceguard_model_8k.pkl")
    return feature_extractor, wav2vec_model, classifier

feature_extractor, wav2vec_model, classifier = load_models()
TARGET_SR = 16000
MAX_SECONDS = 15

uploaded_file = st.file_uploader("Upload audio (wav or mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    with st.spinner("Analyzing audio..."):
        audio_array, sr = librosa.load(uploaded_file, sr=TARGET_SR)
        audio_array = audio_array[: TARGET_SR * MAX_SECONDS]

        inputs = feature_extractor(audio_array, sampling_rate=TARGET_SR, return_tensors="pt")
        with torch.no_grad():
            outputs = wav2vec_model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).numpy()

        proba = classifier.predict_proba(embedding)[0]
        pred = classifier.predict(embedding)[0]

    st.audio(uploaded_file)
    st.divider()

    if pred == 1:
        st.error(f"⚠️ Likely AI-Generated — {proba[1]:.0%} confidence")
    else:
        st.success(f"✅ Likely Real Human Voice — {proba[0]:.0%} confidence")

    with st.expander("How confident is this, really?"):
        st.write(
            "This model performs strongly (91% accuracy) on voice-cloning tools "
            "similar to its training data (10 modern TTS systems from the AUDETER "
            "dataset). It generalizes partially to unseen tools within the same "
            "family (~78% recall on a held-out TTS tool), but its accuracy drops "
            "close to chance level on real-world noisy audio using different "
            "spoofing techniques (tested against the In-The-Wild benchmark). "
            "Treat this as one signal among several, not a certainty verdict."
        )

st.divider()
st.caption(
    "VoiceGuard combines a pretrained speech model (Wav2Vec2) with a Random "
    "Forest classifier trained on 10 diverse AI voice-cloning tools (AUDETER "
    "dataset) plus real human speech (LibriSpeech). Built by someone who also "
    "built a full AI voice-cloning pipeline (XTTS-v2) — understanding the "
    "generation side informed the detection approach."
)
