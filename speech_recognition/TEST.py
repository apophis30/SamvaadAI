import streamlit as st
from audio_recorder_streamlit import audio_recorder
import soundfile as sf
import wave

class Config:
    channels = 2
    sample_width = 2
    sample_rate = 44100

def save_wav_file(file_path, wav_bytes):
    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setnchannels(Config.channels)
        wav_file.setsampwidth(Config.sample_width)
        wav_file.setframerate(Config.sample_rate)
        wav_file.writeframes(wav_bytes)
# audio_file = st.file_uploader("Please upload an audio file", type=["wav", "mp3"])
import numpy as np
voice: bool = st.checkbox("I would like to speak with AI Interviewer")
if voice:
    answer = audio_recorder(pause_threshold = 2.5, sample_rate = 44100)

    save_wav_file(r"C:\Users\Arpan Kumar\Downloads\audio.wav", answer)
