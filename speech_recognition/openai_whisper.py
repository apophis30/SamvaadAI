import openai
from openai import OpenAI
import os
openai.api_key = os.getenv("OPENAI_API_KEY")
import wave
client = OpenAI(api_key="#####")

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

def transcribe(file_path):
    audio_file = open(file_path, 'rb')
    print("ok")
    transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
    print("yaya ok")
    print(transcription.text)
    return transcription.text

