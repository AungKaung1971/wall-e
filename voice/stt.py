# voice/stt.py

import json
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

SAMPLE_RATE = 16000
MODEL_PATH = "models/vosk-model-en-us-0.22-lgraph"

# Load Vosk model once
print("[STT] Loading Vosk model...")
vosk_model = Model(MODEL_PATH)
print("[STT] Model loaded successfully.")


def record_command(duration_sec=4):
    """
    Record microphone audio for a short duration.
    Returns a numpy int16 audio buffer.
    """
    print(f"[STT] Recording command ({duration_sec} seconds)...")
    audio = sd.rec(
        int(duration_sec * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    print("[STT] Recording complete.")
    return audio


def transcribe(audio_np):
    """
    Transcribes the recorded audio buffer into text.
    """
    recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    recognizer.AcceptWaveform(audio_np.reshape(-1).tobytes())
    result = json.loads(recognizer.FinalResult())
    text = result.get("text", "").strip()
    print(f"[STT] Transcription: '{text}'")
    return text
