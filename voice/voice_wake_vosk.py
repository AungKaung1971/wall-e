import queue
import json
import time
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000

# ⭐ Your model is now inside models/
MODEL_PATH = "models/vosk-model-en-us-0.22-lgraph"

print("[VOICE] Loading Vosk model...")
vosk_model = Model(MODEL_PATH)
print("[VOICE] Model loaded successfully.")

# ============================================================
# WAKE-WORD LISTENER
# ============================================================


def detect_wake_word():
    """
    Continuously listens to the microphone until it hears a phrase
    that sounds like 'hey wally'.
    """
    q = queue.Queue()

    def audio_callback(indata, frames, t, status):
        if status:
            print("[VOICE-WARNING]", status)
        q.put(bytes(indata))

    recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)

    WAKE_PHRASES = [
        "hey wally",
        "hey wali",
        "hey wall e",
        "hey walley",
        "hey wolley",
        "hey ollie"   # optional similar-sounding fallback
    ]

    print("[VOICE] Listening for wake word: 'hey wally'...")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback
    ):
        while True:
            data = q.get()

            # Check final results
            if recognizer.AcceptWaveform(data):
                text = json.loads(recognizer.Result()).get("text", "").lower()

                if any(p in text for p in WAKE_PHRASES):
                    print(f"[VOICE] Wake word detected (final): {text}")
                    return

            # Check partial results (faster detection)
            else:
                partial = json.loads(recognizer.PartialResult()).get(
                    "partial", "").lower()

                if any(p in partial for p in WAKE_PHRASES):
                    print(f"[VOICE] Wake word detected (partial): {partial}")
                    return

# ============================================================
# COMMAND RECORDING
# ============================================================


def record_command(duration_sec=4):
    """
    After wake word, record a short voice command.
    """
    print(f"[VOICE] Listening for command ({duration_sec} seconds)...")
    audio = sd.rec(
        int(duration_sec * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    print("[VOICE] Command finished.")
    return audio

# ============================================================
# TRANSCRIPTION
# ============================================================


def transcribe_with_vosk(audio_np):
    """
    Convert the recorded audio into text using Vosk.
    """
    recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    recognizer.AcceptWaveform(audio_np.reshape(-1).tobytes())
    result = json.loads(recognizer.FinalResult())
    return result.get("text", "").strip()

# ============================================================
# COMMAND HANDLER
# ============================================================


def handle_command(text):
    """
    This is where you connect commands to your robot logic.
    """
    if not text:
        print("[VOICE] No speech detected.")
        return

    print(f"[VOICE] You said: '{text}'")

    # TODO: integrate with your robot:
    # response = chat_with_llm(text)
    # speak(response)
    # perform_action(text)

# ============================================================
# MAIN LOOP
# ============================================================


def main_loop():
    print("[VOICE] Starting continuous voice assistant...")
    while True:
        detect_wake_word()

        print("[VOICE] Wake word heard! Say your command.")
        time.sleep(0.4)

        audio = record_command(duration_sec=4)
        text = transcribe_with_vosk(audio)
        handle_command(text)

        time.sleep(0.3)


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n[VOICE] Exiting.")
