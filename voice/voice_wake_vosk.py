import queue
import json
import time
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# Import STT module
from voice.stt import record_command, transcribe

# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_RATE = 16000
MODEL_PATH = "models/vosk-model-en-us-0.22-lgraph"

print("[WAKE] Loading Vosk model...")
vosk_model = Model(MODEL_PATH)
print("[WAKE] Model loaded successfully.")

# ============================================================
# WAKE WORD LISTENER
# ============================================================


def detect_wake_word():
    """
    Continuously listens to the microphone until it hears a phrase
    that sounds like 'hey wally'.
    """
    q = queue.Queue()

    def audio_callback(indata, frames, t, status):
        if status:
            print("[WAKE-WARNING]", status)
        q.put(bytes(indata))

    recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)

    WAKE_PHRASES = [
        "hey wally",
        "hey wali",
        "hey wall e",
        "hey walley",
        "hey wolley",
        "hey ollie"
    ]

    print("[WAKE] Listening for wake word: 'hey wally'...")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        while True:
            data = q.get()

            # Check final results
            if recognizer.AcceptWaveform(data):
                text = json.loads(recognizer.Result()).get("text", "").lower()
                if any(p in text for p in WAKE_PHRASES):
                    print(f"[WAKE] Wake word detected (final): {text}")
                    return

            # Check partial results (faster)
            else:
                partial = json.loads(recognizer.PartialResult()).get(
                    "partial", "").lower()
                if any(p in partial for p in WAKE_PHRASES):
                    print(f"[WAKE] Wake word detected (partial): {partial}")
                    return

# ============================================================
# MAIN LOOP
# ============================================================


def main_loop():
    print("[WAKE] Voice assistant initialized.")

    while True:
        detect_wake_word()

        print("[WAKE] Wake word heard! Listening for your command...")
        time.sleep(0.4)

        # STT MODULE
        audio = record_command(4)
        text = transcribe(audio)

        if text:
            print(f"[WAKE] You said: '{text}'")
        else:
            print("[WAKE] No speech detected.")

        time.sleep(0.3)


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n[WAKE] Exiting cleanly.")


# run code
# python voice/voice_wake_vosk.py
