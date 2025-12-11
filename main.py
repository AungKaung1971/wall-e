# main.py

from voice.voice_wake_vosk import detect_wake_word
from voice.stt import record_command, transcribe
from voice.conversation import ask_llm
import time


def main():
    print("\n=== WALL-E Voice System (Wake Word + STT + LLM) ===\n")
    print("Say 'hey wally' to begin...\n")

    while True:
        # 1. Wake word
        detect_wake_word()
        print("[MAIN] Wake word detected! Listening for your command...\n")
        time.sleep(0.4)

        # 2. Record command
        audio = record_command(4)

        # 3. Transcribe command
        text = transcribe(audio)

        if not text:
            print("[MAIN] No speech detected.\n")
            continue

        print(f"[MAIN] You said: {text}")

        # 4. LLM Brain
        reply = ask_llm(text)

        # 5. Output response (text-only for now)
        print(f"[WALL-E] {reply}\n")
        print("[MAIN] Ready for next wake word...\n")
        time.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down.\n")
