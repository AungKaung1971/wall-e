import platform
import subprocess
import os

# Piper voice model — set the one you install later on your Pi
PIPER_MODEL = "/usr/share/piper/models/en_US-lessac-medium.onnx"


def speak(text: str, voice=None):
    """
    Auto-select TTS engine:
      • macOS -> `say`
      • Raspberry Pi/Linux -> Piper (if installed)
      • Raspberry Pi fallback -> espeak-ng
    """
    system = platform.system().lower()

    # -----------------------------------------
    # MACOS (CURRENT MACHINE)
    # -----------------------------------------
    if "darwin" in system:
        print("[TTS] macOS using 'say'")
        cmd = ["say"]
        if voice:  # allow Zarvox or others
            cmd += ["-v", voice]
        cmd.append(text)
        subprocess.run(cmd)
        return

    # -----------------------------------------
    # LINUX / RASPBERRY PI (FUTURE MACHINE)
    # -----------------------------------------
    elif "linux" in system:

        # Try Piper first
        if os.path.exists(PIPER_MODEL):
            print("[TTS] Raspberry Pi using Piper")

            # Generate audio from Piper
            process = subprocess.Popen(
                ["piper", "--model", PIPER_MODEL, "--output_audio", "-"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE
            )
            process.stdin.write(text.encode("utf-8"))
            process.stdin.close()

            audio = process.stdout.read()

            # Play through aplay
            play = subprocess.Popen(["aplay"], stdin=subprocess.PIPE)
            play.stdin.write(audio)
            play.stdin.close()
            return

        # Fallback ONLY if Piper isn’t installed yet
        print("[TTS] Raspberry Pi fallback -> espeak-ng")
        cmd = ["espeak-ng"]
        if voice:
            cmd += ["-v", voice]
        cmd.append(text)
        subprocess.run(cmd)
        return

    # -----------------------------------------
    # OTHER SYSTEMS
    # -----------------------------------------
    else:
        print("[TTS] Unsupported OS:", system)
        return
