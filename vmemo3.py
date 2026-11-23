### m17/vmemo3.py
#
# Record a voice memo from the laptop's microphone, write a summary of it,
# including action items and due dates, and open an email of the summary
# in the computer's default email program.
#
# Authors: Mike Smith, using OpenAI GPT-6 Luna for first draft of this code and
#          Anthropic Claude Opus 5.5 Medium for the later improvements.
# Date:    20261121

import base64
import io
import os
import subprocess
import sys
import threading
import wave
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import sounddevice as sd
from google import genai
from pydantic import BaseModel

# Recording settings. 16 kHz, mono, 16-bit samples is plenty for speech
# and keeps the recording small.
SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes per sample (16-bit)
BLOCK_FRAMES = SAMPLE_RATE // 10  # samples per read: 0.1 seconds of audio

# Who receives the emailed summary, and its subject line
# EMAIL_TO = "mike_smith@harvard.edu"
EMAIL_TO = "mick191913331@gmail.com"
EMAIL_SUBJECT = "To do"

# Where to save a copy of each recording: the recordings/ folder next to
# this script, no matter which directory the script is run from.
RECORDINGS_DIR = Path(__file__).parent / "recordings"

# The shape of the summary we want back from Gemini. These Pydantic classes
# do two jobs: MemoNotes.model_json_schema() turns them into a JSON Schema
# that tells the model exactly what structure to produce, and
# MemoNotes.model_validate_json() later checks that the model's reply
# actually matches that structure. A MemoNotes holds a list of the memo's
# main points and a list of ActionItems, each with a task and a due date.
class ActionItem(BaseModel):
    task: str
    due_date: str  # Use "unknown" if the recording doesn't say.

class MemoNotes(BaseModel):
    main_points: list[str]
    action_items: list[ActionItem]


def record_memo():
    """Record from the default microphone until the user presses Enter.

    Returns the recording as the bytes of a WAV file.
    """
    chunks = []

    # A helper thread waits for the user to press Enter and then sets this
    # flag. It is a daemon thread, so it won't keep the program alive if
    # we exit some other way (e.g., Ctrl+C).
    stop_requested = threading.Event()

    def wait_for_enter():
        input("Recording... press Enter to stop.")
        stop_requested.set()

    input("Press Enter to start recording...")

    # Open the stream WITHOUT a callback and pull audio from it ourselves.
    # With a callback, PortAudio runs our Python code on its own audio
    # thread, and shutting the stream down could deadlock with that thread
    # and hang the program (ignoring Ctrl+C). Reading in a loop keeps all
    # of our Python code on the main thread.
    stream = sd.RawInputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                               dtype="int16")
    stream.start()
    threading.Thread(target=wait_for_enter, daemon=True).start()
    try:
        # Each read() waits for one block (0.1 seconds) of audio, so we
        # notice the stop flag, or a Ctrl+C, within a tenth of a second.
        while not stop_requested.is_set():
            data, overflowed = stream.read(BLOCK_FRAMES)
            chunks.append(bytes(data))
    finally:
        # abort() stops immediately without waiting for leftover audio.
        stream.abort()
        stream.close()

    if not chunks:
        sys.exit("No audio was recorded.")

    # Package the raw samples as a WAV file, held in memory rather than
    # written to disk. The WAV header records the sample rate, channel
    # count, and sample width so a listener (here, Gemini) can play it back.
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(CHANNELS)
        wav.setsampwidth(SAMPLE_WIDTH)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(b"".join(chunks))
    return buffer.getvalue()


def main():
    # Capture the voice memo from the microphone
    wav_bytes = record_memo()

    # Save a copy of the recording, named with the current date and time
    # (YYYYMMDDHHMM), e.g., recordings/test-202611181530.wav
    RECORDINGS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    wav_path = RECORDINGS_DIR / f"test-{timestamp}.wav"
    wav_path.write_bytes(wav_bytes)
    print(f"Saved recording to {wav_path}")

    # Connect to the Gemini API and summarize the voice memo
    client = genai.Client()

    # Encode the recording in base64, i.e., every 6 bits of the input
    # becomes 1 of 64 common characters. The final decode turns the bytes
    # object into a string, which is what the API expects. Base64 encoding
    # is a common way to send binary data in a text-based format like JSON.
    audio_data = base64.b64encode(wav_bytes).decode("utf-8")

    result = client.interactions.create(
        # model="gemini-3.8-flash",
        model="gemini-3.5-flash-lite",
        input=[
            {
                "type": "text",
                "text": (
                    "Summarize this voice memo. List only action items that are "
                    "actually stated; don't invent tasks or due dates. "
                    'Use "unknown" for a due date that is not stated.'
                ),
            },
            {
                "type": "audio",
                "data": audio_data,
                "mime_type": "audio/wav",
            },
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": MemoNotes.model_json_schema(),
        },
    )

    notes = MemoNotes.model_validate_json(result.output_text)
    notes_json = notes.model_dump_json(indent=2)
    print(notes_json)

    email_notes(wav_path, notes_json)


def email_notes(wav_path, notes_json):
    """Open a message to EMAIL_TO in the default email program.

    Builds a mailto: link and asks the operating system to open it. That
    creates a new draft with the recipient, subject, and body filled in;
    you review it and click Send in your email program.
    """
    body = f"Recording: {wav_path.name}\n\n{notes_json}\n"

    # Characters like spaces, newlines, and quotes aren't allowed in a URL,
    # so quote() replaces each with a %-code (e.g., a space becomes %20).
    url = f"mailto:{EMAIL_TO}?subject={quote(EMAIL_SUBJECT)}&body={quote(body)}"

    # Hand the link to the operating system, which opens it with whatever
    # program is registered for mailto: links (your email program).
    # Note: Python's webbrowser module does NOT work for this on macOS: it
    # sends any non-web link to the default *web browser*, not the default
    # email program.
    try:
        if sys.platform == "darwin":        # macOS
            subprocess.run(["open", url], check=True)
        elif sys.platform == "win32":       # Windows
            os.startfile(url)
        else:                               # Linux and others
            subprocess.run(["xdg-open", url], check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"Could not open your email program: {e}", file=sys.stderr)
    else:
        print(f"Opened an email to {EMAIL_TO} in your email program.")


if __name__ == "__main__":
    main()
