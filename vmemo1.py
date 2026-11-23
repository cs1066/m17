### m17/vmemo1.py
#
# Write a summary of a voice memo, including action items and due dates.
#
# Authors: Mike Smith, using OpenAI GPT-6 Luna for first draft of this code and
#          Anthropic Claude Opus 5.5 Medium for the later improvements.
# Date:    20261118

import base64
import os
import sys
from google import genai
from pydantic import BaseModel

# Audio formats Gemini accepts, keyed by file extension
AUDIO_MIME_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mp3",
    ".m4a": "audio/m4a",
    ".aiff": "audio/aiff",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".flac": "audio/flac",
}

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


def main():
    # Grab and process the voice memo file
    filename = input("Voice memo filename: ").strip()

    ext = os.path.splitext(filename)[1].lower()
    mime_type = AUDIO_MIME_TYPES.get(ext)
    if mime_type is None:
        supported = ", ".join(AUDIO_MIME_TYPES)
        sys.exit(f"Unsupported file type '{ext}'. Supported: {supported}")

    # Connect to the Gemini API and summarize the voice memo
    client = genai.Client()

    # Read the audio file as bytes (not text) and encode it in base64,
    # i.e., every 6 bits of the input becomes 1 of 64 common characters.
    # The final decode turns the bytes object into a string, which is
    # what the API expects. Base64 encoding is a common way to send binary
    # data in a text-based format like JSON. 
    with open(filename, "rb") as f:
        audio_data = base64.b64encode(f.read()).decode("utf-8")

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
                "mime_type": mime_type,
            },
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": MemoNotes.model_json_schema(),
        },
    )

    notes = MemoNotes.model_validate_json(result.output_text)
    print(notes.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
