### m17/vmemo3_test.py
#
# Test the email step of vmemo3.py without recording audio or calling
# Gemini. It uses a fixed set of notes, as if they had come back from the
# model, and a made-up recording filename.
#
# Authors: Mike Smith, with Anthropic Claude Opus 5.5 Medium.
# Date:    20261121

from vmemo3 import RECORDINGS_DIR, MemoNotes, email_notes

# Notes printed by an earlier run of vmemo3.py
NOTES_JSON = """
{
  "main_points": [
    "Pick up dry cleaning on the way home",
    "Get a bottle of milk from the store tomorrow",
    "Take the dog for a walk this weekend"
  ],
  "action_items": [
    {
      "task": "Pick up dry cleaning",
      "due_date": "on the way home"
    },
    {
      "task": "Get a bottle of milk from the store",
      "due_date": "tomorrow"
    },
    {
      "task": "Take the dog for a walk",
      "due_date": "this weekend"
    }
  ]
}
"""


def main():
    # Stand-ins for what vmemo3.main() gets from the microphone and Gemini.
    # The recording file doesn't need to exist; only its name is emailed.
    wav_path = RECORDINGS_DIR / "test-202611211042.wav"
    notes = MemoNotes.model_validate_json(NOTES_JSON)

    # From here on, the same steps as vmemo3.main()
    notes_json = notes.model_dump_json(indent=2)
    print(notes_json)

    email_notes(wav_path, notes_json)


if __name__ == "__main__":
    main()
