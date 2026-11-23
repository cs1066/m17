### m17/inventory1.py
#
# Given an image of the contents of a refrigerator, identify and list them.
#
# Authors: Mike Smith, using OpenAI GPT-6 Luna for first draft of this code and
#          Anthropic Claude Opus 5.5 Medium for the later improvements.
# Date:    20261122

import base64
from google import genai
from pydantic import BaseModel, Field

# Defaults
AI_MODEL = "gemini-3.5-flash-lite"
IMAGE_PATH = "images/fridge.jpg"


# The shape of the summary we want back from Gemini. These Pydantic classes
# do two jobs: Inventory.model_json_schema() turns them into a JSON Schema
# that tells the model exactly what structure to produce, and
# Inventory.model_validate_json() later checks that the model's reply
# actually matches that structure. A Inventory holds a list of food items,
# each with a name, a rough quantity estimate, and a note about uncertainty.
class FoodItem(BaseModel):
    name: str = Field(description="The food item, if visibly identifiable.")
    quantity_estimate: str = Field(
        description="A rough visible amount, or 'unknown' if it cannot be estimated."
    )
    uncertainty: str = Field(
        description="Note ambiguity, occlusion, or uncertainty; otherwise an empty string."
    )

class Inventory(BaseModel):
    items: list[FoodItem]


def main():
    # Connect to the Gemini API and analyze the image
    client = genai.Client()

    with open(IMAGE_PATH, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    result = client.interactions.create(
        model=AI_MODEL,
        input=[
            {
                "type": "text",
                "text": (
                    "List food items that are visibly identifiable in this photo. "
                    "Do not guess about items hidden behind other objects. "
                    "Use 'unknown' when the quantity cannot be estimated, and note "
                    "uncertain identifications."
                ),
            },
            {
                "type": "image",
                "data": image_b64,
                "mime_type": "image/jpeg",
            },
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": Inventory.model_json_schema(),
        },
    )

    inventory = Inventory.model_validate_json(result.output_text)

    for item in inventory.items:
        print(item.name, "—", item.quantity_estimate, item.uncertainty)

if __name__ == "__main__":
    main()
