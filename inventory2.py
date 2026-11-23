### m17/inventory2.py
#
# A web-app version of inventory1.py. The user picks a photo of the contents
# of a refrigerator, Gemini identifies the food items in it, and the user
# can then review and edit that list (change items, delete items, add new
# ones) before saving it as a JSON file.
#
# Run with:  streamlit run inventory2.py
#
# Authors: Mike Smith, using Anthropic Claude Opus 5.5 to turn inventory1.py
#          into a Streamlit web app.
# Date:    20261122

import base64
import datetime
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from google import genai
from pydantic import BaseModel, Field

# Defaults
AI_MODEL = "gemini-3.5-flash-lite"
INVENTORY_DIR = Path(__file__).parent / "inventories"


# The same Pydantic classes as in inventory1.py. Inventory.model_json_schema()
# tells Gemini what structure to produce, and Inventory.model_validate_json()
# checks its reply. We also use Inventory to validate the user's edited list
# before saving it.
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


def analyze_image(image_bytes, mime_type):
    """Send the image to Gemini and return the Inventory it identifies."""
    client = genai.Client()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

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
                "mime_type": mime_type,
            },
        ],
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": Inventory.model_json_schema(),
        },
    )

    return Inventory.model_validate_json(result.output_text)


def table_to_inventory(table):
    """Turn the edited table back into an Inventory.

    Rows the user added but left blank are dropped, and any empty cells
    (which Streamlit gives us as None) become empty strings.
    """
    table = table.fillna("")
    items = [
        FoodItem(**row)
        for row in table.to_dict(orient="records")
        if str(row["name"]).strip() != ""
    ]
    return Inventory(items=items)


def save_inventory(inventory, image_name):
    """Write the inventory to inventories/inventory-YYYYMMDDHHMM.json."""
    INVENTORY_DIR.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    path = INVENTORY_DIR / f"inventory-{timestamp}.json"
    record = {"image": image_name, **inventory.model_dump()}
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return path


def main():
    st.set_page_config(page_title="Fridge Inventory", layout="wide")
    st.title("Fridge Inventory")

    # Step 1: The user picks an image.
    uploaded = st.file_uploader(
        "Choose a photo of your refrigerator",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded is None:
        st.info("Select an image to get started.")
        return

    # Step 2: Ask Gemini to analyze it. Streamlit re-runs this whole script
    # every time the user interacts with the page, so we remember the result
    # in st.session_state and only call Gemini when a *new* image is chosen.
    if st.session_state.get("file_id") != uploaded.file_id:
        with st.spinner("Asking Gemini to identify the food items..."):
            try:
                inventory = analyze_image(uploaded.getvalue(), uploaded.type)
            except Exception as e:
                st.error(f"Gemini could not analyze the image: {e}")
                return
        st.session_state.file_id = uploaded.file_id
        st.session_state.ai_table = pd.DataFrame(
            [item.model_dump() for item in inventory.items],
            columns=list(FoodItem.model_fields),
        )
        st.session_state.saved_path = None

    # Step 3: Show the image next to an editable version of the list.
    image_col, list_col = st.columns([2, 3])

    with image_col:
        st.image(uploaded, caption=uploaded.name, width="stretch")

    with list_col:
        st.subheader("Items found")
        st.caption(
            "Click a cell to edit it. Use the row checkboxes and the trash "
            "icon to delete items, and the + row at the bottom to add one."
        )
        # st.data_editor gives us editing, deleting, and adding rows for
        # free. We pass in Gemini's original table every time; Streamlit
        # keeps track of the user's changes under the widget's key.
        edited = st.data_editor(
            st.session_state.ai_table,
            key=f"editor-{uploaded.file_id}",
            num_rows="dynamic",
            hide_index=True,
            width="stretch",
            column_config={
                "name": st.column_config.TextColumn("Item", required=True),
                "quantity_estimate": st.column_config.TextColumn("Quantity"),
                "uncertainty": st.column_config.TextColumn("Notes"),
            },
        )

        # Step 4: Save the final list as JSON.
        final = table_to_inventory(edited)
        st.write(f"{len(final.items)} item(s) in the list.")

        if st.button("Save list as JSON", type="primary"):
            st.session_state.saved_path = save_inventory(final, uploaded.name)

        if st.session_state.saved_path:
            st.success(f"Saved to {st.session_state.saved_path}")
            st.download_button(
                "Download a copy",
                data=st.session_state.saved_path.read_text(),
                file_name=st.session_state.saved_path.name,
                mime="application/json",
            )


main()
