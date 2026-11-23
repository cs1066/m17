### m17/mplan.py
#
# A meal-planning web app that combines the ideas in vmemo3.py and
# inventory2.py. The user:
#
#   1. Dictates (or types) the ingredients for a meal; Gemini turns that
#      into an ingredient list, which the user edits and marks as accurate.
#   2. Uploads a photo of their refrigerator; Gemini lists the food items
#      it can see, and the user edits that list and marks it as accurate.
#   3. Gemini compares the two lists, and the app builds a grocery list of
#      the ingredients the user doesn't have. The user edits that list,
#      marks it as accurate, and emails it to themselves.
#
# Each list is saved as a JSON file in the mealplans/ folder when the user
# marks it as accurate.
#
# Run with:  streamlit run mplan.py
#
# Authors: Mike Smith, using Anthropic Claude Opus 5.5 to combine vmemo3.py
#          and inventory2.py.
# Date:    20261122

import base64
import datetime
import json
from pathlib import Path
from typing import Literal
from urllib.parse import quote

import pandas as pd
import streamlit as st
from google import genai
from pydantic import BaseModel, Field

# Defaults
AI_MODEL = "gemini-3.5-flash-lite"
PLANS_DIR = Path(__file__).parent / "mealplans"
RECORDINGS_DIR = Path(__file__).parent / "recordings"

# Who receives the emailed grocery list, unless the user types another
# address into the app
EMAIL_TO = "mick191913331@gmail.com"

# The three lists the user reviews, in order. Each one is shown as an
# editable table until the user marks it as accurate.
STEPS = ["ingredients", "inventory", "grocery"]


# ---------------------------------------------------------------------------
# The shapes of the data. As in vmemo3.py and inventory2.py, each Pydantic
# class does two jobs: model_json_schema() tells Gemini what structure to
# produce, and model_validate_json() checks Gemini's reply. We also use
# them to clean up the user's edited tables before saving.

# Step 1: the ingredients for a meal, as Gemini hears (or reads) them
class Ingredient(BaseModel):
    name: str = Field(description="The ingredient, e.g., 'yellow onion'.")
    quantity: str = Field(
        default="unknown",
        description="The amount as stated, e.g., '2 cups', or 'unknown'.",
    )

class Recipe(BaseModel):
    meal: str = Field(description="The name of the meal, or 'unknown'.")
    ingredients: list[Ingredient]

# The user's version of an ingredient adds one column Gemini never sets:
# a checkbox for things like salt and olive oil, which the user knows
# they have but which won't show up in a photo of the refrigerator.
class PlannedIngredient(Ingredient):
    already_have: bool = False

# Step 2: the food items Gemini sees in the photo (the same as inventory2.py)
class FoodItem(BaseModel):
    name: str = Field(description="The food item, if visibly identifiable.")
    quantity_estimate: str = Field(
        default="unknown",
        description="A rough visible amount, or 'unknown' if it cannot be estimated.",
    )
    uncertainty: str = Field(
        default="",
        description="Note ambiguity, occlusion, or uncertainty; otherwise an empty string.",
    )

class Inventory(BaseModel):
    items: list[FoodItem]

# Step 3: Gemini's comparison of the two lists, one Match per ingredient
class Match(BaseModel):
    ingredient: str
    status: Literal["have", "maybe", "missing"]
    fridge_item: str = Field(
        description="The matching item from the fridge list, or '' if missing."
    )
    reason: str = Field(description="A short explanation of the status.")

class Comparison(BaseModel):
    matches: list[Match]

# ... and the grocery list the user edits from that comparison
class GroceryItem(BaseModel):
    buy: bool = True
    name: str
    quantity: str = "unknown"
    note: str = ""


# ---------------------------------------------------------------------------
# Talking to Gemini

def ask_gemini(parts, schema):
    """Send a list of input parts to Gemini and return its reply as a
    validated instance of the Pydantic class `schema`."""
    client = genai.Client()
    result = client.interactions.create(
        model=AI_MODEL,
        input=parts,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": schema.model_json_schema(),
        },
    )
    return schema.model_validate_json(result.output_text)


def text_part(text):
    return {"type": "text", "text": text}


def media_part(kind, data, mime_type):
    """An audio or image input part. As in vmemo3.py, the binary data is
    base64-encoded so it can travel inside the JSON request."""
    return {
        "type": kind,
        "data": base64.b64encode(data).decode("utf-8"),
        "mime_type": mime_type,
    }


RECIPE_INSTRUCTIONS = (
    "Extract the name of the meal (or 'unknown') and each ingredient with "
    "its quantity as stated (or 'unknown' if no amount is given). List only "
    "ingredients that are actually mentioned; don't add any."
)


def recipe_from_audio(wav_bytes, mime_type):
    return ask_gemini(
        [text_part("This recording lists the ingredients for a meal. "
                   + RECIPE_INSTRUCTIONS),
         media_part("audio", wav_bytes, mime_type)],
        Recipe,
    )


def recipe_from_text(text):
    return ask_gemini(
        [text_part("The text below lists the ingredients for a meal. "
                   + RECIPE_INSTRUCTIONS + "\n\n" + text)],
        Recipe,
    )


def inventory_from_image(image_bytes, mime_type):
    return ask_gemini(
        [text_part("List food items that are visibly identifiable in this photo. "
                   "Do not guess about items hidden behind other objects. "
                   "Use 'unknown' when the quantity cannot be estimated, and note "
                   "uncertain identifications."),
         media_part("image", image_bytes, mime_type)],
        Inventory,
    )


def compare_lists(ingredient_names, fridge_names):
    """Ask Gemini which ingredients are in the fridge.

    Exact string matching doesn't work here: the recipe says "scallions"
    and the fridge list says "green onions". So we let Gemini judge, and
    ask it to say "maybe" when it isn't sure, leaving the decision to the
    user.
    """
    comparison = ask_gemini(
        [text_part(
            "Compare a recipe's ingredients with the food items found in a "
            "refrigerator. Give exactly one result for each ingredient, with "
            "status 'have' if a fridge item is clearly the same food (e.g., "
            "'scallions' and 'green onions'); 'maybe' if a fridge item might "
            "be it or could substitute for it, but you aren't sure (e.g., "
            "'cheddar' and 'yellow cheese'); or 'missing' if nothing in the "
            "fridge matches. Ignore quantities.\n\n"
            f"Ingredients: {json.dumps(ingredient_names)}\n"
            f"Fridge items: {json.dumps(fridge_names)}"
        )],
        Comparison,
    )

    # Don't let an ingredient silently disappear if Gemini skipped it.
    answered = {m.ingredient.lower() for m in comparison.matches}
    for name in ingredient_names:
        if name.lower() not in answered:
            comparison.matches.append(Match(
                ingredient=name, status="missing", fridge_item="",
                reason="Gemini did not report on this ingredient.",
            ))
    return comparison


# ---------------------------------------------------------------------------
# Tables, files, and email

def to_table(rows, model):
    """Turn a list of dicts into a DataFrame with one column per field of
    the Pydantic class `model`, in the order the class lists them."""
    return pd.DataFrame(rows, columns=list(model.model_fields))


def clean_table(table, model):
    """Turn an edited table back into a tidy DataFrame.

    Rows the user added but left without a name are dropped. Empty cells
    (which Streamlit gives us as None) are left out, so the Pydantic class
    fills in its default value for them.
    """
    rows = []
    for row in table.to_dict(orient="records"):
        row = {key: value for key, value in row.items() if not pd.isna(value)}
        if str(row.get("name", "")).strip() != "":
            rows.append(model(**row).model_dump())
    return to_table(rows, model)


def save_json(kind, record):
    """Write a record to mealplans/mplan-YYYYMMDDHHMM-<kind>.json.

    All of the files from one planning session share the same timestamp,
    so they're easy to find together.
    """
    PLANS_DIR.mkdir(exist_ok=True)
    path = PLANS_DIR / f"mplan-{st.session_state.stamp}-{kind}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    return path


def email_link(address, meal, groceries):
    """Build a mailto: link that opens a new draft of the grocery list.

    In vmemo3.py we asked the operating system to open the link. In a web
    app, the browser is the user's computer, so we hand the link to the
    browser as a button instead (see st.link_button below).
    """
    lines = []
    for item in groceries:
        if item["quantity"] and item["quantity"] != "unknown":
            lines.append(f"- {item['name']} ({item['quantity']})")
        else:
            lines.append(f"- {item['name']}")
    subject = f"Grocery list for {meal}"
    body = f"{subject}:\n\n" + "\n".join(lines) + "\n"
    # quote() replaces characters that aren't allowed in a URL, such as
    # spaces and newlines, with %-codes.
    return f"mailto:{address}?subject={quote(subject)}&body={quote(body)}"


# ---------------------------------------------------------------------------
# Keeping track of where the user is
#
# Streamlit re-runs this whole script every time the user interacts with
# the page, so everything we need to remember lives in st.session_state.
# For each step in STEPS we keep:
#   <step>_table    the list as a DataFrame
#   <step>_ok       True once the user has marked the list as accurate
#   <step>_version  a counter that's part of the editor widget's key; we
#                   bump it to give the user a fresh editor on new data
#   <step>_path     where the list was saved

def init_state():
    if "stamp" in st.session_state:
        return
    st.session_state.stamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    st.session_state.meal = ""
    st.session_state.comparison = None
    st.session_state.audio_id = None
    st.session_state.image_id = None
    st.session_state.image_name = ""
    st.session_state.source = ""
    for step in STEPS:
        st.session_state[f"{step}_table"] = None
        st.session_state[f"{step}_ok"] = False
        st.session_state[f"{step}_version"] = 0
        st.session_state[f"{step}_path"] = None


def set_table(step, table):
    """Replace a step's list with new data and reopen it for editing."""
    st.session_state[f"{step}_table"] = table
    st.session_state[f"{step}_version"] += 1
    reopen(step)


def reopen(step):
    """Mark a step as not yet accurate. If the ingredients or the fridge
    list change, the comparison and grocery list are out of date too."""
    st.session_state[f"{step}_ok"] = False
    if step != "grocery":
        st.session_state.comparison = None
        st.session_state.grocery_ok = False


def review_step(step, column_config, accept_label, on_accept):
    """Show one step's list: editable until the user marks it as accurate,
    then read-only with an "Edit again" button.

    on_accept(table) is called with the cleaned-up table when the user
    marks it as accurate, and returns the path where it was saved.
    """
    table = st.session_state[f"{step}_table"]

    if st.session_state[f"{step}_ok"]:
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config=column_config)
        st.caption(f"Saved to {st.session_state[f'{step}_path']}")
        if st.button("Edit again", key=f"{step}-reopen"):
            reopen(step)
            st.rerun()
        return

    st.caption(
        "Click a cell to edit it. Use the row checkboxes and the trash "
        "icon to delete rows, and the + row at the bottom to add one."
    )
    # As in inventory2.py, st.data_editor gives us editing, deleting, and
    # adding rows for free.
    edited = st.data_editor(
        table,
        key=f"{step}-editor-{st.session_state[f'{step}_version']}",
        num_rows="dynamic",
        hide_index=True,
        width="stretch",
        column_config=column_config,
    )
    if st.button(accept_label, type="primary", key=f"{step}-accept"):
        model = {"ingredients": PlannedIngredient, "inventory": FoodItem,
                 "grocery": GroceryItem}[step]
        final = clean_table(edited, model)
        # Store the edited version and give the editor a new key, so that
        # if the user chooses "Edit again" they start from these edits.
        st.session_state[f"{step}_table"] = final
        st.session_state[f"{step}_version"] += 1
        st.session_state[f"{step}_path"] = on_accept(final)
        st.session_state[f"{step}_ok"] = True
        st.rerun()


# ---------------------------------------------------------------------------
# The three steps of the app

def ingredients_step():
    st.header("1. What's in the meal?")

    if not st.session_state.ingredients_ok:
        dictate_tab, type_tab = st.tabs(["Dictate", "Type"])

        with dictate_tab:
            audio = st.audio_input("Record the ingredients for your meal")
            # Only call Gemini when there's a *new* recording.
            if audio is not None and audio.file_id != st.session_state.audio_id:
                wav_bytes = audio.getvalue()
                RECORDINGS_DIR.mkdir(exist_ok=True)
                wav_path = RECORDINGS_DIR / f"mplan-{st.session_state.stamp}.wav"
                wav_path.write_bytes(wav_bytes)
                with st.spinner("Asking Gemini to list the ingredients..."):
                    try:
                        recipe = recipe_from_audio(wav_bytes, audio.type)
                    except Exception as e:
                        st.error(f"Gemini could not understand the recording: {e}")
                        return
                st.session_state.audio_id = audio.file_id
                use_recipe(recipe, source=wav_path.name)

        with type_tab:
            text = st.text_area(
                "Type or paste the ingredients for your meal",
                placeholder="Chicken stir-fry: 1 lb chicken breast, 2 bell "
                            "peppers, 3 scallions, soy sauce, rice...",
            )
            if st.button("Get ingredient list") and text.strip():
                with st.spinner("Asking Gemini to list the ingredients..."):
                    try:
                        recipe = recipe_from_text(text)
                    except Exception as e:
                        st.error(f"Gemini could not read the ingredients: {e}")
                        return
                use_recipe(recipe, source="typed")

    if st.session_state.ingredients_table is None:
        st.info("Dictate or type the ingredients to get started.")
        return

    st.text_input("Meal", key="meal",
                  disabled=st.session_state.ingredients_ok)
    st.caption("Check **Already have** for staples, like salt or oil, that "
               "won't show up in a photo of your refrigerator.")
    review_step(
        "ingredients",
        column_config={
            "name": st.column_config.TextColumn("Ingredient", required=True),
            "quantity": st.column_config.TextColumn("Quantity"),
            "already_have": st.column_config.CheckboxColumn("Already have",
                                                            default=False),
        },
        accept_label="Ingredient list is accurate",
        on_accept=lambda table: save_json("ingredients", {
            "meal": st.session_state.meal,
            "source": st.session_state.source,
            "ingredients": table.to_dict(orient="records"),
        }),
    )


def use_recipe(recipe, source):
    """Load Gemini's ingredient list into the editor."""
    st.session_state.meal = recipe.meal
    st.session_state.source = source
    set_table("ingredients", to_table(
        [PlannedIngredient(**i.model_dump()).model_dump()
         for i in recipe.ingredients],
        PlannedIngredient,
    ))


def inventory_step():
    st.header("2. What's in your refrigerator?")

    uploaded = None
    if not st.session_state.inventory_ok:
        uploaded = st.file_uploader(
            "Choose a photo of your refrigerator",
            type=["jpg", "jpeg", "png", "webp"],
        )
        # As in inventory2.py, only call Gemini when a *new* image is chosen.
        if uploaded is not None and uploaded.file_id != st.session_state.image_id:
            with st.spinner("Asking Gemini to identify the food items..."):
                try:
                    inventory = inventory_from_image(uploaded.getvalue(),
                                                     uploaded.type)
                except Exception as e:
                    st.error(f"Gemini could not analyze the image: {e}")
                    return
            st.session_state.image_id = uploaded.file_id
            st.session_state.image_name = uploaded.name
            set_table("inventory", to_table(
                [item.model_dump() for item in inventory.items], FoodItem))

    if st.session_state.inventory_table is None:
        st.info("Select an image of your refrigerator.")
        return

    image_col, list_col = st.columns([2, 3])
    with image_col:
        if uploaded is not None:
            st.image(uploaded, caption=uploaded.name, width="stretch")
        else:
            st.caption(f"Photo: {st.session_state.image_name}")
    with list_col:
        review_step(
            "inventory",
            column_config={
                "name": st.column_config.TextColumn("Item", required=True),
                "quantity_estimate": st.column_config.TextColumn("Quantity"),
                "uncertainty": st.column_config.TextColumn("Notes"),
            },
            accept_label="Fridge list is accurate",
            on_accept=lambda table: save_json("inventory", {
                "image": st.session_state.image_name,
                "items": table.to_dict(orient="records"),
            }),
        )


def grocery_step():
    st.header("3. Your grocery list")

    ingredients = st.session_state.ingredients_table
    to_check = ingredients[~ingredients["already_have"]]

    # Compare the lists once, each time either list is marked accurate.
    if st.session_state.comparison is None:
        fridge_names = list(st.session_state.inventory_table["name"])
        if to_check.empty:
            comparison = Comparison(matches=[])
        else:
            with st.spinner("Asking Gemini to compare the two lists..."):
                try:
                    comparison = compare_lists(list(to_check["name"]),
                                               fridge_names)
                except Exception as e:
                    st.error(f"Gemini could not compare the lists: {e}")
                    return
        st.session_state.comparison = comparison
        set_table("grocery", grocery_table(comparison, to_check))

    comparison = st.session_state.comparison
    with st.expander("How Gemini compared the lists"):
        st.dataframe(
            to_table([m.model_dump() for m in comparison.matches], Match),
            hide_index=True, width="stretch",
        )

    st.caption("Ingredients Gemini wasn't sure about are listed with "
               "**Buy** unchecked. Check the ones you need.")
    review_step(
        "grocery",
        column_config={
            "buy": st.column_config.CheckboxColumn("Buy", default=True),
            "name": st.column_config.TextColumn("Item", required=True),
            "quantity": st.column_config.TextColumn("Quantity"),
            "note": st.column_config.TextColumn("Note"),
        },
        accept_label="Grocery list is accurate",
        on_accept=lambda table: save_json("grocery", {
            "meal": st.session_state.meal,
            "comparison": comparison.model_dump()["matches"],
            "groceries": table.to_dict(orient="records"),
        }),
    )

    if st.session_state.grocery_ok:
        email_step()


def grocery_table(comparison, ingredients):
    """Build the starting grocery list from Gemini's comparison: every
    "missing" ingredient (checked to buy) and every "maybe" ingredient
    (unchecked, with a note saying what Gemini thinks might match)."""
    quantities = dict(zip(ingredients["name"].str.lower(),
                          ingredients["quantity"]))
    rows = []
    for m in comparison.matches:
        if m.status == "have":
            continue
        note = "" if m.status == "missing" else f"Maybe: {m.fridge_item}. {m.reason}"
        rows.append(GroceryItem(
            buy=(m.status == "missing"),
            name=m.ingredient,
            quantity=quantities.get(m.ingredient.lower(), "unknown"),
            note=note,
        ).model_dump())
    return to_table(rows, GroceryItem)


def email_step():
    groceries = [row for row in
                 st.session_state.grocery_table.to_dict(orient="records")
                 if row["buy"]]
    if not groceries:
        st.success("You have everything you need. Nothing to buy!")
        return

    address = st.text_input("Email the grocery list to", value=EMAIL_TO)
    valid = "@" in address and "." in address.split("@")[-1]
    st.link_button(
        "Email grocery list",
        email_link(address.strip(), st.session_state.meal or "your meal",
                   groceries),
        type="primary",
        disabled=not valid,
    )
    st.caption("This opens a draft in your email program. Review it and "
               "click Send there.")


def main():
    st.set_page_config(page_title="Meal Planner", layout="wide")
    st.title("Meal Planner")
    init_state()

    with st.sidebar:
        if st.button("Start over"):
            st.session_state.clear()
            st.rerun()

    # Each step appears once the one before it is marked as accurate.
    ingredients_step()
    if st.session_state.ingredients_ok:
        inventory_step()
    if st.session_state.ingredients_ok and st.session_state.inventory_ok:
        grocery_step()


main()
