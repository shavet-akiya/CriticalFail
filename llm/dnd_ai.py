import os
import json
import datetime
import re
from typing import Optional
import uuid
import requests

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3")


def run_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    model = model or OLLAMA_MODEL
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {"model": model, "prompt": prompt}

    try:
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
    except requests.RequestException as e:
        return f"Ollama API error: {e}"

    output = ""
    for line in response.iter_lines():
        if not line:
            continue
        try:
            chunk = json.loads(line.decode("utf-8"))
            output += chunk.get("response", "")  # <- only append the actual text
        except Exception:
            continue

    return output.strip()


async def extract_session_data(
    transcript: str,
    existing_chars: Optional[list[dict]] = None,
    existing_locs: Optional[list[dict]] = None,
    campaign_id: Optional[str] = None,
):
    existing_chars = existing_chars or []
    existing_locs = existing_locs or []

    prompt = f"""
You are a D&D session scribe. Analyze the transcript below carefully.

Transcript:
\"\"\"{transcript}\"\"\"

Instructions:

1. Extract structured information as JSON with the following format:
{{
  "session_summary": "Summarize the key story events, quests, and lore that occurred in this session into readable text.  
   - Exclude gameplay mechanics such as dice rolls, skill checks, or combat mechanics.  
   - Include only story events - this may include important conversations between characters, battles, or quests.  
   - Ignore any non-game or non-D&D story related discussion.
  "characters": [
      {{"name": "Name or alias of unique character's in the session", "class": "if  or unknown", "race": "if known or unknown", "npc": true or false}}
  ],
  "locations": [{{"location_name": "any location referenced in the session, such as a named place or a notable location (e.g., tavern)", "location_description": "a description or definition of the location"}}],
  "events": [
      {{
          "event": "A title for the event.",
          "event_summary": "A text summary of what occurred during the event.",
          "participants": ["List of character names involved."],
          "location": "Location where event took place.",
          "timeline_order": 1
          "event_tags": Choose one or more from the following relevant to the event: "combat", "exploration", "player-to-player interaction", "npc interaction", "resting", "investigation", and "miscellaneous" miscellaneous should only be used if event is not relevant to any other tags.
      }}
  ],
}}

- The "timeline_order" field should indicate the chronological order of each event.  
- The "summary" field must contain the readable session summary text.  
- Only output valid JSON, with no explanatory text outside the JSON. 
- There are no limits to how many events may occur in a session, ensure to capture as many relevant story events as possible.
- Do not make up information that is not available in the transcript.
"""
    response = run_ollama(prompt)
    response = re.sub(r"```(?:json)?", "", response).strip()

    structured = {}
    try:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            structured = json.loads(match.group(0))
    except Exception:
        structured = {}

    session_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    processed_at = (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

    session_data = {
        "session_id": session_id,
        "campaign_id": campaign_id or 0,
        "summary": {
            "session_summary": structured.get("session_summary", ""),
            "characters": structured.get("characters", []),
            "locations": structured.get("locations", []),
            "events": structured.get("events", []),
        },
        "processed_at": processed_at,
    }

    # --- Characters ---
    default_stats = {
        "AC": 0,
        "HP": 0,
        "STR": 0,
        "DEX": 0,
        "CON": 0,
        "INT": 0,
        "WIS": 0,
        "CHA": 0,
    }
    campaign_chars = []

    for char in structured.get("characters", []):
        if isinstance(char, str):
            char = {"name": char}

        # reuse character_id if name exists
        existing = next(
            (c for c in existing_chars if c["name"] == char.get("name")), None
        )
        character_id = existing["character_id"] if existing else str(uuid.uuid4())[:6]

        # Append session ID if already exists
        session_ids = existing["session_ids"][:] if existing else []
        session_ids.append(session_id)

        full_char = {
            "character_id": character_id,
            "name": char.get("name"),
            "race": char.get("race", "Unknown"),
            "class": char.get("class", "Unknown"),
            "npc": char.get("npc", False),
            "session_ids": session_ids,
            **{k: char.get(k, v) for k, v in default_stats.items()},
        }
        campaign_chars.append(full_char)

    # Update session summary with consistent IDs
    for char in session_data["summary"]["characters"]:
        existing = next((c for c in campaign_chars if c["name"] == char["name"]), None)
        if existing:
            char["character_id"] = existing["character_id"]

    # --- Locations ---
    campaign_locs = []
    for i, loc in enumerate(structured.get("locations", [])):
        if isinstance(loc, str):
            loc = {"location_name": loc}

        existing = next(
            (
                l
                for l in existing_locs
                if l["location_name"] == loc.get("location_name")
            ),
            None,
        )
        location_id = existing["location_id"] if existing else str(uuid.uuid4())[:6]
        session_ids = existing["session_ids"][:] if existing else []
        session_ids.append(session_id)

        full_loc = {
            "location_id": location_id,
            "location_name": loc.get("location_name", f"Location {i+1}"),
            "location_description": loc.get(
                "location_description", "No description provided"
            ),
            "session_ids": session_ids,
        }
        campaign_locs.append(full_loc)

    # Update session summary locations with consistent IDs
    for loc in session_data["summary"]["locations"]:
        existing = next(
            (l for l in campaign_locs if l["location_name"] == loc["location_name"]),
            None,
        )
        if existing:
            loc["location_id"] = existing["location_id"]

    # --- Events ---
    for i, ev in enumerate(session_data["summary"]["events"]):
        if isinstance(ev, dict):
            ev.setdefault("event_id", str(uuid.uuid4())[:6])
            ev.setdefault("event_summary", "")
            ev.setdefault("participants", [])
            ev.setdefault("location", "")
            ev.setdefault("timeline_order", i + 1)
            ev.setdefault("event_tags", ["miscellaneous"])
        else:
            session_data["summary"]["events"][i] = {
                "event_id": str(uuid.uuid4())[:6],
                "event": ev,
                "event_summary": "",
                "participants": [],
                "location": "",
                "timeline_order": i + 1,
                "event_tags": ["miscellaneous"],
            }

    return session_data, campaign_chars, campaign_locs


def delete_from_json(data: dict, key: str, match: Optional[str] = None):
    """Delete a key or a specific item from JSON"""
    if key in data and match is None:
        del data[key]  # delete whole section
    elif key in data and isinstance(data[key], list) and match:
        data[key] = [
            item
            for item in data[key]
            if not (isinstance(item, dict) and match in item.values())
        ]
    return data


def clean_ollama_response(response: str) -> dict:
    # Remove code fences
    response = re.sub(r"```(?:json)?", "", response).strip()

    # Sometimes the AI outputs JSON with extra whitespace/newlines
    try:
        # Extract JSON object using regex
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)  # Parse to actual dict
    except json.JSONDecodeError:
        pass

    return {}  # fallback


def test_extract_session_data():
    sample_transcript = """
DM: Welcome everyone! Today’s adventure begins in the village of Green Hollow, a small settlement surrounded by dense forests.

Alice (Wizard, NPC): I check the shelves in the apothecary for potions that might help us.
Bob (Fighter, NPC): I stand by the entrance, keeping an eye out for any trouble.
Spooky George (Unknown, NPC): Makes a low growl, staring at the forest edge.

DM: Suddenly, a band of goblins emerges from the trees, brandishing crude weapons!

Alice: I cast Magic Missile at the nearest goblin.
Bob: I draw my sword and charge toward the goblins.
Spooky George: I attempt to intimidate the goblins with a fearsome roar.

DM: The goblins are taken aback by your coordinated attack. Alice's spell hits one goblin, Bob slashes another, and Spooky George's roar causes one to flee.

DM: After the battle, you notice a hidden path leading deeper into the forest. Along the path, there’s an old, abandoned shrine covered in moss.

Alice: I carefully examine the shrine for traps or magical wards.
Bob: I check the surrounding area for any signs of more enemies.
Spooky George: I investigate the shrine’s inscriptions, trying to understand its history.

DM: You discover that the shrine was dedicated to an ancient forest deity. A faint magical aura remains, but it seems dormant. You also find a small chest containing gold and a mysterious scroll.

    """

    # Mock existing campaign characters and locations
    existing_chars = [
        {"character_id": "char01", "name": "Alice", "session_ids": ["20251022020000"]},
        {"character_id": "char02", "name": "Bob", "session_ids": ["20251022020000"]},
    ]
    existing_locs = [
        {
            "location_id": "loc01",
            "location_name": "apothecary",
            "session_ids": ["20251022020000"],
        }
    ]

    # Extract structured data
    session_data, campaign_chars, campaign_locs = extract_session_data(
        sample_transcript,
        existing_chars=existing_chars,
        existing_locs=existing_locs,
        campaign_id="camp01",
    )

    print("=== Session Data ===")
    print(json.dumps(session_data, indent=2))

    print("\n=== Campaign Characters ===")
    for c in campaign_chars:
        print(c)

    print("\n=== Campaign Locations ===")
    for l in campaign_locs:
        print(l)


if __name__ == "__main__":
    test_extract_session_data()
