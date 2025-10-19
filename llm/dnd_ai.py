import subprocess
import os
import json
import datetime
import re
from typing import Optional
import uuid

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3")

import requests


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


async def extract_session_data(transcript):
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
      {{"name": "Name or alias of the character", "race": "if known or unknown", "class":"if known or unknown", "npc": true or false}}
  ],
  "locations": [{{"location_name": "any location referenced in the session, such as a named place or a notable location (e.g., tavern)"}}],
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
    # Call your LLM
    response = run_ollama(prompt)
    # 🔹 DEBUG: print the raw response

    # Clean up response: strip code fences
    response = re.sub(r"```(?:json)?", "", response).strip()

    structured = {}
    try:
        # extract the JSON portion (first { ... } block)
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            json_str = match.group(0)
            structured = json.loads(json_str)
    except Exception:
        structured = {}

    # Wrap into the schema expected by save_session_to_chroma
    session_data = {
        "session_id": datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
        "campaign_id": 0,  # update if you have campaign context
        "summary": {
            "session_summary": structured.get("session_summary", ""),
            "characters": structured.get("characters", []),
            "locations": structured.get("locations", []),
            "events": structured.get("events", []),
        },
        "processed_at": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
    }
    # 🔥 Initialise stats for each character
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

    for char in session_data["summary"]["characters"]:
        # ensure a unique character_id
        char.setdefault("character_id", str(uuid.uuid4())[:6])
        # merge defaults without overwriting existing keys
        for stat, val in default_stats.items():
            char.setdefault(stat, val)

    # assign UUIDs to locations
    for i, loc in enumerate(session_data["summary"]["locations"]):
        if isinstance(loc, dict):
            loc.setdefault("location_id", str(uuid.uuid4())[:6])
        else:
            # if locations are just strings, wrap them
            session_data["summary"]["locations"][i] = {
                "location_id": str(uuid.uuid4())[:6],
                "location_name": loc,
            }

    # assign UUIDs and default fields to events
    for i, ev in enumerate(session_data["summary"]["events"]):
        if isinstance(ev, dict):
            ev.setdefault("event_id", str(uuid.uuid4())[:6])
            ev.setdefault("event_summary", "")
            ev.setdefault("participants", [])
            ev.setdefault("location", "")
            ev.setdefault("timeline_order", i + 1)
            ev.setdefault("event_tags", ["miscellaneous"])
        else:
            # if events are just strings, wrap them
            session_data["summary"]["events"][i] = {
                "event_id": str(uuid.uuid4())[:6],
                "event": ev,
                "event_summary": "",
                "participants": [],
                "location": "",
                "timeline_order": i + 1,
                "event_tags": ["miscellaneous"],
            }

    return session_data


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

    # Extract structured data
    result = extract_session_data(sample_transcript)

    # 🔹 Filter out only the events (if present)
    events = result.get("summary", {}).get("events", [])

    # Print just the events for inspection
    print("=== Extracted Events ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    test_extract_session_data()
