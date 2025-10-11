import subprocess
import os
import json
import datetime
import re
from typing import Optional


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3")


import requests


def run_ollama(prompt: str, model: str = None) -> str:
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


def extract_session_data(transcript):
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
   - Ignore any non-game or non-D&D story related discussion.",
  "characters": [
      {{"name": "character's name or alias", "race": "...", "status": "alive/dead/etc"}} 
  ],
  "locations": ["..."],
  "events": [
      {{
          "event": "A title for the event.",
          "event_summary": "A text summary of what occurred during the event.",
          "participants": ["Any character involved in the event."],
          "location": "Location event took place.",
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
        "session_code": datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
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
    DM: The party enters the ancient ruins.
    Alice (Wizard): I cast a light spell to see inside.
    Bob (Fighter): I draw my sword and lead the way.
    DM: Suddenly, a giant spider descends from the ceiling!
    Alice: I try to use my magic to distract it.
    Bob: I attack with my sword!
    """

    result = extract_session_data(sample_transcript)

    # Print for inspection
    print(json.dumps(result, indent=2))


# # Run the test
# if __name__ == "__main__":
#     test_extract_session_data()
