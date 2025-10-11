from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Optional
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

app = FastAPI()

speech_to_text = None
initialization_attempted = False


def initialize_stt():
    """Initialize SpeechToText with lazy loading of models"""
    global speech_to_text, initialization_attempted

    if initialization_attempted:
        return speech_to_text is not None

    initialization_attempted = True

    try:
        print("Attempting to initialize Speech Service...")
        from SpeechToText import SpeechToText

        speech_to_text = SpeechToText(save_folder="transcripts")

        if speech_to_text.models_loaded:
            print("Speech Service ready with all models")
        else:
            print("Speech Service initialized but some models may not be fully loaded")
        return True
    except Exception as e:
        print(f"Failed to initialize Speech Service: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Try to initialize on startup, but don't fail if it doesn't work"""
    initialize_stt()


@app.get("/")
async def root():
    is_ready = (
        speech_to_text is not None and speech_to_text.models_loaded
        if speech_to_text
        else False
    )
    return {
        "message": "Speech Processing Service",
        "status": "ready" if is_ready else "initializing",
        "models_loaded": speech_to_text.models_loaded if speech_to_text else False,
    }


@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    min_speakers: Optional[int] = Form(2),
    max_speakers: Optional[int] = Form(8),
):
    # Try to initialize if not already done
    if not speech_to_text:
        if not initialize_stt():
            return JSONResponse(
                status_code=503,
                content={"error": "Speech service failed to initialize"},
            )

    if not speech_to_text.models_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Speech service models not yet loaded, please try again"},
        )

    try:
        recordings_dir = Path("recordings")
        recordings_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.filename).suffix.lower() or ".webm"
        saved_filename = f"recording_{timestamp}{file_extension}"
        saved_path = recordings_dir / saved_filename

        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"Processing {saved_path}")
        result = speech_to_text.process_audio_file(str(saved_path))

        if result.get("success"):
            transcript_text = result.get("transcript", "")
            speakers = []
            for line in transcript_text.split("\n"):
                if ":" in line:
                    speaker = line.split(":")[0].strip()
                    if speaker.startswith("SPEAKER_") and speaker not in speakers:
                        speakers.append(speaker)

            result["speakers"] = speakers
            result["speaker_count"] = len(speakers)
            result["file_info"] = {
                "original_name": file.filename,
                "saved_name": saved_filename,
                "size_mb": saved_path.stat().st_size / (1024 * 1024),
            }

        return result

    except Exception as e:
        print(f"Error processing audio: {e}")
        import traceback

        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})
