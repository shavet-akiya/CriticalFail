from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import Optional, Dict
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import os
import warnings
import signal
import sys
import uuid
import threading
import time

warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Check offline mode setting
offline_mode = os.environ.get("HF_HUB_OFFLINE", "not set")
print("\n" + "=" * 80)
print("SPEECH SERVER STARTUP")
print("=" * 80)
print(f"HF_HUB_OFFLINE: {offline_mode}")
if offline_mode == "1":
    print("⚠️  WARNING: Running in offline mode - models must be pre-cached!")
else:
    print("✓ Online mode - will download models if needed")
print("=" * 80 + "\n")

app = FastAPI()

speech_to_text = None
initialization_attempted = False

# Job queue (in-memory storage)
jobs: Dict[str, dict] = {}


def handle_crash(signum, frame):
    """Handle segmentation faults and other crashes"""
    print("\n" + "=" * 80)
    print("💥 CRITICAL: Process received signal", signum)
    print("=" * 80)
    print("This is likely a cuDNN/CUDA library incompatibility.")
    print("The container will now exit.")
    print("=" * 80)
    sys.exit(139)


signal.signal(signal.SIGSEGV, handle_crash)


def initialize_stt():
    """Initialize SpeechToText with lazy loading of models"""
    global speech_to_text, initialization_attempted

    if initialization_attempted:
        status = "already loaded" if speech_to_text and speech_to_text.models_loaded else "failed previously"
        print(f"Initialization already attempted: {status}")
        return speech_to_text is not None

    initialization_attempted = True

    try:
        print("\n🚀 Attempting to initialize Speech Service...")
        from SpeechToText import SpeechToText

        speech_to_text = SpeechToText(save_folder="transcripts")

        if speech_to_text.models_loaded:
            print("\n✅ Speech Service ready with all models!")
        else:
            print("\n⚠️  Speech Service initialized but models NOT loaded")
        return True
    except Exception as e:
        print(f"\n❌ Failed to initialize Speech Service: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_audio_job(job_id: str, audio_path: str):
    """Background thread function to process audio"""
    try:
        print(f"\n[JOB {job_id}] Starting background processing...")
        jobs[job_id]["status"] = "processing"
        jobs[job_id]["progress"] = "Initializing..."
        
        result = speech_to_text.process_audio_file(audio_path)
        
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
            
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["result"] = result
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
            
            print(f"[JOB {job_id}] ✅ Completed successfully!")
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = result.get("error", "Unknown error")
            print(f"[JOB {job_id}] ❌ Failed: {jobs[job_id]['error']}")
            
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        print(f"[JOB {job_id}] ❌ Exception: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("startup")
async def startup_event():
    """Try to initialize on startup"""
    print("\n📡 FastAPI startup event triggered")
    initialize_stt()


@app.get("/")
async def root():
    is_ready = (
        speech_to_text is not None and speech_to_text.models_loaded
        if speech_to_text
        else False
    )
    
    return {
        "message": "Speech Processing Service (Job Queue Mode)",
        "status": "ready" if is_ready else "initializing" if not initialization_attempted else "failed",
        "models_loaded": speech_to_text.models_loaded if speech_to_text else False,
        "active_jobs": len([j for j in jobs.values() if j["status"] == "processing"]),
        "total_jobs": len(jobs),
    }


@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    min_speakers: Optional[int] = Form(2),
    max_speakers: Optional[int] = Form(8),
):
    """Submit audio for processing - returns job_id immediately"""
    print("\n" + "=" * 80)
    print("NEW JOB SUBMISSION")
    print("=" * 80)
    print(f"File: {file.filename}")
    print(f"Content-Type: {file.content_type}")
    
    # Check if service is ready
    if not speech_to_text:
        print("Speech service not initialized, attempting initialization...")
        if not initialize_stt():
            return JSONResponse(
                status_code=503,
                content={"error": "Speech service failed to initialize", "success": False},
            )

    if not speech_to_text.models_loaded:
        return JSONResponse(
            status_code=503,
            content={"error": "Models not loaded", "success": False},
        )

    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Save file
        recordings_dir = Path("recordings")
        recordings_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_extension = Path(file.filename).suffix.lower() or ".webm"
        saved_filename = f"recording_{timestamp}{file_extension}"
        saved_path = recordings_dir / saved_filename
        
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_size_mb = saved_path.stat().st_size / (1024 * 1024)
        
        # Create job entry
        jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": "Queued",
            "filename": file.filename,
            "saved_path": str(saved_path),
            "file_size_mb": file_size_mb,
            "created_at": datetime.now().isoformat(),
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }
        
        print(f"✓ Created job: {job_id}")
        print(f"✓ File saved: {saved_path} ({file_size_mb:.2f} MB)")
        
        # Start processing in background thread
        thread = threading.Thread(target=process_audio_job, args=(job_id, str(saved_path)))
        thread.daemon = True
        thread.start()
        
        print(f"✓ Background processing started for job {job_id}")
        print("=" * 80)
        
        # Return job ID immediately
        return {
            "success": True,
            "job_id": job_id,
            "message": "Job queued for processing",
            "status": "queued",
        }
        
    except Exception as e:
        print(f"❌ Error creating job: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "success": False}
        )


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Check status of a processing job"""
    if job_id not in jobs:
        return JSONResponse(
            status_code=404,
            content={"error": "Job not found", "success": False}
        )
    
    job = jobs[job_id]
    
    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", ""),
        "created_at": job["created_at"],
    }
    
    if job["status"] == "completed":
        response["result"] = job["result"]
        response["completed_at"] = job.get("completed_at")
    elif job["status"] == "failed":
        response["error"] = job.get("error")
    
    return response


@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return {
        "total": len(jobs),
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "filename": job["filename"],
                "created_at": job["created_at"],
            }
            for job_id, job in jobs.items()
        ]
    }