import httpx
from typing import Dict, Optional
import asyncio
from pathlib import Path


class SpeechClient:
    def __init__(self, speech_service_url: str = "http://speech:8001"):
        self.speech_service_url = speech_service_url
        self.initialization_complete = False
        self.initialization_error = None
        self.is_processing = False
        self.is_recording = False

    async def initialize(self):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.speech_service_url}/")
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "ready":
                        self.initialization_complete = True
                        print("Speech client connected to speech service")
                        return {"status": "ready"}
            raise Exception("Speech service not ready")
        except Exception as e:
            self.initialization_error = str(e)
            print(f"Speech client initialization error: {e}")
            raise

    def start_recording(self):
        return {
            "success": False,
            "error": "Recording not supported - use browser recording",
        }

    def pause_recording(self):
        return {
            "success": False,
            "error": "Recording not supported - use browser recording",
        }

    def resume_recording(self):
        return {
            "success": False,
            "error": "Recording not supported - use browser recording",
        }

    def stop_recording(self):
        return {
            "success": False,
            "error": "Recording not supported - use browser recording",
        }

    async def process_audio_file(
        self, file_path: str, min_speakers: int = 2, max_speakers: int = 8
    ) -> Dict:
        if self.is_processing:
            return {"success": False, "error": "Already processing"}

        try:
            self.is_processing = True

            async with httpx.AsyncClient(timeout=300.0) as client:
                with open(file_path, "rb") as f:
                    files = {"file": (Path(file_path).name, f, "audio/webm")}
                    data = {
                        "min_speakers": str(min_speakers),
                        "max_speakers": str(max_speakers),
                    }

                    response = await client.post(
                        f"{self.speech_service_url}/process", files=files, data=data
                    )

                if response.status_code == 200:
                    return response.json()
                else:
                    return {
                        "success": False,
                        "error": f"Speech service error: {response.text}",
                    }

        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            self.is_processing = False

    def get_status(self) -> Dict:
        return {
            "initialized": self.initialization_complete,
            "error": self.initialization_error,
            "is_processing": self.is_processing,
            "is_recording": self.is_recording,
        }


speech_service = SpeechClient()
