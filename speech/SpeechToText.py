"""
SpeechToText.py - Audio transcription service with speaker diarization
Processes audio files with WhisperX for transcription and speaker identification
"""

import os
import tempfile
import wave
from datetime import datetime
from pathlib import Path
import numpy as np
import whisperx
import torch
import gc
import warnings
import glob

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# Add FFmpeg to PATH if it exists locally
ffmpeg_dirs = glob.glob(os.path.join(os.path.dirname(__file__), "ffmpeg*", "bin"))
if ffmpeg_dirs:
    os.environ["PATH"] = ffmpeg_dirs[0] + os.pathsep + os.environ.get("PATH", "")


class SpeechToText:
    """Main class for audio transcription with speaker diarization"""

    def __init__(self, model_size="base", save_folder="transcripts"):
        """
        Initialize the Speech to Text service

        Args:
            model_size: Size of WhisperX model ("tiny", "base", "small", "medium", "large")
            save_folder: Folder to save transcripts to
        """
        self.model_size = model_size
        self.save_folder = save_folder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        # Models
        self.whisper_model = None
        self.align_model = None
        self.align_metadata = None
        self.speaker_embedder = None
        self.models_loaded = False

        # Create save folder if it doesn't exist
        os.makedirs(save_folder, exist_ok=True)

        # Initialize models on creation
        self.initialize_models()

    def initialize_models(self):
        """Initialize all AI models - call this once at startup"""
        try:
            print("Initializing WhisperX models...")

            # Load WhisperX model
            self.whisper_model = whisperx.load_model(
                self.model_size,
                self.device,
                compute_type=self.compute_type,
                language="en",
                download_root="models/whisperx",
            )

            # Load alignment model
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code="en",
                device=self.device,
                model_dir="models/whisperx_align",
            )

            # Load speaker diarization model
            try:
                from speechbrain.inference.speaker import EncoderClassifier

                self.speaker_embedder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="models/speechbrain_working",
                    run_opts={"device": self.device},
                )
                print("Speaker diarization model loaded successfully")
            except:
                try:
                    from speechbrain.pretrained import EncoderClassifier as OldEncoder

                    self.speaker_embedder = OldEncoder.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        savedir="models/speechbrain_alt",
                        run_opts={"device": self.device},
                    )
                    print("Speaker diarization model loaded (old API)")
                except Exception as e:
                    print(f"Warning: Speaker diarization not available: {e}")
                    self.speaker_embedder = None

            self.models_loaded = True
            print("All models initialized successfully")
            return True

        except Exception as e:
            print(f"Error initializing models: {e}")
            self.models_loaded = False
            return False

    def process_audio_file(self, audio_file_path):
        """
        Process an audio file and return transcript with speaker diarization

        Args:
            audio_file_path: Path to the audio file to process

        Returns:
            dict: Contains 'success', 'transcript', 'file_path', and 'error' keys
        """
        if not self.models_loaded:
            return {
                "success": False,
                "error": "Models not loaded",
                "transcript": "",
                "file_path": "",
            }

        try:
            print(f"Processing audio file: {audio_file_path}")

            # Transcribe with speaker diarization
            transcript = self._transcribe_audio_file(audio_file_path)

            # Save transcript
            file_path = self._save_transcript(transcript)

            print(f"Transcript saved to: {file_path}")

            return {
                "success": True,
                "transcript": transcript,
                "file_path": file_path,
                "error": None,
            }

        except Exception as e:
            print(f"Error processing audio file: {e}")
            return {
                "success": False,
                "error": str(e),
                "transcript": "",
                "file_path": "",
            }

    def _transcribe_audio_file(self, audio_file_path):
        """Internal method to transcribe audio file with speaker diarization"""
        if not self.whisper_model:
            return "Error: Models not loaded"

        try:
            # Transcribe with WhisperX
            result = self.whisper_model.transcribe(
                audio_file_path, batch_size=16, language="en"
            )

            # Align for better timestamps
            if self.align_model and "segments" in result:
                aligned_result = whisperx.align(
                    result["segments"],
                    self.align_model,
                    self.align_metadata,
                    audio_file_path,
                    self.device,
                    return_char_alignments=False,
                )
                if isinstance(aligned_result, dict) and "segments" in aligned_result:
                    result["segments"] = aligned_result["segments"]
                elif isinstance(aligned_result, list):
                    result["segments"] = aligned_result

            # Apply speaker diarization if available
            if self.speaker_embedder and "segments" in result:
                result["segments"] = self._perform_diarization(
                    audio_file_path, result["segments"]
                )
                transcript = self._format_with_speakers(result)
            else:
                transcript = self._format_plain(result)

            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            return transcript

        except Exception as e:
            print(f"Transcription error: {e}")
            return f"Transcription failed: {e}"

    def _perform_diarization(self, audio_path, segments):
        """Internal method to perform speaker diarization"""
        if not self.speaker_embedder:
            return segments

        try:
            import torchaudio
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score

            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

            # Extract embeddings for each segment
            embeddings = []
            valid_segments = []

            for segment in segments:
                start_time = segment.get("start", 0)
                end_time = segment.get("end", start_time + 1)

                start_sample = int(start_time * 16000)
                end_sample = int(end_time * 16000)

                segment_audio = waveform[:, start_sample:end_sample]

                if segment_audio.shape[1] > 480:  # Min 30ms
                    with torch.no_grad():
                        embedding = self.speaker_embedder.encode_batch(segment_audio)
                        embeddings.append(embedding.squeeze().cpu().numpy())
                        valid_segments.append(segment)

            if len(embeddings) < 2:
                for segment in segments:
                    segment["speaker"] = "SPEAKER_00"
                return segments

            embeddings = np.array(embeddings)

            # Find optimal number of speakers
            max_speakers = min(8, len(embeddings) // 2)
            best_n = 2
            best_score = -1

            for n in range(2, max_speakers + 1):
                try:
                    clusterer = AgglomerativeClustering(n_clusters=n)
                    labels = clusterer.fit_predict(embeddings)
                    score = silhouette_score(embeddings, labels)
                    if score > best_score:
                        best_score = score
                        best_n = n
                except:
                    continue

            # Final clustering
            clusterer = AgglomerativeClustering(n_clusters=best_n)
            speaker_labels = clusterer.fit_predict(embeddings)

            # Assign speakers
            for segment, speaker_id in zip(valid_segments, speaker_labels):
                segment["speaker"] = f"SPEAKER_{speaker_id:02d}"

            for segment in segments:
                if "speaker" not in segment:
                    segment["speaker"] = "SPEAKER_00"

            return segments

        except Exception as e:
            print(f"Diarization error: {e}")
            return segments

    def _format_with_speakers(self, result):
        """Format transcript with speaker labels"""
        formatted = []
        current_speaker = None
        speaker_text = []

        for segment in result.get("segments", []):
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()

            if speaker != current_speaker:
                if current_speaker and speaker_text:
                    formatted.append(f"{current_speaker}: {' '.join(speaker_text)}")
                current_speaker = speaker
                speaker_text = []

            if text:
                speaker_text.append(text)

        if current_speaker and speaker_text:
            formatted.append(f"{current_speaker}: {' '.join(speaker_text)}")

        return "\n\n".join(formatted) if formatted else "No speech detected"

    def _format_plain(self, result):
        """Format transcript without speaker labels"""
        text_parts = []
        for segment in result.get("segments", []):
            text = segment.get("text", "").strip()
            if text:
                text_parts.append(text)

        return " ".join(text_parts) if text_parts else "No speech detected"

    def _save_transcript(self, transcript):
        """Save transcript to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcript_{timestamp}.txt"
        file_path = os.path.join(self.save_folder, filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(
                f"Transcript Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write("=" * 50 + "\n\n")
            f.write(transcript)

        return file_path

    def cleanup(self):
        """Clean up resources - call this when done"""
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        print("Cleanup complete")
