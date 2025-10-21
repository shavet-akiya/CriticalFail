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
import subprocess
import shutil

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Add FFmpeg to PATH if it exists locally
ffmpeg_dirs = glob.glob(os.path.join(os.path.dirname(__file__), "ffmpeg*", "bin"))
if ffmpeg_dirs:
    os.environ["PATH"] = ffmpeg_dirs[0] + os.pathsep + os.environ.get("PATH", "")


class SpeechToText:
    """Main class for audio transcription with speaker diarization"""

    def __init__(self, model_size="large", save_folder="transcripts"):
        """
        Initialize the Speech to Text service

        Args:
            model_size: Size of WhisperX model ("tiny", "base", "small", "medium", "large")
            save_folder: Folder to save transcripts to
        """
        print("=" * 80)
        print("INITIALIZING SPEECH-TO-TEXT SERVICE")
        print("=" * 80)
        
        self.model_size = model_size
        self.save_folder = save_folder
<<<<<<< Updated upstream
        
        # Force CPU - we have CPU-only PyTorch installed
=======


        # GPU Setup with detailed logging
>>>>>>> Stashed changes
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"
            print(f"\n🚀 GPU DETECTED AND ENABLED!")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   PyTorch Version: {torch.__version__}\n")
        else:
            self.compute_type = "int8"
            print("\n⚠️  No GPU detected - using CPU (will be slower)")
            print("   Make sure:")
            print("   1. NVIDIA drivers are installed on host")
            print("   2. nvidia-container-toolkit is installed")
            print("   3. Docker compose has GPU configuration\n")

        print(f"✓ Model size: {model_size}")
        print(f"✓ Device: {self.device}")
<<<<<<< Updated upstream
        if self.device == "cuda":
            print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("⚠️  No NVIDIA GPU detected - using CPU (slower)")
=======
>>>>>>> Stashed changes
        print(f"✓ Compute type: {self.compute_type}")
        print(f"✓ Save folder: {save_folder}")

        # Models
        self.whisper_model = None
        self.align_model = None
        self.align_metadata = None
        self.speaker_embedder = None
        self.models_loaded = False

        # Create save folder if it doesn't exist
        os.makedirs(save_folder, exist_ok=True)
        print(f"✓ Created/verified save folder: {save_folder}")

        # Initialize models on creation
        print("\nStarting model initialization...")
        self.initialize_models()

    def initialize_models(self):
        """Initialize all AI models - call this once at startup"""
        print("\n" + "=" * 80)
        print("LOADING AI MODELS")
        print("=" * 80)
        
        try:
            # Check HF_HUB_OFFLINE setting
            offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
            print(f"HF_HUB_OFFLINE environment variable: {offline_mode}")
            
            if offline_mode == "1":
                print("⚠️  WARNING: HF_HUB_OFFLINE=1 - Models must be pre-cached!")
            else:
                print("✓ Online mode enabled - will download models if needed")

            # Load WhisperX model
            print("\n[1/3] Loading WhisperX transcription model...")
            print(f"      Model: {self.model_size}")
            print(f"      Download root: models/whisperx")
            
            self.whisper_model = whisperx.load_model(
                self.model_size,
                self.device,
                compute_type=self.compute_type,
                language="en",
                download_root="models/whisperx",
            )
            print("✅ WhisperX model loaded successfully!")

            # Load alignment model
            print("\n[2/3] Loading alignment model...")
            print(f"      Language: en")
            print(f"      Model dir: models/whisperx_align")
            
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code="en",
                device=self.device,
                model_dir="models/whisperx_align",
            )
            print("✅ Alignment model loaded successfully!")

            # Load speaker diarization model
            print("\n[3/3] Loading speaker diarization model...")
            print(f"      Model: speechbrain/spkrec-ecapa-voxceleb")
            
            try:
                from speechbrain.inference.speaker import EncoderClassifier

                self.speaker_embedder = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="models/speechbrain_working",
                    run_opts={"device": self.device},
                )
                print("✅ Speaker diarization model loaded successfully (new API)")
            except Exception as e1:
                print(f"⚠️  New API failed: {e1}")
                print("   Trying alternative API...")
                try:
                    from speechbrain.pretrained import EncoderClassifier as OldEncoder

                    self.speaker_embedder = OldEncoder.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        savedir="models/speechbrain_alt",
                        run_opts={"device": self.device},
                    )
                    print("✅ Speaker diarization model loaded successfully (old API)")
                except Exception as e2:
                    print(f"❌ Both APIs failed!")
                    print(f"   Error 1: {e1}")
                    print(f"   Error 2: {e2}")
                    print("⚠️  Continuing without speaker diarization")
                    self.speaker_embedder = None

            self.models_loaded = True
            print("\n" + "=" * 80)
            print("🎉 ALL MODELS LOADED SUCCESSFULLY!")
            print("=" * 80)
            return True

        except Exception as e:
            print("\n" + "=" * 80)
            print("❌ FATAL ERROR DURING MODEL INITIALIZATION")
            print("=" * 80)
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
            
            self.models_loaded = False
            print("\n⚠️  Service will start but models are NOT loaded")
            print("=" * 80)
            return False

    def process_audio_file(self, audio_file_path):
        """
        Process an audio file and return transcript with speaker diarization

        Args:
            audio_file_path: Path to the audio file to process

        Returns:
            dict: Contains 'success', 'transcript', 'file_path', and 'error' keys
        """
        print("\n" + "=" * 80)
        print("PROCESSING AUDIO FILE")
        print("=" * 80)
        print(f"File: {audio_file_path}")
        
        if not self.models_loaded:
            error_msg = "Models not loaded - cannot process audio"
            print(f"❌ {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "transcript": "",
                "file_path": "",
            }

        try:
            print("✓ Models are loaded, starting transcription...")

            # Transcribe with speaker diarization
            print("\nStep 1: Transcribing audio...")
            transcript = self._transcribe_audio_file(audio_file_path)
            print("✓ Transcription complete!")

            # Save transcript
            print("\nStep 2: Saving transcript...")
            file_path = self._save_transcript(transcript)
            print(f"✓ Transcript saved to: {file_path}")

            print("\n✅ PROCESSING COMPLETE!")
            print("=" * 80)

            return {
                "success": True,
                "transcript": transcript,
                "file_path": file_path,
                "error": None,
            }

        except Exception as e:
            error_msg = f"Error processing audio: {str(e)}"
            print(f"\n❌ {error_msg}")
            
            import traceback
            print("\nFull traceback:")
            traceback.print_exc()
            print("=" * 80)
            
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
            import subprocess
            
            # Try to get audio duration (may fail for some formats like browser webm)
            try:
                result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
                    '-of', 'default=noprint_wrappers=1:nokey=1', audio_file_path],
                    capture_output=True, text=True, timeout=10
                )
                duration_str = result.stdout.strip()
                
                # Handle 'N/A' or empty responses
                if duration_str and duration_str != 'N/A':
                    duration = float(duration_str)
                    print(f"  → Audio duration: {duration/60:.1f} minutes")
                else:
                    # Duration unavailable (common for browser recordings)
                    print(f"  → Audio duration: Unknown (will process without chunking)")
                    duration = 0  # Process normally without chunking
            except (ValueError, subprocess.TimeoutExpired) as e:
                print(f"  → Could not determine duration: {e}")
                print(f"  → Will process without chunking")
                duration = 0
            
            # If file is longer than 30 minutes, process in chunks
            if duration > 1800:  # 30 minutes
                print(f"  → Large file detected, processing in 15-minute chunks...")
                return self._transcribe_in_chunks(audio_file_path, duration)
            
            # Normal processing for shorter files or unknown duration
            print("  → Running WhisperX transcription...")
            result = self.whisper_model.transcribe(
                audio_file_path, 
                batch_size=8,
                language="en"
            )
            print(f"  → Found {len(result.get('segments', []))} segments")

            # Align for better timestamps
            if self.align_model and "segments" in result:
                print("  → Aligning timestamps...")
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
                print("  → Alignment complete")

            # Apply speaker diarization if available
            if self.speaker_embedder and "segments" in result:
                print("  → Performing speaker diarization...")
                result["segments"] = self._perform_diarization(
                    audio_file_path, result["segments"]
                )
                transcript = self._format_with_speakers(result)
                print("  → Diarization complete")
            else:
                print("  → No speaker diarization (embedder not available)")
                transcript = self._format_plain(result)

            if self.device == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

            return transcript

        except Exception as e:
            print(f"  ❌ Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return f"Transcription failed: {e}"

    def _transcribe_in_chunks(self, audio_file_path, total_duration):
        """Process large audio files in chunks to save memory"""
        import subprocess
        import os
        
        chunk_duration = 900  # 15 minutes per chunk
        all_segments = []
        
        # Use absolute path for chunk directory
        chunk_dir = os.path.join(os.path.dirname(audio_file_path), "temp_chunks")
        
        try:
            # Create chunk directory
            os.makedirs(chunk_dir, exist_ok=True)
            print(f"  → Created chunk directory: {chunk_dir}")
            
            num_chunks = int(total_duration / chunk_duration) + 1
            print(f"  → Processing {num_chunks} chunks...")
            
            for i in range(num_chunks):
                start_time = i * chunk_duration
                chunk_file = os.path.join(chunk_dir, f"chunk_{i}.wav")
                
                # Extract chunk using ffmpeg
                print(f"  → Chunk {i+1}/{num_chunks}: {start_time/60:.1f}-{(start_time+chunk_duration)/60:.1f} min")
                
                ffmpeg_result = subprocess.run([
                    'ffmpeg', '-i', audio_file_path,
                    '-ss', str(start_time),
                    '-t', str(chunk_duration),
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    '-y',
                    chunk_file
                ], capture_output=True, text=True)
                
                if ffmpeg_result.returncode != 0:
                    print(f"  ❌ FFmpeg error for chunk {i}:")
                    print(ffmpeg_result.stderr)
                    raise Exception(f"Failed to create chunk {i}")
                
                if not os.path.exists(chunk_file):
                    raise Exception(f"Chunk file not created: {chunk_file}")
                
                print(f"     ✓ Chunk saved: {os.path.getsize(chunk_file) / 1024 / 1024:.2f} MB")
                
                # Transcribe chunk
                result = self.whisper_model.transcribe(
                    chunk_file,
                    batch_size=4,  # Smaller batch for chunks
                    language="en"
                )
                
                # Adjust timestamps
                for seg in result.get('segments', []):
                    seg['start'] += start_time
                    seg['end'] += start_time
                
                all_segments.extend(result.get('segments', []))
                
                # Clean up chunk file immediately
                try:
                    os.remove(chunk_file)
                except:
                    pass
                
                # Force garbage collection
                gc.collect()
            
            print(f"  → All chunks processed, total segments: {len(all_segments)}")
            
            # Combine results
            combined_result = {'segments': all_segments}
            
            # Apply speaker diarization to combined segments
            if self.speaker_embedder:
                print("  → Performing speaker diarization on full audio...")
                combined_result["segments"] = self._perform_diarization(
                    audio_file_path, combined_result["segments"]
                )
                transcript = self._format_with_speakers(combined_result)
            else:
                transcript = self._format_plain(combined_result)
            
            return transcript
            
        except Exception as e:
            print(f"  ❌ Chunking error: {e}")
            import traceback
            traceback.print_exc()
            raise
            
        finally:
            # Clean up temp directory
            import shutil
            if os.path.exists(chunk_dir):
                try:
                    shutil.rmtree(chunk_dir)
                    print(f"  → Cleaned up chunk directory")
                except Exception as e:
                    print(f"  ⚠️  Could not clean up chunk directory: {e}")

    def _perform_diarization(self, audio_path, segments):
        """Internal method to perform speaker diarization"""
        if not self.speaker_embedder:
            print("    ⚠️  No speaker embedder available")
            return segments

        try:
            import torchaudio
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import silhouette_score
            import subprocess
            import os

            # Convert to WAV if needed (m4a, mp3, etc.)
            audio_ext = os.path.splitext(audio_path)[1].lower()
            if audio_ext not in ['.wav']:
                print(f"    → Converting {audio_ext} to WAV for diarization...")
                wav_path = audio_path.rsplit('.', 1)[0] + '_temp.wav'
                
                subprocess.run([
                    'ffmpeg', '-i', audio_path,
                    '-ar', '16000',
                    '-ac', '1',
                    '-y',
                    wav_path
                ], capture_output=True, check=True)
                
                audio_path_for_diarization = wav_path
                print(f"    ✓ Converted to WAV: {wav_path}")
            else:
                audio_path_for_diarization = audio_path

            print("    → Loading audio for diarization...")
            waveform, sample_rate = torchaudio.load(audio_path_for_diarization)
            if sample_rate != 16000:
                waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

            print("    → Extracting speaker embeddings...")
            print(f"       Total segments to process: {len(segments)}")
            
            embeddings = []
            valid_segments = []

            for i, segment in enumerate(segments):
                start_time = segment.get("start", 0)
                end_time = segment.get("end", start_time + 1)

                start_sample = int(start_time * 16000)
                end_sample = int(end_time * 16000)

                segment_audio = waveform[:, start_sample:end_sample]

                if segment_audio.shape[1] > 480:  # Min 30ms
                    with torch.no_grad():
                        embedding = self.speaker_embedder.encode_batch(segment_audio)
                        
                        # Ensure 1D array for clustering
                        embedding_np = embedding.squeeze().cpu().numpy()
                        
                        if i == 0:
                            print(f"       First embedding shape: {embedding_np.shape}")
                        
                        embedding_flat = embedding_np.ravel()
                        
                        embeddings.append(embedding_flat)
                        valid_segments.append(segment)

            print(f"       Valid segments with embeddings: {len(embeddings)}")

            # Clean up temp WAV file if created
            if audio_ext in ['.m4a', '.mp4', '.aac']:
                try:
                    os.remove(audio_path_for_diarization)
                    print(f"    ✓ Cleaned up temp file")
                except:
                    pass

            if len(embeddings) < 2:
                print("    ⚠️  Not enough segments for clustering, assigning all to SPEAKER_00")
                for segment in segments:
                    segment["speaker"] = "SPEAKER_00"
                return segments

            embeddings = np.array(embeddings)
            print(f"       Embeddings array shape: {embeddings.shape}")

            print("    → Clustering speakers...")
<<<<<<< Updated upstream
            
            # Use min/max speakers from settings
            min_speakers = 2
            max_speakers = 8
            
            max_clusters = min(max_speakers, len(embeddings) // 2)
            min_clusters = min(min_speakers, max_clusters)
            
            print(f"       Testing {min_clusters} to {max_clusters} clusters...")
            
            best_n = min_clusters
=======

            # Automatically detect optimal number of speakers (2-15 range)
            max_possible_clusters = min(15, len(embeddings) // 3)  # At least 3 segments per speaker
            print(f"       Testing 2 to {max_possible_clusters} speakers...")

            best_n = 2
>>>>>>> Stashed changes
            best_score = -1
            scores = []

            for n in range(2, max_possible_clusters + 1):
                try:
                    clusterer = AgglomerativeClustering(n_clusters=n)
                    labels = clusterer.fit_predict(embeddings)
                    
<<<<<<< Updated upstream
                    if n > 1:
                        score = silhouette_score(embeddings, labels)
                        if score > best_score:
                            best_score = score
                            best_n = n
=======
                    score = silhouette_score(embeddings, labels)
                    scores.append((n, score))
                    
                    # Look for "elbow" - when adding more speakers stops improving significantly
                    if len(scores) >= 3:
                        # Check if improvement is plateauing
                        recent_improvements = [scores[i][1] - scores[i-1][1] for i in range(-2, 0)]
                        avg_improvement = sum(recent_improvements) / len(recent_improvements)
                        
                        # If improvement drops below 5% of max score, we've found enough speakers
                        if avg_improvement < 0.05 * max(s[1] for s in scores):
                            best_n = n - 1  # Use previous n (before plateau)
                            best_score = scores[-2][1]
                            print(f"       Detected plateau at {n} speakers, using {best_n}")
                            break
                    
                    if score > best_score:
                        best_score = score
                        best_n = n
                        
>>>>>>> Stashed changes
                except Exception as e:
                    print(f"       Error with {n} speakers: {e}")
                    break

<<<<<<< Updated upstream
            print(f"    ✓ Selected {best_n} speakers (score: {best_score:.3f})")
            
=======
            print(f"    ✓ Auto-detected {best_n} speakers (silhouette score: {best_score:.3f})")
            print(f"       Tested configurations: {[(n, f'{s:.3f}') for n, s in scores[:best_n]]}")

>>>>>>> Stashed changes
            clusterer = AgglomerativeClustering(n_clusters=best_n)
            speaker_labels = clusterer.fit_predict(embeddings)

            print(f"       Assigning speakers to {len(valid_segments)} segments...")
            for segment, speaker_id in zip(valid_segments, speaker_labels):
                segment["speaker"] = f"SPEAKER_{speaker_id:02d}"

            # Assign default speaker to short segments
            for segment in segments:
                if "speaker" not in segment:
                    segment["speaker"] = "SPEAKER_00"

            print(f"    ✓ Diarization complete: {best_n} speakers identified")
            return segments

        except Exception as e:
            print(f"    ❌ Diarization error: {e}")
            import traceback
            traceback.print_exc()
            
            # On error, assign all to single speaker
            for segment in segments:
                segment["speaker"] = "SPEAKER_00"
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