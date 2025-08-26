#!/usr/bin/env python3
"""
Dungeon Scribe - D&D Session Transcriber with Enhanced Speaker Detection
Records audio, transcribes with speaker identification, exports to text file
"""

import tkinter as tk
from tkinter import messagebox, filedialog, scrolledtext
import threading
import queue
import time
import pyaudio
import whisper
import numpy as np
from datetime import datetime
import os
from collections import defaultdict

# Optional speaker diarization - kept for backward compatibility
DIARIZATION_AVAILABLE = False  # Will be set by RealSpeakerDetection


class LoadingScreen:
    """Loading screen that appears immediately on startup"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Dungeon Scribe - Loading")
        self.root.geometry("500x300")
        self.root.configure(bg="#2c1810")
        self.root.resizable(False, False)

        # Center the window manually
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

        self.create_loading_widgets()
        self.progress_dots = 0

        # Show window immediately
        self.root.update()

    def create_loading_widgets(self):
        """Create loading screen widgets"""
        # Title
        title_label = tk.Label(
            self.root,
            text="Dungeon Scribe",
            font=("Georgia", 20, "bold"),
            fg="#daa520",
            bg="#2c1810",
        )
        title_label.pack(pady=30)

        # Loading message
        self.status_label = tk.Label(
            self.root,
            text="Initializing Application...",
            font=("Georgia", 12),
            fg="#daa520",
            bg="#2c1810",
        )
        self.status_label.pack(pady=20)

        # Animated dots
        self.dots_label = tk.Label(
            self.root,
            text="Loading...",
            font=("Georgia", 16, "bold"),
            fg="#daa520",
            bg="#2c1810",
        )
        self.dots_label.pack(pady=10)

        # Progress info
        self.progress_label = tk.Label(
            self.root,
            text="Loading dependencies...",
            font=("Georgia", 10),
            fg="#8b7355",
            bg="#2c1810",
        )
        self.progress_label.pack(pady=20)

    def update_status(self, message, progress_text=""):
        """Update loading status"""
        self.status_label.config(text=message)
        if progress_text:
            self.progress_label.config(text=progress_text)
        self.root.update()

    def close(self):
        """Close loading screen"""
        self.root.destroy()


class AudioRecorder:
    """Handles audio recording functionality with live transcription support"""

    def __init__(self):
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000  # Whisper works best at 16kHz
        self.recording = False
        self.paused = False
        self.frames = []
        self.audio = None
        self.stream = None
        self.live_transcription_callback = None
        self.live_buffer = []
        self.live_buffer_size = self.rate * 10  # 10 seconds for live transcription

    def set_live_transcription_callback(self, callback):
        """Set callback function for live transcription updates"""
        self.live_transcription_callback = callback

    def test_microphone(self):
        """Test if microphone is accessible"""
        try:
            if self.audio is None:
                self.audio = pyaudio.PyAudio()

            # List available input devices
            print("Available audio input devices:")
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info["maxInputChannels"] > 0:
                    print(
                        f"  Device {i}: {info['name']} (Channels: {info['maxInputChannels']})"
                    )

            # Get default input device
            default_device = self.audio.get_default_input_device_info()
            print(f"Default input device: {default_device['name']}")

            # Test recording a small sample
            test_stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk,
            )

            # Record for 0.1 seconds to test
            test_data = test_stream.read(
                int(self.rate * 0.1), exception_on_overflow=False
            )
            test_stream.close()

            # Check if we got any data
            if len(test_data) > 0:
                audio_array = np.frombuffer(test_data, dtype=np.int16)
                max_amplitude = np.max(np.abs(audio_array))
                print(f"Microphone test successful. Max amplitude: {max_amplitude}")
                return True, default_device["name"]
            else:
                return False, "No audio data received"

        except Exception as e:
            print(f"Microphone test failed: {e}")
            return False, str(e)

    def start_recording(self):
        """Start audio recording"""
        self.recording = True
        self.paused = False
        if not self.frames:  # Only clear if starting fresh
            self.frames = []

        if self.audio is None:
            self.audio = pyaudio.PyAudio()

        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

    def pause_recording(self):
        """Pause recording without stopping the stream"""
        self.paused = True

    def resume_recording(self):
        """Resume recording"""
        self.paused = False

    def stop_recording(self):
        """Stop audio recording and return audio data"""
        self.recording = False
        self.paused = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

        # Convert recorded frames to numpy array
        if not self.frames:
            return np.array([])

        audio_data = b"".join(self.frames)
        audio_np = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )
        return audio_np

    def record_chunk(self):
        """Record a single chunk of audio with live transcription support"""
        if self.stream and self.recording:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)

                if not self.paused:
                    self.frames.append(data)

                    # Add to live buffer for transcription
                    audio_chunk = (
                        np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                    )
                    self.live_buffer.extend(audio_chunk)

                    # If live buffer is full, trigger live transcription
                    if len(self.live_buffer) >= self.live_buffer_size:
                        if self.live_transcription_callback:
                            live_audio = np.array(
                                self.live_buffer[-self.live_buffer_size :]
                            )
                            self.live_transcription_callback(live_audio)
                        # Keep only the last 5 seconds for overlap
                        self.live_buffer = self.live_buffer[-int(self.rate * 5) :]

                return True
            except Exception as e:
                print(f"Record chunk error: {e}")
                return False
        return False

    def cleanup(self):
        """Clean up audio resources"""
        if self.stream:
            try:
                self.stream.close()
            except:
                pass
            self.stream = None
        if self.audio:
            try:
                self.audio.terminate()
            except:
                pass
            self.audio = None


class RealSpeakerDetection:
    """Real speaker detection using actual voice analysis libraries"""

    def __init__(self):
        self.enabled = False
        self.method = None
        self.temp_dir = None

    def initialize(self):
        """Try to initialize available speaker diarization methods"""
        import tempfile

        self.temp_dir = tempfile.gettempdir()

        # Try pyannote.audio first (most accurate)
        try:
            from pyannote.audio import Pipeline
            import torch

            # Check if user has authentication token
            hf_token = os.environ.get("HUGGING_FACE_TOKEN")
            if hf_token:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
                )
                self.method = "pyannote"
                self.enabled = True
                print("Speaker detection: pyannote.audio initialized successfully")
                return True
            else:
                print(
                    "pyannote.audio found but needs Hugging Face token. Set HUGGING_FACE_TOKEN environment variable."
                )
        except ImportError:
            print("pyannote.audio not installed")
        except Exception as e:
            print(f"pyannote.audio initialization failed: {e}")

        # Try speechbrain (newer, works with Python 3.13)
        try:
            from speechbrain.pretrained import SpeakerRecognition

            self.speaker_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb"
            )
            self.method = "speechbrain"
            self.enabled = True
            print("Speaker detection: speechbrain initialized successfully")
            return True
        except ImportError:
            print("speechbrain not installed")
        except Exception as e:
            print(f"speechbrain initialization failed: {e}")

        # Try resemblyzer (simpler, no auth needed)
        try:
            from resemblyzer import VoiceEncoder, preprocess_wav
            from scipy.cluster.hierarchy import linkage, fcluster

            self.voice_encoder = VoiceEncoder()
            self.method = "resemblyzer"
            self.enabled = True
            print("Speaker detection: resemblyzer initialized successfully")
            return True
        except ImportError:
            print("resemblyzer not installed")
        except Exception as e:
            print(f"resemblyzer initialization failed: {e}")

        # Try simple-diarizer (wrapper around other tools)
        try:
            from simple_diarizer.diarizer import Diarizer

            self.diarizer = Diarizer(embed_model="xvec")
            self.method = "simple-diarizer"
            self.enabled = True
            print("Speaker detection: simple-diarizer initialized successfully")
            return True
        except ImportError:
            print("simple-diarizer not installed")

        print("No speaker diarization library available. Install one of:")
        print("  pip install pyannote.audio (needs HuggingFace token)")
        print("  pip install speechbrain")
        print("  pip install resemblyzer")
        print("  pip install simple-diarizer")
        return False

    def process_audio(self, audio_data, sample_rate=16000):
        """Process audio and return speaker segments"""
        if not self.enabled:
            return None

        import tempfile
        import wave

        # Save audio to temporary WAV file (required by most libraries)
        with tempfile.NamedTemporaryFile(
            suffix=".wav", delete=False, dir=self.temp_dir
        ) as tmp_file:
            temp_path = tmp_file.name

            # Convert float32 audio to int16 for WAV
            audio_int16 = (audio_data * 32767).astype(np.int16)

            # Write WAV file
            with wave.open(temp_path, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())

        try:
            if self.method == "pyannote":
                return self._process_with_pyannote(temp_path)
            elif self.method == "speechbrain":
                return self._process_with_speechbrain(audio_data, sample_rate)
            elif self.method == "resemblyzer":
                return self._process_with_resemblyzer(audio_data, sample_rate)
            elif self.method == "simple-diarizer":
                return self._process_with_simple_diarizer(temp_path)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return None

    def _process_with_pyannote(self, audio_path):
        """Process with pyannote.audio"""
        diarization = self.pipeline(audio_path)
        segments = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                {
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": f"Speaker {speaker[-1]}",  # Use last char as speaker ID
                }
            )

        return segments

    def _process_with_resemblyzer(self, audio_data, sample_rate):
        """Process with resemblyzer (voice embeddings)"""
        from resemblyzer import VoiceEncoder, preprocess_wav
        from scipy.cluster.hierarchy import linkage, fcluster
        import scipy.signal

        # Preprocess the entire audio first
        preprocessed_audio = preprocess_wav(audio_data)

        # Process audio in windows
        window_size = int(sample_rate * 1.5)  # 1.5-second windows
        hop_size = int(sample_rate * 0.5)  # 0.5-second hop

        embeddings = []
        times = []

        for i in range(0, len(preprocessed_audio) - window_size, hop_size):
            window = preprocessed_audio[i : i + window_size]
            if len(window) > 0:
                # Get embedding for this window
                embedding = self.voice_encoder.embed_utterance(window)
                embeddings.append(embedding)
                times.append(i / sample_rate)

        if len(embeddings) < 2:
            # Not enough data for clustering
            return [
                {
                    "start": 0,
                    "end": len(audio_data) / sample_rate,
                    "speaker": "Speaker A",
                }
            ]

        # Cluster embeddings to identify speakers
        embeddings = np.array(embeddings)

        # Use hierarchical clustering
        linkage_matrix = linkage(embeddings, method="ward", metric="euclidean")

        # Determine number of clusters (max 4 speakers)
        max_speakers = min(4, len(embeddings) // 2)
        clusters = fcluster(linkage_matrix, max_speakers, criterion="maxclust")

        # Convert to segments with smoothing
        segments = []
        current_speaker = None
        segment_start = 0

        for i, (time, speaker_id) in enumerate(zip(times, clusters)):
            speaker = f"Speaker {chr(64 + int(speaker_id))}"

            if speaker != current_speaker:
                if current_speaker is not None and i > 0:
                    segments.append(
                        {
                            "start": segment_start,
                            "end": time,
                            "speaker": current_speaker,
                        }
                    )
                segment_start = time
                current_speaker = speaker

        # Add final segment
        if current_speaker is not None:
            segments.append(
                {
                    "start": segment_start,
                    "end": len(audio_data) / sample_rate,
                    "speaker": current_speaker,
                }
            )

        # Merge very short segments (less than 0.5 seconds)
        merged_segments = []
        for segment in segments:
            duration = segment["end"] - segment["start"]
            if duration < 0.5 and merged_segments:
                # Merge with previous segment
                merged_segments[-1]["end"] = segment["end"]
            else:
                merged_segments.append(segment)

        return (
            merged_segments
            if merged_segments
            else [
                {
                    "start": 0,
                    "end": len(audio_data) / sample_rate,
                    "speaker": "Speaker A",
                }
            ]
        )

    def _process_with_speechbrain(self, audio_data, sample_rate):
        """Process with speechbrain"""
        import torch
        from sklearn.cluster import AgglomerativeClustering
        import numpy as np

        # Process audio in windows
        window_size = int(sample_rate * 2)  # 2-second windows
        hop_size = int(sample_rate * 1)  # 1-second hop

        embeddings = []
        times = []

        for i in range(0, len(audio_data) - window_size, hop_size):
            window = audio_data[i : i + window_size]
            if np.max(np.abs(window)) > 0.01:  # Skip silent windows
                # Convert to tensor
                window_tensor = torch.tensor(window).unsqueeze(0)
                # Get embedding
                embedding = self.speaker_model.encode_batch(window_tensor)
                embeddings.append(embedding.squeeze().numpy())
                times.append(i / sample_rate)

        if len(embeddings) < 2:
            return [
                {
                    "start": 0,
                    "end": len(audio_data) / sample_rate,
                    "speaker": "Speaker A",
                }
            ]

        # Cluster embeddings - limit to reasonable number of speakers
        embeddings = np.array(embeddings)

        # Estimate number of speakers (max 6 for D&D game)
        n_speakers = min(6, max(2, len(embeddings) // 10))

        clustering = AgglomerativeClustering(n_clusters=n_speakers, linkage="ward")
        clusters = clustering.fit_predict(embeddings)

        # Smooth the clusters to reduce rapid switching
        smoothed_clusters = self._smooth_clusters(clusters, times)

        # Convert to segments
        segments = []
        current_speaker = None
        segment_start = 0

        for i, (time, speaker_id) in enumerate(zip(times, smoothed_clusters)):
            # Ensure speaker_id is within valid range
            speaker_id = int(speaker_id) % 26  # Limit to A-Z
            speaker = f"Speaker {chr(65 + speaker_id)}"

            if speaker != current_speaker:
                if current_speaker is not None:
                    segments.append(
                        {
                            "start": segment_start,
                            "end": time,
                            "speaker": current_speaker,
                        }
                    )
                segment_start = time
                current_speaker = speaker

        # Add final segment
        if current_speaker is not None:
            segments.append(
                {
                    "start": segment_start,
                    "end": len(audio_data) / sample_rate,
                    "speaker": current_speaker,
                }
            )

        # Merge very short segments
        merged_segments = []
        for segment in segments:
            duration = segment["end"] - segment["start"]
            if duration < 1.0 and merged_segments:  # Less than 1 second
                # Check if should merge with previous
                if merged_segments[-1]["speaker"] == segment["speaker"]:
                    merged_segments[-1]["end"] = segment["end"]
                elif duration < 0.5:  # Very short, just extend previous
                    merged_segments[-1]["end"] = segment["end"]
                else:
                    merged_segments.append(segment)
            else:
                merged_segments.append(segment)

        return merged_segments

    def _smooth_clusters(self, clusters, times):
        """Smooth cluster assignments to reduce rapid speaker switching"""
        smoothed = clusters.copy()
        window = 3  # Look at 3 segments at a time

        for i in range(window, len(clusters) - window):
            # Get surrounding clusters
            surrounding = clusters[i - window : i + window + 1]
            # Use mode (most common) in the window
            unique, counts = np.unique(surrounding, return_counts=True)
            mode = unique[np.argmax(counts)]

            # If current is different from mode and appears rarely, smooth it
            if clusters[i] != mode and np.sum(surrounding == clusters[i]) <= 2:
                smoothed[i] = mode

        return smoothed

    def _process_with_simple_diarizer(self, audio_path):
        """Process with simple-diarizer"""
        segments = self.diarizer.diarize(audio_path)

        # Convert to our format
        formatted_segments = []
        for start, end, speaker in segments:
            formatted_segments.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": f"Speaker {chr(65 + int(speaker))}",
                }
            )

        return formatted_segments


class TranscriptionEngine:
    """Handles speech-to-text using Whisper with real speaker detection"""

    def __init__(self):
        self.model = None
        self.model_size = "base"  # Can be: tiny, base, small, medium, large
        self.speaker_detector = None

    def initialize(self):
        """Load Whisper model and speaker detection"""
        try:
            print(f"Loading Whisper {self.model_size} model...")
            self.model = whisper.load_model(self.model_size)

            # Initialize real speaker detection
            self.speaker_detector = RealSpeakerDetection()
            if self.speaker_detector.initialize():
                print("Real speaker detection initialized successfully")
            else:
                print(
                    "Speaker detection unavailable - transcription will not identify speakers"
                )
                self.speaker_detector = None

            return True
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            return False

    def transcribe_audio(self, audio_data, use_speaker_detection=True):
        """Transcribe audio with real speaker detection"""
        if not self.model:
            return "Error: Whisper model not loaded"

        if len(audio_data) == 0:
            return "Error: No audio data to transcribe"

        try:
            # Ensure audio is properly formatted
            if len(audio_data.shape) > 1:
                audio_data = audio_data.flatten()

            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))

            print("Starting Whisper transcription...")

            # Get speaker segments if detection is available
            speaker_segments = None
            if (
                use_speaker_detection
                and self.speaker_detector
                and self.speaker_detector.enabled
            ):
                print(
                    f"Processing speaker detection with {self.speaker_detector.method}..."
                )
                speaker_segments = self.speaker_detector.process_audio(
                    audio_data, sample_rate=16000
                )
                if speaker_segments:
                    print(
                        f"Identified {len(set(s['speaker'] for s in speaker_segments))} unique speakers"
                    )

            # Transcribe with Whisper
            result = self.model.transcribe(
                audio_data, language="en", task="transcribe", verbose=False
            )

            # Get base transcript
            transcript_text = result.get("text", "").strip()
            if not transcript_text:
                return "No speech detected in audio"

            # Combine with real speaker detection if available
            if speaker_segments and "segments" in result:
                return self._combine_transcription_with_speakers(
                    result, speaker_segments
                )
            else:
                # Return plain transcript if no speaker detection
                return transcript_text

        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback

            traceback.print_exc()
            return f"Transcription failed: {e}"

    def _combine_transcription_with_speakers(self, whisper_result, speaker_segments):
        """Combine Whisper transcription with real speaker identification"""
        combined_text = ""
        current_speaker = None

        for segment in whisper_result["segments"]:
            segment_start = segment["start"]
            segment_text = segment["text"].strip()

            # Find which speaker was talking during this segment
            speaker = self._find_speaker_for_time(segment_start, speaker_segments)

            if speaker != current_speaker:
                if combined_text:
                    combined_text += "\n\n"
                combined_text += f"{speaker}: "
                current_speaker = speaker

            combined_text += segment_text + " "

        return combined_text.strip()

    def _find_speaker_for_time(self, timestamp, speaker_segments):
        """Find which speaker was talking at a given timestamp"""
        for segment in speaker_segments:
            if segment["start"] <= timestamp <= segment["end"]:
                return segment["speaker"]
        return "Speaker"


class DungeonScribeApp:
    """Main application GUI"""

    def __init__(self, root):
        self.root = root
        self.root.title("Dungeon Scribe - D&D Session Recorder")
        self.root.geometry("800x600")
        self.root.configure(bg="#2c1810")

        # Initialize components
        self.audio_recorder = AudioRecorder()
        self.transcription = TranscriptionEngine()

        # Track last download location for folder opening
        self.last_download_folder = None

        # Recording state
        self.recording = False
        self.paused = False
        self.live_transcription_queue = queue.Queue()
        self.full_transcript = ""

        # Create GUI
        self.create_widgets()

        # Initialize engines
        self.initialize_engines()

        # Set up live transcription
        self.audio_recorder.set_live_transcription_callback(
            self.handle_live_transcription
        )

        # Start live transcription processor
        self.start_live_transcription_processor()

        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        """Create the main GUI widgets"""
        # Title
        title_label = tk.Label(
            self.root,
            text="🐉 Dungeon Scribe 🐉",
            font=("Georgia", 24, "bold"),
            fg="#daa520",
            bg="#2c1810",
        )
        title_label.pack(pady=20)

        # Control buttons frame
        controls_frame = tk.Frame(self.root, bg="#2c1810")
        controls_frame.pack(pady=10)

        self.start_btn = tk.Button(
            controls_frame,
            text="🎙️ Start Recording",
            command=self.start_recording,
            font=("Georgia", 12, "bold"),
            bg="#8b4513",
            fg="#daa520",
            width=16,
            height=2,
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.pause_btn = tk.Button(
            controls_frame,
            text="⏸️ Pause",
            command=self.pause_recording,
            font=("Georgia", 12, "bold"),
            bg="#b8860b",
            fg="#daa520",
            width=16,
            height=2,
            state=tk.DISABLED,
        )
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(
            controls_frame,
            text="⏹️ Stop & Finalize",
            command=self.stop_recording,
            font=("Georgia", 12, "bold"),
            bg="#dc143c",
            fg="#daa520",
            width=16,
            height=2,
            state=tk.DISABLED,
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Status label with recording indicator
        status_frame = tk.Frame(self.root, bg="#2c1810")
        status_frame.pack(pady=15)

        self.status_label = tk.Label(
            status_frame,
            text="Initializing...",
            font=("Georgia", 12, "bold"),
            fg="#daa520",
            bg="#2c1810",
        )
        self.status_label.pack(side=tk.LEFT)

        self.live_indicator = tk.Label(
            status_frame, text="", font=("Georgia", 10), fg="#32cd32", bg="#2c1810"
        )
        self.live_indicator.pack(side=tk.LEFT, padx=(10, 0))

        # Transcript area
        transcript_frame = tk.Frame(self.root, bg="#2c1810")
        transcript_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        tk.Label(
            transcript_frame,
            text="Session Transcript:",
            font=("Georgia", 14, "bold"),
            fg="#daa520",
            bg="#2c1810",
        ).pack(anchor=tk.W, pady=(0, 10))

        self.transcript_text = scrolledtext.ScrolledText(
            transcript_frame,
            font=("Georgia", 12),
            bg="#654321",
            fg="#daa520",
            height=15,
            wrap=tk.WORD,
            insertbackground="#daa520",
        )
        self.transcript_text.pack(fill=tk.BOTH, expand=True)

        # Bottom controls
        bottom_frame = tk.Frame(self.root, bg="#2c1810")
        bottom_frame.pack(fill=tk.X, padx=20, pady=20)

        # Filename entry
        filename_frame = tk.Frame(bottom_frame, bg="#2c1810")
        filename_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(
            filename_frame,
            text="Filename:",
            font=("Georgia", 12, "bold"),
            fg="#daa520",
            bg="#2c1810",
        ).pack(side=tk.LEFT)

        self.filename_entry = tk.Entry(
            filename_frame,
            font=("Georgia", 12),
            bg="#654321",
            fg="#daa520",
            insertbackground="#daa520",
        )
        self.filename_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 5))

        tk.Label(
            filename_frame,
            text=".txt",
            font=("Georgia", 12, "bold"),
            fg="#daa520",
            bg="#2c1810",
        ).pack(side=tk.LEFT)

        # Set default filename
        default_name = f"dnd_session_{datetime.now().strftime('%Y%m%d_%H%M')}"
        self.filename_entry.insert(0, default_name)

        # Action buttons
        action_frame = tk.Frame(bottom_frame, bg="#2c1810")
        action_frame.pack(fill=tk.X, pady=5)

        self.download_btn = tk.Button(
            action_frame,
            text="📜 Download Transcript",
            command=self.download_transcript,
            font=("Georgia", 10, "bold"),
            bg="#8b4513",
            fg="#daa520",
            width=18,
            height=2,
        )
        self.download_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = tk.Button(
            action_frame,
            text="🗑️ Clear",
            command=self.clear_transcript,
            font=("Georgia", 10, "bold"),
            bg="#8b4513",
            fg="#daa520",
            width=18,
            height=2,
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.folder_btn = tk.Button(
            action_frame,
            text="📁 Open Folder",
            command=self.open_downloads_folder,
            font=("Georgia", 10, "bold"),
            bg="#8b4513",
            fg="#daa520",
            width=18,
            height=2,
        )
        self.folder_btn.pack(side=tk.LEFT, padx=5)

    def initialize_engines(self):
        """Initialize all processing engines"""

        def init_worker():
            # Initialize Whisper and speaker detection
            self.root.after(
                0,
                lambda: self.update_status(
                    "Loading Whisper model and speaker detection..."
                ),
            )
            whisper_ok = self.transcription.initialize()

            # Update UI with results
            self.root.after(0, lambda: self.initialization_complete(whisper_ok))

        threading.Thread(target=init_worker, daemon=True).start()

    def initialization_complete(self, whisper_ok):
        """Called when initialization is complete"""
        if whisper_ok:
            if (
                self.transcription.speaker_detector
                and self.transcription.speaker_detector.enabled
            ):
                method = self.transcription.speaker_detector.method
                status = f"✅ Ready - Transcription with {method} speaker detection!"
            else:
                status = "✅ Ready - Transcription available (no speaker detection)"
            self.start_btn.config(state=tk.NORMAL)
        else:
            status = "❌ Error - Whisper model failed to load"
            self.start_btn.config(state=tk.DISABLED)

        self.update_status(status)

    def start_live_transcription_processor(self):
        """Start the live transcription processor thread"""

        def live_processor():
            while True:
                try:
                    audio_data = self.live_transcription_queue.get(timeout=1)
                    if audio_data is None:  # Shutdown signal
                        break

                    # Quick transcription for live feedback
                    if len(audio_data) > 0 and np.max(np.abs(audio_data)) > 0.005:
                        try:
                            result = self.transcription.model.transcribe(audio_data)
                            live_text = result.get("text", "").strip()
                            if live_text:
                                self.root.after(
                                    0,
                                    lambda t=live_text: self.update_live_transcription(
                                        t
                                    ),
                                )
                        except:
                            pass  # Ignore live transcription errors

                except queue.Empty:
                    continue
                except:
                    break

        threading.Thread(target=live_processor, daemon=True).start()

    def handle_live_transcription(self, audio_data):
        """Handle live audio for transcription"""
        try:
            self.live_transcription_queue.put_nowait(audio_data)
        except queue.Full:
            pass  # Skip if queue is full

    def update_live_transcription(self, text):
        """Update the live transcription indicator"""
        # Show last few words as live preview
        words = text.split()
        preview = " ".join(words[-8:]) if len(words) > 8 else text
        self.live_indicator.config(text=f"🔴 Live: ...{preview}")

        # Add to full transcript for context
        self.full_transcript += " " + text

    def update_status(self, message):
        """Update status label"""
        self.status_label.config(text=message)

    def start_recording(self):
        """Start audio recording with live transcription"""
        try:
            # Test microphone access first
            success, mic_info = self.audio_recorder.test_microphone()

            if not success:
                messagebox.showerror(
                    "Microphone Error",
                    f"Cannot access microphone: {mic_info}\n\n"
                    "Please check:\n"
                    "• Microphone is connected\n"
                    "• Windows microphone permissions\n"
                    "• No other apps using microphone",
                )
                return

            self.update_status(f"🎤 Recording... Speak clearly!")
            self.live_indicator.config(text="🔴 Live transcription starting...")

            self.recording = True
            self.paused = False
            self.full_transcript = ""

            # Update button states
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL, text="⏸️ Pause")
            self.stop_btn.config(state=tk.NORMAL)
            self.download_btn.config(state=tk.DISABLED)
            self.clear_btn.config(state=tk.DISABLED)

            # Start recording thread
            threading.Thread(target=self.recording_worker, daemon=True).start()

        except Exception as e:
            messagebox.showerror("Recording Error", f"Failed to start recording: {e}")
            self.reset_buttons()

    def pause_recording(self):
        """Pause or resume recording"""
        if not self.paused:
            # Pause
            self.paused = True
            self.audio_recorder.pause_recording()
            self.pause_btn.config(text="▶️ Resume")
            self.update_status("⏸️ Recording paused")
            self.live_indicator.config(text="⏸️ Paused")
        else:
            # Resume
            self.paused = False
            self.audio_recorder.resume_recording()
            self.pause_btn.config(text="⏸️ Pause")
            self.update_status("🎤 Recording resumed... Speak clearly!")
            self.live_indicator.config(text="🔴 Live transcription active...")

    def stop_recording(self):
        """Stop recording and process final transcription"""
        self.recording = False
        self.paused = False

        # Update UI immediately
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="⏸️ Pause")
        self.stop_btn.config(state=tk.DISABLED)
        self.live_indicator.config(text="")
        self.update_status("⏳ Finalizing transcription...")

        # Process final audio in separate thread
        threading.Thread(target=self.process_final_audio, daemon=True).start()

    def recording_worker(self):
        """Worker thread for audio recording with live updates"""
        try:
            self.audio_recorder.start_recording()

            while self.recording:
                if not self.audio_recorder.record_chunk():
                    break
                time.sleep(0.01)  # Small delay to prevent high CPU usage

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Recording Error", str(e)))
            self.root.after(0, self.reset_buttons)

    def process_final_audio(self):
        """Process the complete recorded audio for final transcription"""
        try:
            # Get complete audio data
            audio_data = self.audio_recorder.stop_recording()

            print(f"Final audio data length: {len(audio_data)}")

            if len(audio_data) == 0:
                self.root.after(0, lambda: self.update_status("❌ No audio recorded"))
                self.root.after(0, self.reset_buttons)
                return

            # Check audio level
            audio_level = np.max(np.abs(audio_data))
            print(f"Final audio level: {audio_level}")

            if audio_level < 0.001:
                self.root.after(0, lambda: self.update_status("❌ Audio too quiet"))
                self.root.after(0, self.reset_buttons)
                return

            # Final high-quality transcription of complete audio
            self.root.after(
                0,
                lambda: self.update_status(
                    "⏳ Creating final transcription with speaker detection..."
                ),
            )
            final_transcript = self.transcription.transcribe_audio(
                audio_data, use_speaker_detection=True
            )

            print(f"Final transcript: '{final_transcript}'")

            # Check if transcription worked
            if final_transcript.startswith("Error:"):
                self.root.after(0, lambda: self.update_status(f"❌ {final_transcript}"))
                self.root.after(0, self.reset_buttons)
                return
            elif not final_transcript or final_transcript.strip() == "":
                self.root.after(
                    0,
                    lambda: self.update_status("❌ No speech detected in final audio"),
                )
                self.root.after(0, self.reset_buttons)
                return

            # Update UI with final transcript
            self.root.after(0, lambda: self.show_final_transcript(final_transcript))

        except Exception as e:
            error_msg = f"Final processing failed: {e}"
            print(f"Process final audio error: {error_msg}")
            self.root.after(
                0, lambda: messagebox.showerror("Processing Error", error_msg)
            )
            self.root.after(0, lambda: self.update_status("❌ Final processing failed"))
            self.root.after(0, self.reset_buttons)

    def show_final_transcript(self, transcript):
        """Display final transcript and enable download"""
        self.transcript_text.delete(1.0, tk.END)
        self.transcript_text.insert(1.0, transcript)

        # Update status and enable buttons
        self.update_status(
            "✅ Recording complete! Final transcription with speaker detection ready."
        )
        self.live_indicator.config(text="")
        self.reset_buttons()

    def reset_buttons(self):
        """Reset button states"""
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED, text="⏸️ Pause")
        self.stop_btn.config(state=tk.DISABLED)
        self.download_btn.config(state=tk.NORMAL)
        self.clear_btn.config(state=tk.NORMAL)

    def clear_transcript(self):
        """Clear the transcript area and reset live transcription"""
        self.transcript_text.delete(1.0, tk.END)
        self.full_transcript = ""
        self.live_indicator.config(text="")
        self.update_status("Ready for new recording")

    def download_transcript(self):
        """Download transcript to text file in app's transcripts folder"""
        transcript = self.transcript_text.get(1.0, tk.END).strip()

        if not transcript:
            messagebox.showwarning("No Content", "No transcript to download!")
            return

        filename = self.filename_entry.get().strip()
        if not filename:
            filename = f"dnd_session_{datetime.now().strftime('%Y%m%d_%H%M')}"

        # Create transcripts folder next to the app if it doesn't exist
        app_dir = os.path.dirname(os.path.abspath(__file__))
        transcripts_dir = os.path.join(app_dir, "transcripts")

        if not os.path.exists(transcripts_dir):
            try:
                os.makedirs(transcripts_dir)
                print(f"Created transcripts folder: {transcripts_dir}")
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Could not create transcripts folder:\n{e}"
                )
                return

        # Create full file path
        file_path = os.path.join(transcripts_dir, f"{filename}.txt")

        # If file already exists, add a number to make it unique
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(transcripts_dir, f"{filename}_{counter}.txt")
            counter += 1

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                # Add header with session info
                f.write(f"D&D Session Transcript\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if (
                    self.transcription.speaker_detector
                    and self.transcription.speaker_detector.enabled
                ):
                    f.write(
                        f"Speaker Detection: {self.transcription.speaker_detector.method}\n"
                    )
                else:
                    f.write(f"Speaker Detection: Not available\n")
                f.write("=" * 50 + "\n\n")
                f.write(transcript)

            # Store the transcripts folder for the "Open Folder" button
            self.last_download_folder = transcripts_dir

            messagebox.showinfo("Success", f"Transcript saved to:\n{file_path}")
            self.update_status(f"Transcript saved as {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")
        """Download transcript to text file in app's transcripts folder"""
        transcript = self.transcript_text.get(1.0, tk.END).strip()

        if not transcript:
            messagebox.showwarning("No Content", "No transcript to download!")
            return

        filename = self.filename_entry.get().strip()
        if not filename:
            filename = f"dnd_session_{datetime.now().strftime('%Y%m%d_%H%M')}"

        # Create transcripts folder next to the app if it doesn't exist
        app_dir = os.path.dirname(os.path.abspath(__file__))
        transcripts_dir = os.path.join(app_dir, "transcripts")

        if not os.path.exists(transcripts_dir):
            try:
                os.makedirs(transcripts_dir)
                print(f"Created transcripts folder: {transcripts_dir}")
            except Exception as e:
                messagebox.showerror(
                    "Error", f"Could not create transcripts folder:\n{e}"
                )
                return

        # Create full file path
        file_path = os.path.join(transcripts_dir, f"{filename}.txt")

        # If file already exists, add a number to make it unique
        counter = 1
        original_path = file_path
        while os.path.exists(file_path):
            name_without_ext = filename
            file_path = os.path.join(
                transcripts_dir, f"{name_without_ext}_{counter}.txt"
            )
            counter += 1

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                # Add header with session info
                f.write(f"D&D Session Transcript\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                if (
                    self.transcription.speaker_detector
                    and self.transcription.speaker_detector.enabled
                ):
                    f.write(
                        f"Speaker Detection: {self.transcription.speaker_detector.method}\n"
                    )
                else:
                    f.write(f"Speaker Detection: Not available\n")
                f.write("=" * 50 + "\n\n")
                f.write(transcript)

            # Store the transcripts folder for the "Open Folder" button
            self.last_download_folder = transcripts_dir

            messagebox.showinfo("Success", f"Transcript saved to:\n{file_path}")
            self.update_status(f"Transcript saved as {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not create transcripts folder:\n{e}")
            return

        # Create full file path
        file_path = os.path.join(transcripts_dir, f"{filename}.txt")

        # If file already exists, add a number to make it unique
        counter = 1
        original_path = file_path
        while os.path.exists(file_path):
            name_without_ext = filename
            file_path = os.path.join(
                transcripts_dir, f"{name_without_ext}_{counter}.txt"
            )
            counter += 1

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                # Add header with session info
                f.write(f"D&D Session Transcript\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(
                    f"Speaker Detection: {'Enhanced Algorithm' if self.speaker_detection.enabled else 'Basic'}\n"
                )
                f.write("=" * 50 + "\n\n")
                f.write(transcript)

            # Store the transcripts folder for the "Open Folder" button
            self.last_download_folder = transcripts_dir

            messagebox.showinfo("Success", f"Transcript saved to:\n{file_path}")
            self.update_status(f"Transcript saved as {os.path.basename(file_path)}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")

    def open_downloads_folder(self):
        """Open the transcripts folder"""
        try:
            # Always use the transcripts folder next to the app
            app_dir = os.path.dirname(os.path.abspath(__file__))
            transcripts_dir = os.path.join(app_dir, "transcripts")

            # Create folder if it doesn't exist
            if not os.path.exists(transcripts_dir):
                os.makedirs(transcripts_dir)

            # Open the folder using system default
            import subprocess
            import platform

            system = platform.system()
            if system == "Windows":
                os.startfile(transcripts_dir)
            elif system == "Darwin":  # macOS
                subprocess.run(["open", transcripts_dir])
            else:  # Linux
                subprocess.run(["xdg-open", transcripts_dir])

            self.update_status(f"Opened transcripts folder")

        except Exception as e:
            # Fallback: Show folder path in a dialog
            messagebox.showinfo(
                "Folder Location",
                f"Transcripts folder:\n{transcripts_dir}\n\n"
                f"Could not open folder automatically: {e}",
            )
            print(f"Could not open folder: {e}")

    def on_closing(self):
        """Handle application closing"""
        if self.recording:
            self.recording = False
            time.sleep(0.1)  # Give recording thread time to stop

        self.audio_recorder.cleanup()
        self.root.destroy()


def check_dependencies(loading_screen):
    """Check for required dependencies with loading updates"""
    missing_deps = []

    loading_screen.update_status("Checking Dependencies...", "Verifying PyAudio...")
    try:
        import pyaudio

        loading_screen.update_status("Checking Dependencies...", "PyAudio found")
    except ImportError:
        missing_deps.append("pyaudio")
        loading_screen.update_status("Checking Dependencies...", "PyAudio missing")

    time.sleep(0.3)  # Brief pause for visual feedback

    loading_screen.update_status("Checking Dependencies...", "Verifying Whisper...")
    try:
        import whisper

        loading_screen.update_status("Checking Dependencies...", "Whisper found")
    except ImportError:
        missing_deps.append("openai-whisper")
        loading_screen.update_status("Checking Dependencies...", "Whisper missing")

    time.sleep(0.3)

    loading_screen.update_status("Checking Dependencies...", "Verifying NumPy...")
    try:
        import numpy

        loading_screen.update_status("Checking Dependencies...", "NumPy found")
    except ImportError:
        missing_deps.append("numpy")
        loading_screen.update_status("Checking Dependencies...", "NumPy missing")

    time.sleep(0.3)

    loading_screen.update_status(
        "Checking Dependencies...", "Checking speaker identification..."
    )
    try:
        from pyannote.audio import Pipeline

        loading_screen.update_status("Checking Dependencies...", "Speaker ID available")
    except ImportError:
        loading_screen.update_status(
            "Checking Dependencies...", "Speaker ID unavailable (optional)"
        )

    time.sleep(0.5)

    if missing_deps:
        loading_screen.update_status(
            "Missing Dependencies", "Installation required - see console"
        )
        time.sleep(2)
        return missing_deps
    else:
        loading_screen.update_status(
            "All Dependencies Found", "Launching application..."
        )
        time.sleep(1)
        return []


def main():
    """Main application entry point"""
    print("Starting Dungeon Scribe...")

    # Import tkinter at the beginning of main
    import tkinter as tk
    from tkinter import messagebox

    # Show loading screen
    loading_screen = LoadingScreen()

    # Check dependencies
    missing = check_dependencies(loading_screen)

    if missing:
        loading_screen.close()

        # Show error in console
        print("\n" + "=" * 50)
        print("ERROR: Missing required dependencies!")
        print("=" * 50)
        print("\nPlease install the following packages:")
        for dep in missing:
            print(f"  pip install {dep}")
        print("\nFor full speaker identification support (optional):")
        print("  pip install pyannote.audio")
        print("=" * 50)

        # Show GUI error
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Missing Dependencies",
            f"Missing required packages:\n{', '.join(missing)}\n\n"
            "Please install them using:\n"
            f"pip install {' '.join(missing)}",
        )
        return

    # Close loading screen
    loading_screen.close()

    try:
        # Create main application
        root = tk.Tk()
        app = DungeonScribeApp(root)

        # Handle window closing
        root.protocol("WM_DELETE_WINDOW", app.on_closing)

        print("Application launched successfully")

        # Run the application
        root.mainloop()

    except Exception as e:
        print(f"Error launching application: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
