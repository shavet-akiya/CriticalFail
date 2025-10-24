"use client";
import React, { useState, useRef, useCallback, useEffect } from "react";

import {
    Mic,
    MicOff,
    Pause,
    Play,
    Download,
    Upload,
    Loader2,
    Users,
    Trash2,
    CheckCircle,
    XCircle,
    Clock,
} from "lucide-react";

interface JobStatus {
    job_id: string;
    status: "queued" | "processing" | "completed" | "failed";
    progress?: string;
    created_at: string;
    completed_at?: string;
    result?: {
        success: boolean;
        transcript: string;
        file_path?: string;
        error?: string;
        speakers?: string[];
        speaker_count?: number;
    };
    error?: string;
}

const API_BASE =
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:9000";
const RecordingInterface: React.FC = () => {
    // Recording states
    const [isRecording, setIsRecording] = useState(false);
    const [isPaused, setIsPaused] = useState(false);
    const [recordingTime, setRecordingTime] = useState(0);

    // Audio data
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
    const [audioUrl, setAudioUrl] = useState<string | null>(null);

    // Job processing
    const [currentJobId, setCurrentJobId] = useState<string | null>(null);
    const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
    const [isPolling, setIsPolling] = useState(false);

    // Results
    const [error, setError] = useState<string | null>(null);

    // Settings
    const [minSpeakers, setMinSpeakers] = useState(2);
    const [maxSpeakers, setMaxSpeakers] = useState(8);

    // Refs
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const timerRef = useRef<NodeJS.Timeout | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
            if (pollingIntervalRef.current)
                clearInterval(pollingIntervalRef.current);
            if (streamRef.current) {
                streamRef.current.getTracks().forEach((track) => track.stop());
            }
            if (audioUrl) {
                URL.revokeObjectURL(audioUrl);
            }
        };
    }, [audioUrl]);

    // Recording timer
    useEffect(() => {
        if (isRecording && !isPaused) {
            timerRef.current = setInterval(() => {
                setRecordingTime((prev) => prev + 1);
            }, 1000);
        } else {
            if (timerRef.current) {
                clearInterval(timerRef.current);
                timerRef.current = null;
            }
        }
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
        };
    }, [isRecording, isPaused]);

    // Poll job status
    const pollJobStatus = useCallback(async (jobId: string) => {
        try {
            const response = await fetch(`/api/speech/status/${jobId}`);
            if (!response.ok) {
                console.error("Failed to fetch job status");
                return;
            }

            const status: JobStatus = await response.json();
            setJobStatus(status);

            console.log(`Job ${jobId} status:`, status.status);

            // Stop polling if job is complete or failed
            if (status.status === "completed" || status.status === "failed") {
                setIsPolling(false);
                if (pollingIntervalRef.current) {
                    clearInterval(pollingIntervalRef.current);
                    pollingIntervalRef.current = null;
                }

                if (status.status === "failed") {
                    setError(status.error || "Processing failed");
                }
            }
        } catch (err) {
            console.error("Error polling job status:", err);
        }
    }, []);

    // Start polling when job is submitted
    useEffect(() => {
        if (isPolling && currentJobId) {
            pollJobStatus(currentJobId);

            pollingIntervalRef.current = setInterval(() => {
                pollJobStatus(currentJobId);
            }, 3000);

            return () => {
                if (pollingIntervalRef.current) {
                    clearInterval(pollingIntervalRef.current);
                    pollingIntervalRef.current = null;
                }
            };
        }
    }, [isPolling, currentJobId, pollJobStatus]);

    const formatTime = (seconds: number): string => {
        const hrs = Math.floor(seconds / 3600);
        const mins = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        return `${hrs.toString().padStart(2, "0")}:${mins
            .toString()
            .padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
    };

    const startRecording = async () => {
        try {
            setError(null);
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 16000,
                },
            });

            streamRef.current = stream;

            let mimeType = "audio/webm";
            const types = [
                "audio/webm;codecs=opus",
                "audio/webm",
                "audio/ogg;codecs=opus",
                "audio/mp4",
            ];

            for (const type of types) {
                if (MediaRecorder.isTypeSupported(type)) {
                    mimeType = type;
                    break;
                }
            }

            const mediaRecorder = new MediaRecorder(stream, { mimeType });
            mediaRecorderRef.current = mediaRecorder;
            audioChunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    audioChunksRef.current.push(event.data);
                }
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunksRef.current, {
                    type: mimeType,
                });
                setAudioBlob(audioBlob);
                const url = URL.createObjectURL(audioBlob);
                setAudioUrl(url);
            };

            mediaRecorder.start(1000);
            setIsRecording(true);
            setRecordingTime(0);
            setJobStatus(null);
            setCurrentJobId(null);
        } catch (err) {
            console.error("Error starting recording:", err);
            setError(
                "Failed to start recording. Please ensure microphone permissions are granted."
            );
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            setIsPaused(false);

            if (streamRef.current) {
                streamRef.current.getTracks().forEach((track) => track.stop());
                streamRef.current = null;
            }
        }
    };

    const pauseRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            if (isPaused) {
                mediaRecorderRef.current.resume();
                setIsPaused(false);
            } else {
                mediaRecorderRef.current.pause();
                setIsPaused(true);
            }
        }
    };

    const uploadAudio = async () => {
        if (!audioBlob) {
            setError("No audio to upload");
            return;
        }

        setError(null);
        setJobStatus(null);

        try {
            const formData = new FormData();
            const fileName = `recording_${new Date()
                .toISOString()
                .replace(/[:.]/g, "-")}.webm`;
            formData.append("file", audioBlob, fileName);
            formData.append("min_speakers", minSpeakers.toString());
            formData.append("max_speakers", maxSpeakers.toString());

            console.log("Submitting job:", fileName);

            const response = await fetch(`${API_BASE}/speech/upload`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || "Upload failed");
            }

            const data = await response.json();

            if (data.job_id) {
                console.log("Job submitted:", data.job_id);
                setCurrentJobId(data.job_id);
                setJobStatus({
                    job_id: data.job_id,
                    status: "queued",
                    progress: "Job queued for processing",
                    created_at: new Date().toISOString(),
                });
                setIsPolling(true);
            } else {
                throw new Error("No job ID returned");
            }
        } catch (err) {
            console.error("Upload error:", err);
            setError(
                err instanceof Error ? err.message : "Failed to upload audio"
            );
        }
    };

    const handleFileUpload = async (
        event: React.ChangeEvent<HTMLInputElement>
    ) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setError(null);
        setJobStatus(null);

        try {
            const formData = new FormData();
            formData.append("file", file);
            formData.append("min_speakers", minSpeakers.toString());
            formData.append("max_speakers", maxSpeakers.toString());

            console.log("Submitting file:", file.name);

            const response = await fetch("/api/speech/upload", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || "Upload failed");
            }

            const data = await response.json();

            if (data.job_id) {
                console.log("Job submitted:", data.job_id);
                setCurrentJobId(data.job_id);
                setJobStatus({
                    job_id: data.job_id,
                    status: "queued",
                    progress: "Job queued for processing",
                    created_at: new Date().toISOString(),
                });
                setIsPolling(true);
            } else {
                throw new Error("No job ID returned");
            }
        } catch (err) {
            console.error("Upload error:", err);
            setError(
                err instanceof Error
                    ? err.message
                    : "Failed to process audio file"
            );
        }
    };

    const downloadAudio = () => {
        if (audioUrl && audioBlob) {
            const a = document.createElement("a");
            a.href = audioUrl;
            a.download = `recording_${new Date()
                .toISOString()
                .replace(/[:.]/g, "-")}.webm`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }
    };

    const downloadTranscript = () => {
        if (jobStatus?.result?.transcript) {
            const blob = new Blob([jobStatus.result.transcript], {
                type: "text/plain",
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `transcript_${new Date()
                .toISOString()
                .replace(/[:.]/g, "-")}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    };

    const clearRecording = () => {
        setAudioBlob(null);
        if (audioUrl) {
            URL.revokeObjectURL(audioUrl);
            setAudioUrl(null);
        }
        setRecordingTime(0);
        setJobStatus(null);
        setCurrentJobId(null);
        setError(null);
        audioChunksRef.current = [];
    };

    const isProcessing = !!(
        isPolling &&
        jobStatus &&
        jobStatus.status !== "completed" &&
        jobStatus.status !== "failed"
    );

    return (
        <div className="max-w-6xl mx-auto p-6">
            <div className="bg-white rounded-lg shadow-lg p-6">
                <h2 className="text-3xl font-bold mb-6 text-gray-800">
                    Audio Processor
                </h2>

                {/* Error Display */}
                {error && (
                    <div className="mb-6 p-4 bg-red-50 border-l-4 border-red-500 text-red-700">
                        <p className="font-medium">Error</p>
                        <p className="text-sm">{error}</p>
                    </div>
                )}

                {/* Recording Controls */}
                <div className="mb-6 p-6 bg-gradient-to-r from-indigo-50 to-blue-50 rounded-lg">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center space-x-4">
                            {!isRecording ? (
                                <button
                                    onClick={startRecording}
                                    className="flex items-center px-6 py-3 bg-red-500 white-colour rounded-lg hover:bg-red-600 transition-colors shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
                                    disabled={isProcessing}
                                >
                                    <Mic className="mr-2" size={20} />
                                    Start Recording
                                </button>
                            ) : (
                                <>
                                    <button
                                        onClick={stopRecording}
                                        className="flex items-center px-6 py-3 bg-gray-600 white-colour rounded-lg hover:bg-gray-700 transition-colors shadow-md"
                                    >
                                        <MicOff className="mr-2" size={20} />
                                        Stop
                                    </button>
                                    <button
                                        onClick={pauseRecording}
                                        className="flex items-center px-6 py-3 bg-blue-500 white-colour rounded-lg hover:bg-blue-600 transition-colors shadow-md"
                                    >
                                        {isPaused ? (
                                            <>
                                                <Play
                                                    className="mr-2"
                                                    size={20}
                                                />
                                                Resume
                                            </>
                                        ) : (
                                            <>
                                                <Pause
                                                    className="mr-2"
                                                    size={20}
                                                />
                                                Pause
                                            </>
                                        )}
                                    </button>
                                </>
                            )}

                            {/* File Upload */}
                            <div className="relative">
                                <input
                                    type="file"
                                    accept="audio/*"
                                    onChange={handleFileUpload}
                                    className="hidden"
                                    id="file-upload"
                                    disabled={isRecording || isProcessing}
                                />
                                <label
                                    htmlFor="file-upload"
                                    className={`flex items-center px-6 py-3 bg-green-500 white-colour rounded-lg hover:bg-green-600 transition-colors shadow-md cursor-pointer ${isRecording || isProcessing
                                            ? "opacity-50 cursor-not-allowed"
                                            : ""
                                        }`}
                                >
                                    <Upload className="mr-2" size={20} />
                                    Upload Audio
                                </label>
                            </div>
                        </div>

                        <div className="text-2xl font-mono font-bold text-gray-700">
                            {formatTime(recordingTime)}
                        </div>
                    </div>

                    {/* Recording Status */}
                    {isRecording && (
                        <div className="flex items-center bg-white/50 rounded-lg px-4 py-2 inline-flex">
                            <div
                                className={`w-3 h-3 rounded-full mr-3 ${isPaused
                                        ? "bg-yellow-500"
                                        : "bg-red-500 animate-pulse"
                                    }`}
                            />
                            <span className="text-sm font-medium text-gray-700">
                                {isPaused
                                    ? "Recording Paused"
                                    : "Recording in Progress..."}
                            </span>
                        </div>
                    )}
                </div>

                {/* Speaker Settings */}
                <div className="mb-6 p-4 bg-gray-50 rounded-lg">
                    <div className="flex items-center mb-3">
                        <Users className="mr-2 text-indigo-600" size={20} />
                        <h3 className="font-semibold text-gray-800">
                            Speaker Detection Settings
                        </h3>
                    </div>
                    <p className="text-sm text-gray-600 mb-3">
                        These settings help the AI identify different speakers
                        in your audio. Adjust based on how many people are
                        speaking.
                    </p>
                    <div className="flex space-x-6">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Minimum Speakers
                            </label>
                            <input
                                type="number"
                                min="1"
                                max="10"
                                value={minSpeakers}
                                onChange={(e) =>
                                    setMinSpeakers(
                                        parseInt(e.target.value) || 2
                                    )
                                }
                                className="w-24 px-3 py-2 border border-gray-300 rounded-md bg-white text-gray-900 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                                disabled={isRecording || isProcessing}
                            />
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Maximum Speakers
                            </label>
                            <input
                                type="number"
                                min="2"
                                max="20"
                                value={maxSpeakers}
                                onChange={(e) =>
                                    setMaxSpeakers(
                                        parseInt(e.target.value) || 8
                                    )
                                }
                                className="w-24 px-3 py-2 border border-gray-300 rounded-md bg-white text-gray-900 focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                                disabled={isRecording || isProcessing}
                            />
                        </div>
                    </div>
                </div>

                {/* Audio Preview */}
                {audioUrl && !isRecording && (
                    <div className="mb-6 p-4 bg-gray-50 rounded-lg">
                        <h3 className="font-semibold mb-3 text-gray-800">
                            Recorded Audio
                        </h3>
                        <audio controls className="w-full mb-4">
                            <source src={audioUrl} />
                        </audio>

                        <div className="flex space-x-3">
                            <button
                                onClick={uploadAudio}
                                disabled={isProcessing}
                                className="flex items-center px-4 py-2 bg-indigo-600 white-colour rounded-lg hover:bg-indigo-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {isProcessing ? (
                                    <>
                                        <Loader2
                                            className="mr-2 animate-spin"
                                            size={18}
                                        />
                                        Processing...
                                    </>
                                ) : (
                                    <>
                                        <Upload className="mr-2" size={18} />
                                        Process Audio
                                    </>
                                )}
                            </button>

                            <button
                                onClick={downloadAudio}
                                className="flex items-center px-4 py-2 bg-blue-500 white-colour rounded-lg hover:bg-blue-600 transition-colors"
                            >
                                <Download className="mr-2" size={18} />
                                Download Audio
                            </button>

                            <button
                                onClick={clearRecording}
                                className="flex items-center px-4 py-2 bg-gray-500 white-colour rounded-lg hover:bg-gray-600 transition-colors"
                            >
                                <Trash2 className="mr-2" size={18} />
                                Clear
                            </button>
                        </div>
                    </div>
                )}

                {/* Job Status Display */}
                {jobStatus && (
                    <div className="mb-6 p-6 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg">
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-xl font-semibold text-gray-800">
                                Processing Status
                            </h3>
                            <div className="flex items-center">
                                {jobStatus.status === "queued" && (
                                    <Clock
                                        className="text-yellow-500 mr-2"
                                        size={24}
                                    />
                                )}
                                {jobStatus.status === "processing" && (
                                    <Loader2
                                        className="text-blue-500 mr-2 animate-spin"
                                        size={24}
                                    />
                                )}
                                {jobStatus.status === "completed" && (
                                    <CheckCircle
                                        className="text-green-500 mr-2"
                                        size={24}
                                    />
                                )}
                                {jobStatus.status === "failed" && (
                                    <XCircle
                                        className="text-red-500 mr-2"
                                        size={24}
                                    />
                                )}
                                <span className="font-medium text-gray-700 capitalize">
                                    {jobStatus.status}
                                </span>
                            </div>
                        </div>

                        <div className="bg-white p-4 rounded-lg mb-4">
                            <p className="text-sm text-gray-600">
                                Job ID:{" "}
                                <span className="font-mono text-xs">
                                    {jobStatus.job_id}
                                </span>
                            </p>
                            {jobStatus.progress && (
                                <p className="text-sm text-gray-700 mt-2">
                                    {jobStatus.progress}
                                </p>
                            )}
                        </div>

                        {/* Processing Animation */}
                        {isProcessing && (
                            <div className="text-center py-4">
                                <div className="inline-flex items-center px-6 py-3 bg-white rounded-lg shadow">
                                    <Loader2
                                        className="mr-3 animate-spin text-indigo-600"
                                        size={20}
                                    />
                                    <span className="text-gray-700">
                                        Processing audio with WhisperX AI...
                                    </span>
                                </div>
                                <p className="text-sm text-gray-600 mt-3">
                                    Large files may take 15-20 minutes. Please
                                    don't close this page.
                                </p>
                            </div>
                        )}

                        {/* Completed Results */}
                        {jobStatus.status === "completed" &&
                            jobStatus.result && (
                                <div className="mt-4">
                                    <div className="bg-green-100 border border-green-300 rounded-lg p-3 mb-4">
                                        <p className="text-green-800 font-medium flex items-center">
                                            <CheckCircle
                                                className="mr-2"
                                                size={18}
                                            />
                                            Transcription Complete!
                                        </p>
                                    </div>

                                    {/* Speaker Info */}
                                    {jobStatus.result.speaker_count !==
                                        undefined && (
                                            <div className="bg-white p-4 rounded-lg mb-4">
                                                <h4 className="font-medium text-gray-700 mb-2">
                                                    Speaker Analysis
                                                </h4>
                                                <p className="text-sm">
                                                    <span className="text-gray-600">
                                                        Speakers Detected:
                                                    </span>
                                                    <span className="ml-2 font-bold text-indigo-600">
                                                        {
                                                            jobStatus.result
                                                                .speaker_count
                                                        }
                                                    </span>
                                                </p>
                                                {jobStatus.result.speakers &&
                                                    jobStatus.result.speakers
                                                        .length > 0 && (
                                                        <p className="text-sm mt-1">
                                                            <span className="text-gray-600">
                                                                IDs:
                                                            </span>
                                                            <span className="ml-2 font-mono text-xs">
                                                                {jobStatus.result.speakers.join(
                                                                    ", "
                                                                )}
                                                            </span>
                                                        </p>
                                                    )}
                                            </div>
                                        )}

                                    {/* Transcript */}
                                    {jobStatus.result.transcript && (
                                        <div>
                                            <div className="flex justify-between items-center mb-2">
                                                <h4 className="font-medium text-gray-700">
                                                    Transcript
                                                </h4>
                                                <button
                                                    onClick={downloadTranscript}
                                                    className="flex items-center px-3 py-1 bg-green-600 white-colour text-sm rounded hover:bg-green-700"
                                                >
                                                    <Download
                                                        className="mr-1"
                                                        size={14}
                                                    />
                                                    Download
                                                </button>
                                            </div>
                                            <div className="bg-white p-4 rounded-lg border border-gray-200 max-h-96 overflow-y-auto">
                                                <pre className="whitespace-pre-wrap font-sans text-sm text-gray-700">
                                                    {
                                                        jobStatus.result
                                                            .transcript
                                                    }
                                                </pre>
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}

                        {/* Failed Status */}
                        {jobStatus.status === "failed" && (
                            <div className="bg-red-100 border border-red-300 rounded-lg p-4">
                                <p className="text-red-800 font-medium flex items-center">
                                    <XCircle className="mr-2" size={18} />
                                    Processing Failed
                                </p>
                                {jobStatus.error && (
                                    <p className="text-red-700 text-sm mt-2">
                                        {jobStatus.error}
                                    </p>
                                )}
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default RecordingInterface;
