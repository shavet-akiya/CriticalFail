"use client";
import { useRef, useState, useEffect } from "react";
import {
    Upload,
    Mic,
    MicOff,
    Pause,
    Play,
    CheckCircle,
    XCircle,
    Loader2,
    Database,
} from "lucide-react";

export default function NewSession() {
    const fileInputRef = useRef<HTMLInputElement>(null);

    // Upload states
    const [isUploading, setIsUploading] = useState(false);
    const [uploadStatus, setUploadStatus] = useState<string>("");
    const [uploadError, setUploadError] = useState<string>("");

    // Recording states
    const [isRecording, setIsRecording] = useState(false);
    const [isPaused, setIsPaused] = useState(false);
    const [recordingTime, setRecordingTime] = useState(0);
    const [audioBlob, setAudioBlob] = useState<Blob | null>(null);

    // Processing states
    const [currentJobId, setCurrentJobId] = useState<string | null>(null);
    const [completedTranscript, setCompletedTranscript] = useState<
        string | null
    >(null);
    const [speakerCount, setSpeakerCount] = useState<number>(0);

    // Refs for recording
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<Blob[]>([]);
    const timerRef = useRef<NodeJS.Timeout | null>(null);
    const streamRef = useRef<MediaStream | null>(null);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            if (timerRef.current) clearInterval(timerRef.current);
            if (streamRef.current) {
                streamRef.current.getTracks().forEach((track) => track.stop());
            }
        };
    }, []);

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
            setUploadError("");
            setCompletedTranscript(null);

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
                const blob = new Blob(audioChunksRef.current, {
                    type: mimeType,
                });
                setAudioBlob(blob);
                // Auto-upload after recording stops
                uploadRecording(blob);
            };

            mediaRecorder.start(1000);
            setIsRecording(true);
            setRecordingTime(0);
        } catch (err) {
            console.error("Error starting recording:", err);
            setUploadError(
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

    const uploadRecording = async (blob: Blob) => {
        setIsUploading(true);
        setUploadStatus("Uploading recorded audio...");

        try {
            const formData = new FormData();
            const fileName = `recording_${new Date()
                .toISOString()
                .replace(/[:.]/g, "-")}.webm`;
            formData.append("file", blob, fileName);
            formData.append("min_speakers", "2");
            formData.append("max_speakers", "8");

            await processAudioJob(formData);
        } catch (error: any) {
            console.error("Error:", error);
            setUploadError(error.message || "Failed to process recording");
            setUploadStatus("");
            setIsUploading(false);
        }
    };

    const handleUploadClick = () => {
        if (!isUploading && !isRecording) {
            fileInputRef.current?.click();
        }
    };

    const handleFileChange = async (
        event: React.ChangeEvent<HTMLInputElement>
    ) => {
        const file = event.target.files?.[0];
        if (!file) return;

        console.log("Selected file:", file.name);
        setIsUploading(true);
        setUploadStatus("Uploading audio file...");
        setUploadError("");
        setCompletedTranscript(null);

        try {
            const formData = new FormData();
            formData.append("file", file);
            formData.append("min_speakers", "2");
            formData.append("max_speakers", "8");

            await processAudioJob(formData);
        } catch (error: any) {
            console.error("Error:", error);
            setUploadError(error.message || "Failed to process audio file");
            setUploadStatus("");
            setIsUploading(false);
        }
    };

    const processAudioJob = async (formData: FormData) => {
        // Submit job
        const response = await fetch("/api/speech/upload", {
            method: "POST",
            body: formData,
        });

        if (!response.ok) {
            throw new Error("Upload failed");
        }

        const result = await response.json();
        console.log("Job submitted:", result.job_id);

        if (!result.job_id) {
            throw new Error("No job ID returned");
        }

        setCurrentJobId(result.job_id);

        // Poll for completion
        setUploadStatus(
            `Processing audio (Job: ${result.job_id.substring(0, 8)}...)`
        );

        let attempts = 0;
        const maxAttempts = 400; // 20 minutes

        while (attempts < maxAttempts) {
            await new Promise((resolve) => setTimeout(resolve, 3000));
            attempts++;

            const statusResponse = await fetch(
                `/api/speech/status/${result.job_id}`
            );
            if (!statusResponse.ok) {
                console.error("Status check failed");
                continue;
            }

            const statusData = await statusResponse.json();
            console.log("Job status:", statusData.status);

            if (statusData.status === "processing") {
                setUploadStatus(
                    `Processing audio... (${Math.floor((attempts * 3) / 60)}m ${
                        (attempts * 3) % 60
                    }s)`
                );
            }

            if (statusData.status === "completed" && statusData.result) {
                setUploadStatus(
                    `✅ Processing complete! ${
                        statusData.result.speaker_count || 0
                    } speakers identified.`
                );
                setCompletedTranscript(statusData.result.transcript || "");
                setSpeakerCount(statusData.result.speaker_count || 0);
                setIsUploading(false);

                setTimeout(() => setUploadStatus(""), 3000);
                break;
            }

            if (statusData.status === "failed") {
                throw new Error(statusData.error || "Processing failed");
            }
        }

        if (attempts >= maxAttempts) {
            throw new Error("Processing timeout");
        }
    };

    const handleSaveToDatabase = async () => {
        if (!completedTranscript) return;

        try {
            setUploadStatus("🧠 Processing transcript with AI...");
            setIsUploading(true);
            setUploadError("");

            const response = await fetch("/api/sessions", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ transcript: completedTranscript }),
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.details || "Failed to save session");
            }

            const data = await response.json();

            console.log("LLM processed session:", data);
            setUploadStatus("✅ Session saved to database!");
            setIsUploading(false);

            // Optional: you could show the structured output here
            alert("Session saved successfully!");
        } catch (err: any) {
            console.error(err);
            setUploadError(err.message || "Failed to save session");
            setIsUploading(false);
            setUploadStatus("");
        }
    };

    const isProcessing = isUploading || isRecording;

    return (
        <div className="flex flex-col items-center justify-center min-h-[80vh] gap-8 p-8">
            {/* Header */}
            <div className="text-center">
                <h1 className="text-5xl font-bold text-gray-800 mb-4">
                    Create New Session
                </h1>
                <p className="text-lg text-gray-600">
                    Upload audio or start recording your D&D session
                </p>
            </div>

            {/* Recording Timer */}
            {isRecording && (
                <div className="text-center">
                    <div className="text-4xl font-mono font-bold text-gray-700 mb-2">
                        {formatTime(recordingTime)}
                    </div>
                    <div className="flex items-center justify-center bg-red-50 rounded-lg px-4 py-2 inline-flex">
                        <div
                            className={`w-3 h-3 rounded-full mr-3 ${
                                isPaused
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
                </div>
            )}

            {/* Action Buttons */}
            <div className="flex flex-col items-center gap-4">
                <div className="flex gap-6">
                    <button
                        className="flex items-center gap-3 px-8 py-4 bg-indigo-600 text-white text-lg font-semibold rounded-lg hover:bg-indigo-700 transition-colors shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
                        onClick={handleUploadClick}
                        disabled={isProcessing}
                    >
                        {isUploading && !isRecording ? (
                            <>
                                <Loader2 className="animate-spin" size={24} />
                                Processing...
                            </>
                        ) : (
                            <>
                                <Upload size={24} />
                                Upload Audio File
                            </>
                        )}
                    </button>

                    {!isRecording ? (
                        <button
                            className="flex items-center gap-3 px-8 py-4 bg-red-500 text-white text-lg font-semibold rounded-lg hover:bg-red-600 transition-colors shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
                            onClick={startRecording}
                            disabled={isUploading}
                        >
                            <Mic size={24} />
                            Start Recording
                        </button>
                    ) : (
                        <button
                            className="flex items-center gap-3 px-8 py-4 bg-gray-600 text-white text-lg font-semibold rounded-lg hover:bg-gray-700 transition-colors shadow-lg"
                            onClick={stopRecording}
                        >
                            <MicOff size={24} />
                            Stop Recording
                        </button>
                    )}
                </div>

                {/* Pause Button (appears when recording) */}
                {isRecording && (
                    <button
                        onClick={pauseRecording}
                        className="flex items-center gap-2 px-6 py-2 bg-yellow-500 text-white font-semibold rounded-lg hover:bg-yellow-600 transition-colors shadow"
                    >
                        {isPaused ? (
                            <>
                                <Play size={18} />
                                Resume
                            </>
                        ) : (
                            <>
                                <Pause size={18} />
                                Pause
                            </>
                        )}
                    </button>
                )}
            </div>

            {/* Status Messages */}
            {uploadStatus && !completedTranscript && (
                <div
                    className={`flex items-center gap-3 p-4 rounded-lg shadow ${
                        uploadStatus.includes("✅")
                            ? "bg-green-50 border-l-4 border-green-500"
                            : "bg-blue-50 border-l-4 border-blue-500"
                    }`}
                >
                    {uploadStatus.includes("✅") ? (
                        <CheckCircle className="text-green-600" size={20} />
                    ) : (
                        <Loader2
                            className="animate-spin text-blue-600"
                            size={20}
                        />
                    )}
                    <p
                        className={
                            uploadStatus.includes("✅")
                                ? "text-green-800 font-medium"
                                : "text-blue-800 font-medium"
                        }
                    >
                        {uploadStatus}
                    </p>
                </div>
            )}

            {uploadError && (
                <div className="flex items-center gap-3 p-4 bg-red-50 border-l-4 border-red-500 rounded-lg shadow">
                    <XCircle className="text-red-600" size={20} />
                    <p className="text-red-800 font-medium">{uploadError}</p>
                </div>
            )}

            {/* Completed Transcript Display */}
            {completedTranscript && (
                <div className="w-full max-w-4xl mt-8 p-6 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg shadow-lg border-2 border-green-200">
                    <div className="flex items-center justify-between mb-4">
                        <div>
                            <h3 className="text-2xl font-bold text-gray-800 flex items-center">
                                <CheckCircle
                                    className="mr-2 text-green-600"
                                    size={28}
                                />
                                Transcription Complete
                            </h3>
                            <p className="text-sm text-gray-600 mt-1">
                                {speakerCount} speaker
                                {speakerCount !== 1 ? "s" : ""} identified
                            </p>
                        </div>
                    </div>

                    <div className="bg-white p-6 rounded-lg border border-gray-200 max-h-96 overflow-y-auto mb-4">
                        <pre className="whitespace-pre-wrap font-sans text-sm text-gray-700">
                            {completedTranscript}
                        </pre>
                    </div>

                    <div className="flex justify-end">
                        <button
                            onClick={handleSaveToDatabase}
                            className="flex items-center gap-2 px-6 py-3 bg-indigo-600 text-white font-semibold rounded-lg hover:bg-indigo-700 transition-colors shadow"
                        >
                            <Database size={20} />
                            Save to Database
                        </button>
                    </div>
                </div>
            )}

            <input
                type="file"
                accept="audio/*"
                ref={fileInputRef}
                onChange={handleFileChange}
                className="hidden"
            />
        </div>
    );
}
