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
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useRecording } from "@/contexts/RecordingContext";
import Loading from "@/components/Loading";
import { useParams } from "next/navigation";

export default function NewSession() {
    const baseUrl = "http://localhost:9000"; // FastAPI backend
    const fileInputRef = useRef<HTMLInputElement>(null);

    // inside the component
    const { campaignId } = useParams();

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
        const response = await fetch(`${baseUrl}/speech/upload`, {
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
                `${baseUrl}/speech/status/${result.job_id}`
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
            setUploadStatus("Sending transcript for processing...");
            setIsUploading(true);
            setUploadError("");

            // Send transcript for LLM processing
            const response = await fetch(`${baseUrl}/sessions`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    transcript: completedTranscript,
                    campaign_id: campaignId,
                }),
            });

            const { job_id } = await response.json();

            if (!job_id) throw new Error("No job ID returned from server.");

            setUploadStatus("AI is analyzing the session...");

            // Poll until job is completed
            let resultData = null;
            while (true) {
                const statusRes = await fetch(
                    `${baseUrl}/sessions/status/${job_id}`
                );
                const job = await statusRes.json();

                if (job.status === "completed") {
                    resultData = job.result;
                    break;
                } else if (job.status === "error") {
                    throw new Error(job.error || "LLM processing failed");
                }

                await new Promise((r) => setTimeout(r, 3000));
            }

            console.log("LLM processed session:", resultData);

            // Merge new characters into the campaign
            if (resultData?.characters?.length) {
                // Fetch existing campaign data
                const campaignRes = await fetch(
                    `${baseUrl}/campaigns/${campaignId}`
                );
                if (!campaignRes.ok)
                    throw new Error("Failed to fetch campaign data");
                const campaignData = await campaignRes.json();

                const existingChars = campaignData.characters || [];
                const newChars = resultData.characters;

                // Merge characters by name
                const mergedChars = [...existingChars];

                newChars.forEach((char: any) => {
                    const existingChar = mergedChars.find(
                        (c) => c.name.toLowerCase() === char.name.toLowerCase()
                    );
                    if (existingChar) {
                        // Append new session_id if not already included
                        if (
                            !existingChar.session_ids.includes(
                                resultData.session_id
                            )
                        ) {
                            existingChar.session_ids.push(
                                resultData.session_id
                            );
                        }
                    } else {
                        // Add entirely new character
                        mergedChars.push({
                            ...char,
                            session_ids: [resultData.session_id],
                        });
                    }
                });

                // Send PATCH request with updated characters
                try {
                    const patchRes = await fetch(
                        `${baseUrl}/campaigns/${campaignId}`,
                        {
                            method: "PATCH",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                                characters: mergedChars,
                                session_ids: [resultData.session_id],
                            }),
                        }
                    );

                    if (!patchRes.ok) {
                        const errText = await patchRes.text();
                        console.error("Failed to update campaign:", errText);
                        setUploadError("Failed to update campaign characters.");
                    } else {
                        setUploadStatus(
                            "Session saved and campaign characters updated successfully!"
                        );
                    }
                } catch (patchErr: any) {
                    console.error("Error patching campaign:", patchErr);
                    setUploadError(
                        patchErr.message || "Failed to update campaign"
                    );
                }
            }

            setIsUploading(false);
            setTimeout(() => setUploadStatus(""), 3000);
        } catch (err: any) {
            console.error(err);
            setUploadError(err.message || "Failed to process session");
            setIsUploading(false);
        }
    };

    const isProcessing = isUploading || isRecording;

    return (
        <div className="flex flex-col items-center justify-center min-h-[80vh] gap-8 p-8 select-none">
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
            <div className="text-center flex flex-col items-center gap-4">
                {isRecording && (
                    <>
                        {/* Timer */}
                        <div className="text-4xl font-mono font-bold text-gray-700 mb-2">
                            {formatTime(recordingTime)}
                        </div>

                        {/* Status */}
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
                    </>
                )}

                {/* SVG — always visible, only pulses while recording and not paused */}
                <svg
                    width="400px"
                    height="250px"
                    viewBox="0 0 512 512"
                    xmlns="http://www.w3.org/2000/svg"
                    className={`w-1/2 text-cyan-800 transition-opacity ${
                        isRecording && !isPaused ? "animate-pulse" : ""
                    }`}
                >
                    <path
                        fill="#361947ff"
                        d="M248 20.3L72.33 132.6 248 128.8zm16 0v108.5l175.7 3.8zm51.4 58.9c6.1 3.5 8.2 7.2 15.1 4.2 10.7.8 22.3 5.8 27.6 15.7 4.7 4.5 1.5 12.6-5.2 12.6-9.7.1-19.7-6.1-14.6-8.3 4.7-2 14.7.9 10-5.5-3.6-4.5-11-7.8-16.3-5.9-1.6 6.8-9.4 4-12-.7-2.3-5.8-9.1-8.2-15-7.9-6.1 2.7 1.6 8.8 5.3 9.9 7.9 2.2.2 7.5-4.1 5.1-4.2-2.4-15-9.6-13.5-18.3 5.8-7.39 15.8-4.62 22.7-.9zm-108.5-3.5c5.5.5 12.3 3 10.2 9.9-4.3 7-9.8 13.1-18.1 14.8-6.5 3.4-14.9 4.4-21.6 1.9-3.7-2.3-13.5-9.3-14.9-3.4-2.1 14.8.7 13.1-11.1 17.8V92.3c9.9-3.9 21.1-4.5 30.3 1.3 8 4.2 19.4 1.5 24.2-5.7 1.4-6.5-8.1-4.6-12.2-3.4-2.7-8.2 7.9-7.5 13.2-8.8zm35 69.2L55.39 149l71.21 192.9zm28.2 0l115.3 197L456.6 149zm-14.1 7.5L138.9 352.6h234.2zm133.3 21.1c13.9 8.3 21.5 26.2 22.1 43-1.3 13.6-.7 19.8-15.2 21.4-14.5 1.6-23.9-19.2-29.7-32.6-3.4-9.9-5.8-24 1.7-31.3 6.1-4.8 15-4.1 21.1-.5zm-223.7 16.1c2.1 4-.5 11.4-4.8 12.1-4.9.7-3.8-9.3-9.4-11.6-6.9-2.3-13.6 5.6-15 11.6 10.4-4 20.3 7.1 20.3 17-.4 11.7-7.9 24.8-19.7 28.1h-5.6c-12.7-.7-18.3-15.8-14.2-26.6 4.4-15.8 10.8-33.9 27.2-40.6 8.5-3.9 19 3.2 21.2 10zm213.9-8.4c-7.1-.1-4.4 10-3.3 14.5 3.5 11.5 7.3 26.6 18.9 30 6.8-1.2 4.4-12.8 3.7-16.5-4.7-10.9-7.1-23.3-19.3-28zM52 186v173.2l61.9-5.7zm408 0l-61.9 167.5 61.9 5.7zm-117.9.7l28.5 63.5-10 4.4-20-43.3c-6.1 3-13 8.9-14.6-1.4-1.3-3.9 8.5-5.1 8.1-11.9-.3-6.9 2.2-12.2 8-11.3zm-212 27.4c-2.4 5.1-4.1 10.3-2.7 15.9 1.7 8.8 13.5 6.4 15.6-.8 2.7-5 3.9-11.7-.5-15.7-4.1-3.4-8.9-2.8-12.4.6zm328.4 41.6c-.1 18.6 1.1 39.2-9.7 55.3-.9 1.2-2.2 1.9-3.7 2.5-5.8-4.1-3-11.3 1.2-15.5 1 7.3 5.5-2.9 6.6-5.6 1.3-3.2 3.6-17.7-1-10.2.7 4-6.8 13.1-9.3 8.1-5-14.4 0-30.5 7-43.5 5.7-6.2 9.9 4.4 8.9 8.9zM59.93 245.5c.59.1 1.34 1 2.48 3.6v61.1c-7.3-7-4.47-18-4.45-26.4 0-8.4 1.65-16.3-1.28-23.2-4.62-1.7-5.79-17-3.17-12.7 4.41 4.8 4.66-2.7 6.42-2.4zm178.77 7.6c8.1 4.5 13.8 14.4 10.8 23.6-2.1 15.2-27 21.1-30.4 29.7-1.2 3 25.4 1.6 30.2 1.6.5 4 1.5 10.7-3.8 11.7-14.5-1.2-29.9-.6-45.1-.6.4-11.2 7.4-21.3 17-26.8 6.9-4.9 15.4-9.3 18.1-17.9 1.8-4.5-.6-9.3-4.6-11.5-4.2-2.9-11-2.3-13.2 2.7-2 3.8-4.4 9.1-8.7 9.6-2.9.4-9 .5-7.2-4.9 1.4-5.6 3.4-11.5 8.2-15.2 8.8-6.3 19.9-6.7 28.7-2zm53.3-1.4c6.8 2.2 12 7.9 14.3 14.6 6.1 14.7 5.5 33.1-4.4 45.9-4.5 4.8-10.2 9.1-17 9.1-12.5-.1-22.4-11.1-24.8-22.8-3.1-13.4-1.8-28.7 6.9-39.8 6.8-7.6 16-10.3 25-7zm156.1 8.1c-1.6 5.9-3.3 13.4-.7 19.3 5.1-2 5.4-9.6 6.6-14.5.9-6.1-3.5-12.6-5.9-4.8zm-176.2 21.1c.6 10.5 1.7 22.8 9.7 28.2 4.9 1.8 9.7-2.2 11.1-6.7 1.9-6.3 2.3-12.9 2.4-19.4-.2-7.1-1.5-15-6.7-20.1-12.2-4.4-15.3 10.9-16.5 18zM434 266.8V328l-4.4 6.7v-42.3c-4.6 7.5-9.1 9.1-6.1-.9 6.1-7.1 4.8-17.4 10.5-24.7zM83.85 279c.8 3.6 5.12 17.8 2.04 14.8-1.97-1.3-3.62-4.9-3.41-6.1-1.55-3-2.96-6.1-4.21-9.2-2.95 4-3.96 8.3-3.14 13.4.2-1.6 1.18-2.3 3.39-.7 7.84 12.6 12.17 29.1 7.29 43.5l-2.22 1.1c-10.36-5.8-11.4-19.4-13.43-30-1.55-12.3-.79-24.7 2.3-36.7 5.2-3.8 9.16 5.4 11.39 9.9zm-7.05 20.2c-4.06 4.7-2.26 12.8-.38 18.4 1.11 5.5 6.92 10.2 6.06 1.6.69-11.1-2.33-12.7-5.68-20zm66.4 69.4L256 491.7l112.8-123.1zm-21.4.3l-53.84 4.9 64.24 41.1c-2.6-2.7-4.9-5.7-7.1-8.8-5.2-6.9-10.5-13.6-18.9-16.6-8.75-6.5-4.2-5.3 2.9-2.6-1-1.8-.7-2.6.1-2.6 2.2-.2 8.4 4.2 9.8 6.3l24.7 31.6 65.1 41.7zm268.4 0l-42.4 46.3c6.4-3.1 11.3-8.5 17-12.4 2.4-1.4 3.7-1.9 4.3-1.9 2.1 0-5.4 7.1-7.7 10.3-9.4 9.8-16 23-28.6 29.1l18.9-24.5c-2.3 1.3-6 3.2-8.2 4.1l-40.3 44 74.5-47.6c5.4-6.7 1.9-5.6-5.7-.9l-11.4 6c11.4-13.7 30.8-28.3 40-35.6 9.2-7.3 15.9-9.8 8.2-1.5l-12.6 16c10-7.6.9 3.9-4.5 5.5-.7 1-1.4 2-2.2 2.9l54.5-34.9zM236 385.8v43.4h-13.4v-30c-5-1.4-10.4 1.7-15.3-.3-3.8-2.9 1-6.8 4.5-5.9 3.3-.1 7.6.2 9.3-3.2 4.4-4.5 9.6-4.4 14.9-4zm29 .5c12.1 1.2 24.2.6 36.6.6 1.5 3 .8 7.8-3.3 7.9-7.7.3-21-1.6-25.9.6-8.2 10.5 5.7 3.8 11.4 5.2 7 1.1 15 2.9 19.1 9.2 2.1 3.1 2.7 7.3.7 10.7-5.8 6.8-17 11.5-25.3 10.9-7.3-.6-15.6-1.1-20.6-7.1-6.4-10.6 10.5-6.7 12.2-3.2 6 5.3 20.3 1.9 20.7-4.7.6-4.2-2.1-6.3-6.9-7.8-4.8-1.5-12.6 1-17.3 1.8-4.7.8-9.6.5-9-4.4.8-4.2 2.7-8.1 2.7-12.5.1-3 1.7-7 4.9-7.2zm133.5 5c-.2-.2-7 5.8-9.9 8.1l-15.8 13.1c10.6-6.5 19.3-12 25.7-21.2zm-247 14.2c2.4 0 7.5 4.6 9.4 7l26.1 31.1c-7.7-2.1-13.3-7.1-17.6-13.7-6.5-7.3-11.3-16.6-21.2-19.6-9-5-5.2-6.4 2.1-2.2-.3-1.9.2-2.6 1.2-2.6z"
                    />
                </svg>
            </div>

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
            {/* Status Messages */}
            {uploadStatus &&
                !completedTranscript &&
                (uploadStatus.toLowerCase().includes("processing") ? (
                    // Show your custom Loading component while transcription is processing
                    <Loading />
                ) : (
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
                ))}

            {uploadError && (
                <div className="flex items-center gap-3 p-4 bg-red-50 border-l-4 border-red-500 rounded-lg shadow">
                    <XCircle className="text-red-600" size={20} />
                    <p className="text-red-800 font-medium">{uploadError}</p>
                </div>
            )}

            {/* Modal for Completed Transcript */}
            {completedTranscript && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                    <div className="bg-white w-full max-w-3xl mx-4 rounded-2xl shadow-2xl border border-green-200 overflow-hidden animate-in fade-in duration-300">
                        {/* Header */}
                        <div className="flex items-center justify-between bg-green-100 px-6 py-4 border-b border-green-200">
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
                            <button
                                onClick={() => setCompletedTranscript(null)} // close modal
                                className="text-gray-500 hover:text-gray-700 transition"
                            >
                                ✕
                            </button>
                        </div>

                        {/* Transcript Scroll Area */}
                        <div className="p-6 max-h-[70vh] overflow-y-auto bg-gray-50">
                            <pre className="whitespace-pre-wrap font-sans text-sm text-gray-800">
                                {completedTranscript}
                            </pre>
                        </div>

                        {/* Footer */}
                        <div className="flex justify-end gap-4 px-6 py-4 bg-green-50 border-t border-green-200">
                            <button
                                onClick={() => setCompletedTranscript(null)} // close modal
                                className="px-5 py-2 rounded-lg text-gray-600 bg-white border border-gray-300 hover:bg-gray-100 transition"
                            >
                                Close
                            </button>
                            <button
                                onClick={handleSaveToDatabase}
                                className="flex items-center gap-2 px-6 py-2 bg-indigo-600 text-white font-semibold rounded-lg hover:bg-indigo-700 transition-colors shadow"
                            >
                                Scribe Your Session
                            </button>
                        </div>
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
