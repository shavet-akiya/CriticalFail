"use client";

import { useRecording } from "@/contexts/RecordingContext";

export default function RecordingPopup() {
    const {
        isRecording,
        isPaused,
        isProcessing,
        pauseRecording,
        resumeRecording,
        stopRecording,
    } = useRecording();

    if (!isRecording && !isPaused && !isProcessing) return null;

    const bgColor = isProcessing
        ? "bg-blue-600"
        : isPaused
            ? "bg-yellow-600"
            : "bg-red-600";

    return (
        <div
            className={`fixed bottom-6 right-6 ${bgColor} white-colour px-4 py-3 rounded-full shadow-lg flex items-center gap-3 z-50`}
        >
            {isRecording && (
                <>
                    <span className="font-medium">Recording in progress...</span>
                    <button
                        onClick={pauseRecording}
                        className="bg-white text-red-600 px-3 py-1 rounded-md font-semibold hover:bg-gray-100 transition"
                    >
                        Pause
                    </button>
                    <button
                        onClick={stopRecording}
                        className="bg-white text-red-600 px-3 py-1 rounded-md font-semibold hover:bg-gray-100 transition"
                    >
                        Stop
                    </button>
                </>
            )}

            {isPaused && (
                <>
                    <span className="font-medium">Recording paused</span>
                    <button
                        onClick={resumeRecording}
                        className="bg-white text-yellow-600 px-3 py-1 rounded-md font-semibold hover:bg-gray-100 transition"
                    >
                        Resume
                    </button>
                    <button
                        onClick={stopRecording}
                        className="bg-white text-yellow-600 px-3 py-1 rounded-md font-semibold hover:bg-gray-100 transition"
                    >
                        Stop
                    </button>
                </>
            )}

            {isProcessing && (
                <>
                    <span className="font-medium">Processing recording...</span>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                </>
            )}
        </div>
    );
}
