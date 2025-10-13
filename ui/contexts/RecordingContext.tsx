"use client";

import { createContext, useContext, useState } from "react";

interface RecordingContextType {
    isRecording: boolean;
    isProcessing: boolean;
    isPaused: boolean;
    startRecording: () => void;
    pauseRecording: () => void;
    resumeRecording: () => void;
    stopRecording: () => void;
    finishProcessing: () => void;
}

const RecordingContext = createContext<RecordingContextType | undefined>(undefined);

export function RecordingProvider({ children }: { children: React.ReactNode }) {
    const [isRecording, setIsRecording] = useState(false);
    const [isPaused, setIsPaused] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);

    const startRecording = () => {
        setIsRecording(true);
        setIsPaused(false);
        setIsProcessing(false);
    };

    const pauseRecording = () => {
        setIsPaused(true);
        setIsRecording(false);
    };

    const resumeRecording = () => {
        setIsPaused(false);
        setIsRecording(true);
    };

    const stopRecording = () => {
        setIsRecording(false);
        setIsPaused(false);
        setIsProcessing(true);

        // simulate processing delay
        setTimeout(() => setIsProcessing(false), 3000);
    };

    const finishProcessing = () => setIsProcessing(false);

    return (
        <RecordingContext.Provider
            value={{
                isRecording,
                isPaused,
                isProcessing,
                startRecording,
                pauseRecording,
                resumeRecording,
                stopRecording,
                finishProcessing,
            }}
        >
            {children}
        </RecordingContext.Provider>
    );
}

export function useRecording() {
    const context = useContext(RecordingContext);
    if (!context) throw new Error("useRecording must be used within RecordingProvider");
    return context;
}
