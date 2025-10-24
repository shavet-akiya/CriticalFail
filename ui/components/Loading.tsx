"use client";

import { useEffect, useState } from "react";

// AI was used to generate these texts and incorporate them into the loading component
const flavorTexts = [
    "Rolling for initiative...",
    "Consulting the ancient tomes...",
    "Polishing your +1 sword...",
    "Negotiating with the goblins...",
    "Casting Detect Magic...",
    "Sharpening your wit...",
];

export default function Loading() {
    const [currentText, setCurrentText] = useState(0);
    const [fade, setFade] = useState(true);

    useEffect(() => {
        const interval = setInterval(() => {
            setFade(false);
            setTimeout(() => {
                setCurrentText((prev) => (prev + 1) % flavorTexts.length);
                setFade(true);
            }, 500);
        }, 5000);

        return () => clearInterval(interval);
    }, []);

    return (
        <div className="fixed inset-0 bg-black/50  flex items-center justify-center z-50">
            <div className="bg-white-colour rounded-2xl shadow-xl flex flex-col items-center justify-center p-8 w-80 h-96">
                <img
                    src="/images/200.gif"
                    alt="Loading"
                    className="w-64 h-64 mb-3"
                />

                <h1 className="obsidian-colour text-2xl mb-2">Loading...</h1>

                <p
                    className={`text-gray-700 text-center text-lg italic transition-opacity duration-500 ${fade ? "opacity-100" : "opacity-0"
                        }`}
                >
                    {flavorTexts[currentText]}
                </p>
            </div>
        </div>
    );
}
