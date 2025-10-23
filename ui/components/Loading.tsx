"use client";

import { useEffect, useState } from "react";

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
            setFade(false); // start fade out
            setTimeout(() => {
                setCurrentText((prev) => (prev + 1) % flavorTexts.length);
                setFade(true); // fade in new text
            }, 500); // match fade duration
        }, 5000);

        return () => clearInterval(interval);
    }, []);

    return (
        <div className="fixed inset-0 bg-purple-colour flex items-center justify-center z-50">
            {/* --- White Card Container --- */}
            <div className="bg-white rounded-2xl shadow-xl flex flex-col items-center justify-center p-8 w-80 h-96">
                {/* GIF */}
                <img
                    src="/images/200.gif" // replace with your GIF path
                    alt="Loading"
                    className="w-64 h-64 mb-3"
                />

                {/* Loading Text */}
                <h1 className="text-black text-2xl mb-2">Loading...</h1>

                {/* D&D Flavor Text with Fade */}
                <p
                    className={`text-gray-700 text-center text-lg italic transition-opacity duration-500 ${
                        fade ? "opacity-100" : "opacity-0"
                    }`}
                >
                    {flavorTexts[currentText]}
                </p>
            </div>
        </div>
    );
}
