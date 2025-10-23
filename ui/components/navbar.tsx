"use client";

import Link from "next/link";
import { useState } from "react";
import { useRecording } from "@/contexts/RecordingContext";
import { useCampaign } from "@/contexts/CampaignContext";

type NavBarProps = {
    found?: boolean;
};

export default function NavBar({ found = true }: NavBarProps) {
    const { isRecording, isPaused } = useRecording();
    const { selectedCampaign } = useCampaign();
    const [menuOpen, setMenuOpen] = useState(false);


    if (!found) {
        // Blank version
        return (
            <div className="fixed top-0 left-0 right-0 z-50 navbar bg-[#0B1215]">
                <div className="flex-1 font-metal-mania">
                    <Link
                        href="/"
                        className="btn btn-ghost text-3xl red-colour"
                    >
                        Dungeon Scribe
                    </Link>
                </div>
            </div>
        );
    }

    return (
        <nav className="fixed top-0 left-0 right-0 z-50 bg-[#0B1215] flex items-center justify-between px-4 py-3">
            {/* Logo */}
            <div className="flex-1 font-metal-mania">
                <Link href="/" className="btn btn-ghost text-3xl red-colour">
                    ↞ Dungeon Scribe
                </Link>
            </div>

            {/* Hamburger Button (visible on small screens) */}
            <button
                className="lg:hidden text-white focus:outline-none"
                onClick={() => setMenuOpen((prev) => !prev)}
                aria-label="Toggle navigation menu"
            >
                <svg
                    className="w-8 h-8"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    viewBox="0 0 24 24"
                >
                    {menuOpen ? (
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M6 18L18 6M6 6l12 12"
                        />
                    ) : (
                        <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M4 6h16M4 12h16M4 18h16"
                        />
                    )}
                </svg>
            </button>

            <ul className="hidden lg:flex menu menu-horizontal px-1 gap-4">
                <li><Link href={`/campaign/${selectedCampaign?.campaign_id}/summary`} className="white-colour">Campaign Summary</Link></li>
                <li><Link href={`/campaign/${selectedCampaign?.campaign_id}/sessions`} className="white-colour">Session Notes</Link></li>
                <li><Link href={`/campaign/${selectedCampaign?.campaign_id}/events`} className="white-colour">Event Timeline</Link></li>
                <li><Link href={`/campaign/${selectedCampaign?.campaign_id}/characters`} className="white-colour">Characters</Link></li>
                <li><Link href={`/campaign/${selectedCampaign?.campaign_id}/locations`} className="white-colour">Locations</Link></li>
                <li>
                    <Link
                        href={
                            isRecording || isPaused
                                ? `/campaign/${selectedCampaign?.campaign_id}/new_session/recording`
                                : `/campaign/${selectedCampaign?.campaign_id}/new_session`
                        }
                        className="border-1 white-colour"
                    >
                        New Session
                    </Link>
                </li>
            </ul>

            {/* Dropdown Menu (mobile) */}
            {menuOpen && (
                <ul className="absolute top-[64px] left-0 right-0 bg-[#0B1215] flex flex-col items-center space-y-4 py-6 border-t border-gray-700 lg:hidden">
                    <li><Link href={`/campaign/${selectedCampaign?.campaign_id}/summary`} className="white-colour" onClick={() => setMenuOpen(false)}>Campaign Summary</Link></li>
                    <li><Link href={`/campaign/${selectedCampaign?.campaign_id}/sessions`} className="white-colour" onClick={() => setMenuOpen(false)}>Session Notes</Link></li>
                    <li><Link href={`/campaign/${selectedCampaign?.campaign_id}/events`} className="white-colour" onClick={() => setMenuOpen(false)}>Event Timeline</Link></li>
                    <li><Link href={`/campaign/${selectedCampaign?.campaign_id}/characters`} className="white-colour" onClick={() => setMenuOpen(false)}>Characters</Link></li>
                    <li><Link href={`/campaign/${selectedCampaign?.campaign_id}/locations`} className="white-colour" onClick={() => setMenuOpen(false)}>Locations</Link></li>
                    <li>
                        <Link
                            href={
                                isRecording || isPaused
                                    ? `/campaign/${selectedCampaign?.campaign_id}/new_session/recording`
                                    : `/campaign/${selectedCampaign?.campaign_id}/new_session`
                            }
                            className="border-1 white-colour"
                            onClick={() => setMenuOpen(false)}
                        >
                            New Session
                        </Link>
                    </li>
                </ul>
            )}
        </nav>
    );
}
