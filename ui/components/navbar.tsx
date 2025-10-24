"use client";

import Link from "next/link";
import { useState } from "react";
import { useRecording } from "@/contexts/RecordingContext";
import { useCampaign } from "@/contexts/CampaignContext";
import { usePathname } from "next/navigation";

type NavBarProps = {
    found?: boolean;
};

export default function NavBar({ found = true }: NavBarProps) {
    const { isRecording, isPaused } = useRecording();
    const { selectedCampaign } = useCampaign();
    const [menuOpen, setMenuOpen] = useState(false);
    const pathname = usePathname();

    if (!found) {
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

    const menuItems = [
        {
            label: "Campaign Summary",
            href: `/campaign/${selectedCampaign?.campaign_id}/summary`,
        },
        {
            label: "Session Notes",
            href: `/campaign/${selectedCampaign?.campaign_id}/sessions`,
        },
        {
            label: "Event Timeline",
            href: `/campaign/${selectedCampaign?.campaign_id}/events`,
        },
        {
            label: "Characters",
            href: `/campaign/${selectedCampaign?.campaign_id}/characters`,
        },
        {
            label: "Locations",
            href: `/campaign/${selectedCampaign?.campaign_id}/locations`,
        },
        {
            label: " + New Session",
            href: `/campaign/${selectedCampaign?.campaign_id}/new_session`,
        },
    ];

    return (
        <nav className="fixed top-0 left-0 right-0 z-50 bg-[#0B1215] flex items-center justify-between px-4 py-3">
            {/* Logo */}
            <div className="flex-1 font-metal-mania">
                <Link href="/" className="btn btn-ghost text-3xl red-colour">
                    ↞ Dungeon Scribe
                </Link>
            </div>

            {/* Hamburger Button */}
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

            {/* Desktop Menu */}
            <ul className="hidden lg:flex menu menu-horizontal px-1 gap-4">
                {menuItems.map((item) => {
                    const isActive = pathname === item.href;
                    const isNewSession = item.label === " + New Session";

                    return (
                        <li key={item.href}>
                            <Link
                                href={item.href}
                                className={`px-2 py-1 rounded transition ${isActive
                                    ? "bg-[#a80d18] obsidian-colour font-bold"
                                    : "text-white hover:text-[#a80d18]"
                                    } ${isNewSession ? "outline-2 outline-[#a80d18]" : ""}`}
                            >
                                {item.label}
                            </Link>
                        </li>
                    );
                })}
            </ul>


            {/* Mobile Menu */}
            {menuOpen && (
                <ul className="absolute top-[64px] left-0 right-0 bg-[#0B1215] flex flex-col items-center space-y-4 py-6 border-t border-gray-700 lg:hidden">
                    {menuItems.map((item) => {
                        const isActive = pathname === item.href;
                        const isNewSession = item.label === " + New Session";

                        return (
                            <li key={item.href}>
                                <Link
                                    href={item.href}
                                    className={`px-2 py-1 rounded transition ${isActive ? "bg-[#a80d18] obsidian-colour font-bold" : "text-white hover:text-[#a80d18]"
                                        } ${isNewSession ? "outline-2 outline-[#a80d18]" : ""}`}
                                    onClick={() => setMenuOpen(false)}
                                >
                                    {item.label}
                                </Link>
                            </li>
                        );
                    })}

                </ul>
            )}
        </nav>
    );
}
