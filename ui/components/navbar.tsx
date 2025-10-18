"use client";

import Link from "next/link";
import { useRecording } from "@/contexts/RecordingContext";

interface NavBarProps {
    campaignID?: number;
}

// if campaignID: returns a nav bar
// else: returns nothing
function NavBar({ campaignID }: NavBarProps) {
    const { isRecording, isPaused } = useRecording();

    if (campaignID) {
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
                <div className="flex-none">
                    <ul className="menu menu-horizontal px-1 gap-4">
                        <li>
                            <Link
                                href={`/campaign/${campaignID}/campaign`}
                                className="white-colour"
                            >
                                Campaign Summary
                            </Link>
                        </li>
                        <li>
                            <Link
                                href={`/campaign/${campaignID}/sessions`}
                                className="white-colour"
                            >
                                Session List
                            </Link>
                        </li>
                        <li>
                            <Link
                                href={`/campaign/${campaignID}/events`}
                                className="white-colour"
                            >
                                Event Timeline
                            </Link>
                        </li>
                        <li>
                            <Link
                                href={`/campaign/${campaignID}/characters`}
                                className="white-colour"
                            >
                                Characters
                            </Link>
                        </li>
                        <li>
                            <Link
                                href={`/campaign/${campaignID}/locations`}
                                className="white-colour"
                            >
                                Locations
                            </Link>
                        </li>
                        <li>
                            <Link
                                href={
                                    isRecording || isPaused
                                        ? `/campaign/${campaignID}/new_session/recording`
                                        : `/campaign/${campaignID}/new_session`
                                }
                                className="border-1 white-colour"
                            >
                                New Session
                            </Link>
                        </li>
                    </ul>
                </div>
            </div>
        );
    }
    return;
}

export default NavBar;
