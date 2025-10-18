"use client";

import Link from "next/link";
import { useRecording } from "@/contexts/RecordingContext";

const campaignID = 0; // need to do this dynamically

function NavBar() {
    const { isRecording, isPaused } = useRecording();

    return (
        <div className="fixed top-0 left-0 right-0 z-50 navbar bg-base-100">
            <div className="flex-1 font-metal-mania">
                <Link href="/" className="btn btn-ghost text-3xl">
                    Dungeon Scribe
                </Link>
            </div>
            <div className="flex-none">
                <ul className="menu menu-horizontal px-1 gap-4">
                    <li>
                        <Link href={`/campaign/${campaignID}/summary`}>Campaign Summary</Link>
                    </li>
                    <li>
                        <Link href={`/campaign/${campaignID}/sessions`}>Session List</Link>
                    </li>
                    <li>
                        <Link href={`/campaign/${campaignID}/timeline`}>Event Timeline</Link>
                    </li>
                    <li>
                        <Link href={`/campaign/${campaignID}/characters`}>Characters</Link>
                    </li>
                    <li>
                        <Link href={`/campaign/${campaignID}/locations`}>Locations</Link>
                    </li>
                    <li>
                        <Link
                            href={
                                isRecording || isPaused
                                    ? `/campaign/${campaignID}/new_session/recording`
                                    : `/campaign/${campaignID}/new_session`
                            }
                            className="border-1"
                        >
                            New Session
                        </Link>
                    </li>
                </ul>
            </div>
        </div>
    );
}

export default NavBar;
