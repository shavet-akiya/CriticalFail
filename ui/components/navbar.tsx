"use client";

import Link from "next/link";
import { useRecording } from "@/contexts/RecordingContext";

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
                        <Link href="/sessions">Session List</Link>
                    </li>
                    <li>
                        <Link href="/summary">Campaign Summary</Link>
                    </li>
                    <li>
                        <Link href="/timeline">Event Timeline</Link>
                    </li>
                    <li>
                        <Link href="/characters">Characters</Link>
                    </li>
                    <li>
                        <Link href="/locations">Locations</Link>
                    </li>
                    <li>
                        <Link
                            href={
                                isRecording || isPaused
                                    ? "/new_session/recording"
                                    : "/new_session"
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
