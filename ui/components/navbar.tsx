"use client";

import Link from "next/link";
import { useRecording } from "@/contexts/RecordingContext";

function NavBar() {
  const { isRecording, isPaused } = useRecording();

  return (
    <div className="fixed top-0 left-0 right-0 z-50 navbar bg-[#0B1215]">
      <div className="flex-1 font-metal-mania">
        <Link href="/" className="text-3xl red-colour">
          Dungeon Scribe
        </Link>
      </div>
      <div className="flex-none">
        <ul className="menu menu-horizontal px-1 gap-4">
          <li>
            <Link href="/session_list" className="white-colour">Session List</Link>
          </li>
          <li>
            <Link href="/summary" className="white-colour">Campaign Summary</Link>
          </li>
          <li>
            <Link href="/timeline" className="white-colour">Event Timeline</Link>
          </li>
          <li>
            <Link href="/characters" className="white-colour">Characters</Link>
          </li>
          <li>
            <Link href="/locations" className="white-colour">Locations</Link>
          </li>
          <li>
            <Link
              href={isRecording || isPaused ? "/new_session/recording" : "/new_session"}
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

export default NavBar;
