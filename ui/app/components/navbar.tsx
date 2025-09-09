import Link from "next/link";

function NavBar() {
  return (
    <div className="navbar bg-base-100 shadow-sm">
      <div className="flex-1">
        <Link href="/" className="btn btn-ghost text-xl">
          Dungeon Scribe
        </Link>
      </div>
      <div className="flex-none">
        <ul className="menu menu-horizontal px-1">
          <li>
            <Link href="/summary">Campaign Summary</Link>
          </li>
          <li>
            <Link href="/timeline">Timeline</Link>
          </li>
          <li>
            <a>Characters</a>
          </li>
          <li>
            <details>
              <summary>Start Session</summary>
              <ul className="bg-base-100 rounded-t-none p-2">
                <li>
                  <a>Upload recording</a>
                </li>
                <li>
                  <a className="text-red-400">Start recording</a>
                </li>
              </ul>
            </details>
          </li>
        </ul>
      </div>
    </div>
  );
}

export default NavBar; // Default export
