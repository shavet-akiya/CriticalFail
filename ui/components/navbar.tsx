import Link from "next/link";

function NavBar() {
  return (
    <div className="fixed navbar bg-base-100 shadow-sm">
      <div className="flex-1">
        <Link href="/" className="btn btn-ghost text-xl">
          Dungeon Scribe
        </Link>
      </div>
      <div className="flex-none">
        <ul className="menu menu-horizontal px-1 gap-4">
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
            <Link href="/new_session" className="border-1">
              New Session
            </Link>
          </li>
        </ul>
      </div>
    </div>
  );
}

export default NavBar;
