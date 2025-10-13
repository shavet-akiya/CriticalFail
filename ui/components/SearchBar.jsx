export default function SearchBar() {
    return (
        <div className="sticky top-[4.5rem] z-40 pb-4 mb-4">
            <label className="input flex items-center gap-2">
                <svg
                    className="h-[1em] opacity-50"
                    xmlns="http://www.w3.org/2000/svg"
                    viewBox="0 0 24 24"
                >
                    <g
                        strokeLinejoin="round"
                        strokeLinecap="round"
                        strokeWidth="2.5"
                        fill="none"
                        stroke="currentColor"
                    >
                        <circle cx="11" cy="11" r="8"></circle>
                        <path d="m21 21-4.3-4.3"></path>
                    </g>
                </svg>
                <input
                    type="search"
                    required
                    placeholder="Search"
                    className="w-full bg-transparent outline-none"
                />
            </label>
        </div>
    );
}
