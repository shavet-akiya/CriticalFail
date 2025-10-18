// src/App.jsx
import React from "react";
import Link from "next/link";

function App() {
    return (
        <div>
            <p>Landing</p>
            <button className="btn btn-primary">
                <Link href="/campaign">Create Campaign</Link>
            </button>
            <button className="btn btn-primary">
                <Link href="/campaign">Select Campaign</Link>
            </button>
        </div>
    );
}

export default App;
