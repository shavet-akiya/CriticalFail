"use client";
import { useEffect, useState } from "react";

type Location = {
    location_id: string;
    name: string;
    description?: string;
};

export default function Locations() {
    const [locations, setLocations] = useState<Location[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch("/api/locations")
            .then((res) => res.json())
            .then((data) => setLocations(data.locations || []))
            .finally(() => setLoading(false));
    }, []);

    if (loading) return <div>Loading…</div>;

    return (
        <div className="max-w-3xl mx-auto space-y-4">
            <h1 className="text-2xl font-bold">Locations</h1>
            <ul className="space-y-2">
                {locations.map((loc) => (
                    <li
                        key={loc.location_id}
                        className="p-4 border rounded hover:bg-gray-100"
                    >
                        <h2 className="font-semibold">{loc.name}</h2>
                        {loc.description && <p>{loc.description}</p>}
                    </li>
                ))}
            </ul>
        </div>
    );
}
