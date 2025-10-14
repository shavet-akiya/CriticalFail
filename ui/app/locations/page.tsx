"use client";
import { useEffect, useState } from "react";

export type Location = {
    id: string;
    name: string;
};

export default function Locations() {
    const [locations, setLocations] = useState<Location[]>([]);
    const [loading, setLoading] = useState(true);
    const baseUrl = "/api";

    useEffect(() => {
        const fetchLocations = async () => {
            try {
                const res = await fetch(`${baseUrl}/locations`);
                const data = await res.json();

                // Map DB results to Location type
                const mappedLocations = (data.locations || []).map(
                    (loc: any) => ({
                        id: loc.location_id,
                        name: loc.location_name,
                    })
                );

                setLocations(mappedLocations);
            } catch (err) {
                console.error("Failed to fetch locations:", err);
            } finally {
                setLoading(false);
            }
        };

        fetchLocations();
    }, []);

    if (loading) return <div>Loading…</div>;

    return (
        <div className="max-w-3xl mx-auto space-y-4">
            <h1 className="text-2xl font-bold">Locations</h1>
            <ul className="space-y-2">
                {locations.map((loc) => (
                    <li
                        key={loc.id}
                        className="p-4 border rounded hover:bg-gray-100"
                    >
                        {loc.name}
                    </li>
                ))}
            </ul>
        </div>
    );
}
