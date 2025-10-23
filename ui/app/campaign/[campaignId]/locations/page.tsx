"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { LocationCard } from "@/components/locationCard";
import Loading from "@/components/Loading";

export type Location = {
    location_id: string;
    location_name: string;
    description?: string;
    session_id?: string;
};

export default function LocationsPage() {
    const { campaignId } = useParams();
    const [locations, setLocations] = useState<Location[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || "";

    useEffect(() => {
        if (!campaignId) return;

        const fetchLocations = async () => {
            try {
                const res = await fetch(`${baseUrl}/locations/${campaignId}`);
                if (!res.ok)
                    throw new Error(`Failed to fetch locations: ${res.status}`);
                const data = await res.json();
                setLocations(data.locations || []);
            } catch (err: any) {
                console.error("Failed to fetch locations:", err);
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchLocations();
    }, [campaignId, baseUrl]);

    if (loading) return <Loading />
    if (error) return <div className="text-red-500">Error: {error}</div>;
    if (locations.length === 0) return <div>No locations found.</div>;

    return (
        <div className="max-w-3xl mx-auto space-y-4 p-4">
            <h1 className="text-2xl font-bold">Locations</h1>
            <div className="space-y-2">
                {locations.map((loc) => (
                    <LocationCard key={loc.location_id} location={loc} />
                ))}
            </div>
        </div>
    );
}
