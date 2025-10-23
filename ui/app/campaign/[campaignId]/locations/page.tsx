"use client";

import { LocationCard } from "@/components/locationCard";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import type { Location } from "@/types/types";

export default function Locations() {
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
    const { campaignId } = useParams<{ campaignId: string }>();

    const [locations, setLocations] = useState<Location[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [showModal, setShowModal] = useState(false);
    const [searchQuery, setSearchQuery] = useState("");

    const [newLocation, setNewLocation] = useState({
        location_name: "",
        location_description: "",
    });

    // --- Fetch all locations ---
    const fetchLocations = async (): Promise<Location[]> => {
        if (!campaignId) return [];
        const res = await fetch(`${baseUrl}/campaign/locations/${campaignId}`, {
            cache: "no-store",
        });
        if (!res.ok) throw new Error(`GET /locations failed: ${res.status}`);
        const data = await res.json();

        return (data.locations ?? []).map((loc: any) => ({
            location_id: loc.location_id,
            location_name: loc.location_name,
            location_description: loc.location_description,
            session_ids: loc.session_ids || [],
        }));
    };

    useEffect(() => {
        fetchLocations()
            .then(setLocations)
            .catch((e) => setError(e instanceof Error ? e.message : String(e)));
    }, [campaignId]);

    const filteredLocations = locations.filter((loc) =>
        loc.location_name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    // --- Handle location creation ---
    const handleCreateLocation = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!campaignId) return;

        try {
            const createRes = await fetch(
                `${baseUrl}/locations/${campaignId}`,
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        ...newLocation,
                        campaign_id: campaignId,
                    }),
                }
            );
            if (!createRes.ok)
                throw new Error(`Create location failed: ${createRes.status}`);
            const locationData = await createRes.json();
            const createdLocation: Location = locationData.location;

            setLocations((prev) => [...prev, createdLocation]);

            setShowModal(false);
            setNewLocation({ location_name: "", location_description: "" });
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        }
    };

    return (
        <div className="pl-16 pr-16 pt-16 text-black">
            <h1 className="text-3xl font-bold">Locations</h1>

            <div className="mb-8 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <input
                    type="text"
                    placeholder="Search by name..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="border px-3 py-2 rounded-lg w-full sm:w-72"
                />
                <button
                    onClick={() => setShowModal(true)}
                    className="bg-black text-white px-4 py-2 rounded-lg hover:bg-gray-800 transition"
                >
                    + Add Location
                </button>
            </div>

            {locations.length === 0 && (
                <div className="text-gray-500 text-center mt-16">
                    <p className="mb-2">No locations available.</p>
                    <p>Create a location to get started.</p>
                </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-16">
                {filteredLocations.map((loc) => (
                    <LocationCard key={loc.location_id} location={loc} />
                ))}
            </div>

            {showModal && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="bg-white text-black rounded-2xl shadow-xl p-8 w-full max-w-lg overflow-y-auto max-h-[90vh]">
                        <h2 className="text-2xl font-bold mb-4">
                            Create New Location
                        </h2>
                        <form
                            onSubmit={handleCreateLocation}
                            className="space-y-4"
                        >
                            <div>
                                <label className="block text-sm font-semibold mb-1">
                                    Name
                                </label>
                                <input
                                    type="text"
                                    value={newLocation.location_name}
                                    onChange={(e) =>
                                        setNewLocation({
                                            ...newLocation,
                                            location_name: e.target.value,
                                        })
                                    }
                                    className="w-full border rounded-lg px-3 py-2"
                                    required
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-semibold mb-1">
                                    Description
                                </label>
                                <textarea
                                    value={newLocation.location_description}
                                    onChange={(e) =>
                                        setNewLocation({
                                            ...newLocation,
                                            location_description:
                                                e.target.value,
                                        })
                                    }
                                    className="w-full border rounded-lg px-3 py-2"
                                />
                            </div>

                            <div className="flex justify-end gap-4 pt-4">
                                <button
                                    type="button"
                                    onClick={() => setShowModal(false)}
                                    className="px-4 py-2 rounded-lg border border-gray-400 hover:bg-gray-200 transition"
                                >
                                    Cancel
                                </button>
                                <button
                                    type="submit"
                                    className="bg-black text-white px-4 py-2 rounded-lg hover:bg-gray-800 transition"
                                >
                                    Save
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
}
