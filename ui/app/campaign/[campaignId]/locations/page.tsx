"use client";

import { LocationCard } from "@/components/locationCard";
import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import type { Location } from "@/types/types";

export default function Locations() {
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
    const { campaignId } = useParams<{ campaignId: string }>();
    const router = useRouter();

    const [locations, setLocations] = useState<Location[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [showModal, setShowModal] = useState(false);
    const [searchQuery, setSearchQuery] = useState("");

    const [newLocation, setNewLocation] = useState({
        location_name: "",
        location_description: "",
    });

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
        <div className="pl-16 pr-16 text-black h-screen flex flex-col">
            {/* Sticky header + search */}
            <div className="sticky top-0 z-10 bg-white">
                <div className="w-full bg-purple-colour rounded-b-lg shadow-md py-6 px-4 mb-4 text-center">
                    <h1 className="text-4xl font-bold text-white tracking-wide">
                        Locations
                    </h1>
                </div>

                <div className="mb-4 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 px-2">
                    <input
                        type="text"
                        placeholder="Search by name..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="border border-gray-300 px-3 py-2 rounded-lg w-full sm:w-72 text-black focus:outline-none focus:ring-2 focus:ring-purple-400 focus:border-purple-400 transition shadow-sm"
                    />
                    <button
                        onClick={() => setShowModal(true)}
                        className="bg-purple-colour text-white px-4 py-2 rounded-lg font-semibold hover:bg-purple-700 transition shadow-md"
                    >
                        + Add Location
                    </button>
                </div>
            </div>

            {/* Scrollable location cards */}
            <div className="flex-1 overflow-auto">
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
            </div>

            {/* Modal */}
            {showModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-purple-colour">
                    <div className="flex flex-col gap-2 p-8 max-w-2xl w-full bg-white rounded-xl shadow-lg relative border-2 border-purple">
                        {/* X button */}
                        <button
                            type="button"
                            onClick={() => setShowModal(false)}
                            className="absolute top-4 right-4 text-gray-500 hover:text-gray-800 text-3xl font-bold"
                        >
                            ×
                        </button>
                        <h1 className="text-4xl text-center pb-4 obsidian-colour font-bold">
                            Create New Location
                        </h1>
                        <form
                            onSubmit={handleCreateLocation}
                            className="space-y-4"
                        >
                            <div>
                                <label className="block text-lg purple-colour font-semibold mb-1">
                                    Location Name
                                </label>
                                <input
                                    type="text"
                                    value={newLocation.location_name}
                                    placeholder="e.g. Baldur's Gate"
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
                                <label className="block text-lg purple-colour font-semibold mb-1">
                                    Description
                                </label>
                                <textarea
                                    placeholder="e.g. A bustling port city of trade and intrigue, where crowded streets hide both opportunity and danger."
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
                                    className="px-3 py-1 bg-gray-500 rounded hover:bg-gray-600 font-bold cursor-pointer"
                                >
                                    Cancel
                                </button>
                                <button
                                    type="submit"
                                    className="px-3 py-1 bg-green-600 rounded hover:bg-green-700 font-bold cursor-pointer"
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
