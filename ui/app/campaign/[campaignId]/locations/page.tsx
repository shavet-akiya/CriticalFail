"use client";

import { LocationCard } from "@/components/locationCard";
import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import type { Location } from "@/types/types";
import { useCampaign } from "@/contexts/CampaignContext";
import Loading from "@/components/Loading";
import { Search } from "lucide-react";

export default function Locations() {
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
    const { selectedCampaign } = useCampaign();
    const [loading, setLoading] = useState(false);

    const [locations, setLocations] = useState<Location[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [showModal, setShowModal] = useState(false);
    const [searchQuery, setSearchQuery] = useState("");

    const [newLocation, setNewLocation] = useState({
        location_name: "",
        location_description: "",
    });

    const fetchLocations = async (): Promise<Location[]> => {
        if (!selectedCampaign?.campaign_id) return [];
        const res = await fetch(
            `${baseUrl}/campaign/locations/${selectedCampaign?.campaign_id}`,
            {
                cache: "no-store",
            }
        );
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
        if (!selectedCampaign?.campaign_id) return;

        const fetchData = async () => {
            setLoading(true);
            try {
                const locs = await fetchLocations();
                setLocations(locs);
            } catch (e: any) {
                setError(e instanceof Error ? e.message : String(e));
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [selectedCampaign?.campaign_id]);

    const filteredLocations = locations.filter((loc) =>
        loc.location_name.toLowerCase().includes(searchQuery.toLowerCase())
    );

    const handleCreateLocation = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!selectedCampaign?.campaign_id) return;

        try {
            const createRes = await fetch(
                `${baseUrl}/locations/${selectedCampaign?.campaign_id}`,
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        ...newLocation,
                        campaign_id: selectedCampaign?.campaign_id,
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

    return loading ? (
        <Loading />
    ) : (
        <div className="h-screen w-[80vw] select-none gap-5 overflow-hidden padding-box">
            <div className="sticky top-0 z-30 heading-banner obsidian-colour px-8 select-none shadow-md">
                <h1 className="page-heading pb-4">Locations</h1>
                <div className="mb-4 flex flex-col sm:flex-row gap-4 px-2">
                    <div className="relative flex-1">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 w-5 h-5 pointer-events-none" />
                        <input
                            type="text"
                            placeholder="Search by name..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="border bg-white border-gray-300 pl-10 pr-3 py-2 rounded-lg w-full obsidian-colour focus:outline-none focus:ring-2 focus:ring-purple-400 focus:border-purple-400 transition shadow-sm"
                        />
                    </div>

                    <button
                        onClick={() => setShowModal(true)}
                        className="bg-[#a80d18] white-colour px-4 py-2 rounded-lg font-semibold hover:bg-purple-700 transition shadow-md w-full sm:w-auto"
                    >
                        + Add Location
                    </button>
                </div>
            </div>

            <div className="flex-1 overflow-auto min-h-0">
                {locations.length === 0 && (
                    <div className="text-gray-500 text-center mt-16">
                        <p className="mb-2">No locations available.</p>
                        <p>Create a location to get started.</p>
                    </div>
                )}

                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                    {filteredLocations.map((loc) => (
                        <LocationCard key={loc.location_id} location={loc} />
                    ))}
                </div>
            </div>

            {showModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-purple-colour bg-opacity-50">
                    <div className="flex flex-col gap-2 p-8 max-w-2xl w-full bg-white rounded-xl shadow-lg relative border-2 border-purple">
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
                                    placeholder="e.g. A bustling port city of trade and intrigue."
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
