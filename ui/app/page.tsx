"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import Image from "next/image";
import CampaignCard from "@/components/CampaignCard";
import Loading from "@/components/Loading";
import { useCampaign } from "@/contexts/CampaignContext";

interface Campaign {
    campaign_id: string;
    campaign_name: string;
    session_ids: string[];
    campaign_image_url?: string;
}

function App() {
    const [campaigns, setCampaigns] = useState<Campaign[]>([]);
    const [filteredCampaigns, setFilteredCampaigns] = useState<Campaign[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [searchTerm, setSearchTerm] = useState("");
    const [sortOption, setSortOption] = useState<
        "alphabetical" | "sessions" | "recent" | ""
    >("");
    const router = useRouter();

    const { setSelectedCampaign } = useCampaign();

    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

    // Fetch all campaigns
    useEffect(() => {
        fetchCampaigns();
    }, [baseUrl]);

    // Filter + sort campaigns whenever searchTerm, campaigns, or sortOption changes
    useEffect(() => {
        let filtered = campaigns.filter((c) =>
            c.campaign_name.toLowerCase().includes(searchTerm.toLowerCase())
        );

        // Sort campaigns
        switch (sortOption) {
            case "alphabetical":
                filtered.sort((a, b) =>
                    a.campaign_name.localeCompare(b.campaign_name)
                );
                break;
            case "sessions":
                filtered.sort(
                    (a, b) => b.session_ids.length - a.session_ids.length
                );
                break;
            case "recent":
                // assuming campaign_id order approximates creation date
                filtered.sort((a, b) =>
                    b.campaign_id > a.campaign_id ? 1 : -1
                );
                break;
        }

        setFilteredCampaigns(filtered);
    }, [searchTerm, campaigns, sortOption]);

    const fetchCampaigns = async () => {
        try {
            setLoading(true);
            const res = await fetch(`${baseUrl}/campaign`);
            if (!res.ok) throw new Error(`Failed to fetch campaigns`);
            const data = await res.json();
            setCampaigns(data);
            setFilteredCampaigns(data);
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="h-screen w-full overflow-y-scroll snap-y snap-mandatory scroll-smooth">
            {/* Hero Section */}
            <section className="min-h-screen flex flex-col items-center justify-center bg-white-colour purple-colour snap-start relative sm:gap-16 lg:gap-32 overflow-hidden">
                <Image
                    src="/images/homepage-dice.png"
                    alt="Dice top left"
                    width={400}
                    height={400}
                    className="absolute top-[-120px] left-[-120px] rotate-[-20deg] opacity-90 pointer-events-none select-none z-0"
                />
                <Image
                    src="/images/homepage-dice.png"
                    alt="Dice bottom right"
                    width={400}
                    height={400}
                    className="absolute bottom-[-140px] right-[-140px] rotate-[25deg] opacity-90 pointer-events-none select-none z-0"
                />
                <h1 className="text-9xl font-bold select-none font-metal-mania red-colour text-center text-shadow-lg z-10">
                    Dungeon Scribe
                </h1>
                <div className="flex gap-15">
                    <a
                        href="#campaign_selection"
                        className="btn btn-primary w-60 h-20 py-2 text-center"
                    >
                        Select Campaign
                    </a>
                    <Link
                        href="/new_campaign"
                        className="btn btn-primary w-60 h-20 py-2 text-center"
                    >
                        New Campaign
                    </Link>
                </div>
            </section>

            {/* Campaign Selection Section */}
            <section
                id="campaign_selection"
                className="flex flex-col items-center justify-start bg-[#f5f1ec] snap-start min-h-screen select-none"
            >
                {loading && <Loading />}

                {!loading && (
                    <>
                        {/* Sticky title + search bar */}
                        <div className="w-full max-w-7xl sticky top-0 bg-purple-colour z-20 py-6 px-4 rounded-b-lg shadow-md">
                            {/* Top Bar: Back / Title / New Campaign */}
                            <div className="flex justify-between items-center mb-6">
                                {/* Back Button */}
                                <Link
                                    href="/"
                                    className="btn bg-gray-400 hover:bg-gray-500 w-40 py-2 text-center transition-colors duration-200"
                                >
                                    ← Back
                                </Link>

                                {/* Title */}
                                <h2 className="text-4xl font-bold text-white text-center flex-1">
                                    Campaign Selection
                                </h2>

                                {/* New Campaign Button */}
                                <Link
                                    href="/new_campaign"
                                    className="btn bg-[#a80d18] w-40 py-2 text-center"
                                >
                                    + New Campaign
                                </Link>
                            </div>

                            {/* Search + Sort */}
                            {campaigns.length > 0 && (
                                <div className="flex flex-col sm:flex-row justify-center sm:justify-between items-center gap-4">
                                    <input
                                        type="text"
                                        placeholder="Search Campaigns..."
                                        value={searchTerm}
                                        onChange={(e) =>
                                            setSearchTerm(e.target.value)
                                        }
                                        className="w-full sm:w-1/2 bg-white text-black rounded-md border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-400 focus:border-purple-400 transition shadow-sm"
                                    />

                                    <select
                                        value={sortOption}
                                        onChange={(e) =>
                                            setSortOption(e.target.value as any)
                                        }
                                        className="w-full sm:w-1/4 bg-white text-black rounded-md border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-purple-400 focus:border-purple-400 transition shadow-sm"
                                    >
                                        <option value="" disabled>
                                            Sort By...
                                        </option>
                                        <option value="alphabetical">
                                            Alphabetical
                                        </option>
                                        <option value="sessions">
                                            Number of Sessions
                                        </option>
                                        <option value="recent">
                                            Most Recently Created
                                        </option>
                                    </select>
                                </div>
                            )}
                        </div>

                        {error && <p className="text-red-500 mt-4">{error}</p>}

                        {/* No campaigns */}
                        {!error && filteredCampaigns.length === 0 && (
                            <div className="flex flex-col items-center justify-center flex-1 text-center mt-10">
                                <p className="text-lg mb-2 purple-colour">
                                    You don’t have any campaigns yet!
                                </p>
                                <p className="text-lg purple-colour">
                                    Hit <i>New Campaign</i> to start your
                                    adventure — your world awaits.
                                </p>
                                <Link
                                    href="/new_campaign"
                                    className="btn btn-primary mt-4"
                                >
                                    New Campaign
                                </Link>
                            </div>
                        )}

                        {/* Campaigns grid */}
                        {filteredCampaigns.length > 0 && (
                            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-8 pt-8 w-full max-w-7xl">
                                {filteredCampaigns.map((c) => (
                                    <CampaignCard
                                        key={c.campaign_id}
                                        campaignID={c.campaign_id}
                                        campaignName={c.campaign_name}
                                        sessionCount={c.session_ids.length}
                                        imageUrl={c.campaign_image_url}
                                        onClick={() => setSelectedCampaign(c)}
                                        onDelete={fetchCampaigns}
                                    />
                                ))}
                            </div>
                        )}
                    </>
                )}
            </section>
        </div>
    );
}

export default App;
