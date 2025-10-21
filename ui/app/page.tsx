"use client";

import React, { useState, useEffect } from "react";
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
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const { setSelectedCampaign } = useCampaign();

    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

    // Fetch all campaigns
    useEffect(() => {
        async function fetchCampaigns() {
            try {
                setLoading(true);
                const res = await fetch(`${baseUrl}/campaign`);
                if (!res.ok) throw new Error(`Failed to fetch campaigns`);
                const data = await res.json();
                setCampaigns(data);
            } catch (err: unknown) {
                setError(err instanceof Error ? err.message : String(err));
            } finally {
                setLoading(false);
            }
        }

        fetchCampaigns();
    }, [baseUrl]);

    // Delete campaign handler
    const handleDelete = async (campaignId: string) => {
        if (
            !confirm(
                "Are you sure you want to delete this campaign? This will also remove its sessions."
            )
        ) {
            return;
        }

        try {
            const res = await fetch(`${baseUrl}/campaign/${campaignId}`, {
                method: "DELETE",
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.error || "Failed to delete campaign");
            }

            // Remove deleted campaign from local state
            setCampaigns((prev) =>
                prev.filter((c) => c.campaign_id !== campaignId)
            );
        } catch (err: unknown) {
            alert(
                err instanceof Error ? err.message : "Error deleting campaign"
            );
        }
    };

    return (
        <div
            className="
        h-screen w-full overflow-y-scroll 
        snap-y snap-mandatory scroll-smooth
      "
        >
            {/* Hero Section */}
            <section
                className="
          min-h-screen flex flex-col items-center justify-center 
          bg-white-colour purple-colour snap-start relative sm:gap-16 lg:gap-32 overflow-hidden
        "
            >
                {/* Decorative dice images */}
                <Image
                    src="/images/homepage-dice.png"
                    alt="Dice top left"
                    width={400}
                    height={400}
                    className="absolute top-[-120px] left-[-120px] rotate-[-20deg] opacity-90 
                     pointer-events-none select-none z-0"
                />
                <Image
                    src="/images/homepage-dice.png"
                    alt="Dice bottom right"
                    width={400}
                    height={400}
                    className="absolute bottom-[-140px] right-[-140px] rotate-[25deg] opacity-90 
                     pointer-events-none select-none z-0"
                />

                <h1 className="text-9xl font-bold select-none font-metal-mania red-colour text-center text-shadow-lg z-10">
                    Dungeon Scribe
                </h1>
                <div className="bottom-20 text-center flex flex-row gap-8 z-10">
                    <a
                        href="#campaign_selection"
                        className="btn btn-primary scroll-smooth"
                    >
                        Select Campaign
                    </a>
                    <Link href="/new_campaign" className="btn btn-primary">
                        New Campaign
                    </Link>
                </div>
            </section>

            {/* Campaign Selection Section */}
            <section
                id="campaign_selection"
                className="flex flex-col items-center justify-start bg-[#e0d6cb] text-[#3c1642] snap-start min-h-screen select-none gap-16"
            >
                {loading && <Loading />}

                {!loading && (
                    <h2 className="text-4xl font-bold mb-4 pt-8">
                        Campaign Selection
                    </h2>
                )}

                {error && <p className="text-red-500">{error}</p>}
                {!loading && !error && campaigns.length === 0 && (
                    <p>No campaigns available.</p>
                )}

                <div className="grid grid-cols-1 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 pb-16">
                    {campaigns.map((c) => (
                        <div
                            key={c.campaign_id}
                            className="relative flex flex-col items-center"
                        >
                            <CampaignCard
                                campaignID={c.campaign_id}
                                campaignName={c.campaign_name}
                                sessionCount={c.session_ids.length}
                                imageUrl={c.campaign_image_url}
                                onClick={() => setSelectedCampaign(c)}
                            />
                            <button
                                onClick={() => handleDelete(c.campaign_id)}
                                className="mt-3 bg-red-600 hover:bg-red-700 text-white py-2 px-4 rounded-lg transition-colors"
                            >
                                Delete
                            </button>
                        </div>
                    ))}
                </div>
            </section>
        </div>
    );
}

export default App;
