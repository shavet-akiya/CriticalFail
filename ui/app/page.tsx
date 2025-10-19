"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import CampaignCard from "@/components/CampaignCard";
import Loading from "@/components/Loading";

interface Campaign {
    campaign_id: string;
    campaign_name: string;
    session_ids: string[];
    campaign_image_url?: string; // ✅ add this
}

function App() {
    const [campaigns, setCampaigns] = useState<Campaign[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

    useEffect(() => {
        async function fetchCampaigns() {
            try {
                setLoading(true);
                const res = await fetch(`${baseUrl}/campaigns`);
                if (!res.ok) throw new Error(`Failed to fetch campaigns`);
                const data = await res.json();
                setCampaigns(data); // Expecting an array of campaigns
            } catch (err: unknown) {
                setError(err instanceof Error ? err.message : String(err));
            } finally {
                setLoading(false);
            }
        }

        fetchCampaigns();
    }, [baseUrl]);

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
          bg-white-colour purple-colour snap-start relative gap-32
        "
            >
                <h1 className="text-9xl font-bold select-none font-metal-mania red-colour text-center text-shadow-lg text-shadow-gray-300">
                    Dungeon Scribe
                </h1>
                <div className="bottom-20 text-center flex flex-row gap-8">
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
                className="flex flex-col items-center justify-center bg-[#e0d6cb] text-[#3c1642] snap-start min-h-screen"
            >
                <h2 className="text-4xl font-bold mb-4 pt-16">
                    Campaign Selection
                </h2>

                {loading && <Loading />}
                {error && <p className="text-red-500">{error}</p>}
                {!loading && !error && campaigns.length === 0 && (
                    <p>No campaigns available.</p>
                )}

                {!loading &&
                    campaigns.map((c) => (
                        <CampaignCard
                            key={c.campaign_id}
                            campaignID={c.campaign_id}
                            campaignName={c.campaign_name}
                            sessionCount={c.session_ids.length}
                            imageUrl={c.campaign_image_url} // ✅ pass through
                        />
                    ))}
            </section>
        </div>
    );
}

export default App;
