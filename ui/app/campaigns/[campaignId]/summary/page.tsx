"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";

interface Campaign {
    campaign_id: string;
    campaign_name: string;
    session_ids?: string[] | null;
    campaign_image_url?: string;
}

export default function CampaignSummaryPage() {
    const { campaignId } = useParams();
    const [campaign, setCampaign] = useState<Campaign | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);

    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

    useEffect(() => {
        async function fetchCampaign() {
            try {
                const res = await fetch(`${baseUrl}/campaigns/${campaignId}`);
                if (!res.ok)
                    throw new Error(`Campaign not found: ${res.status}`);
                const data: Campaign = await res.json();
                setCampaign(data);
            } catch (e: any) {
                setError(e.message);
            } finally {
                setLoading(false);
            }
        }

        if (campaignId) fetchCampaign();
    }, [campaignId]);

    if (loading) {
        return (
            <div className="p-6">
                <p>Loading campaign...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="p-6 text-red-500">
                <h1>Error</h1>
                <p>{error}</p>
            </div>
        );
    }

    if (!campaign) {
        return (
            <div className="p-6">
                <p>Campaign not found.</p>
            </div>
        );
    }

    // ✅ normalize session_ids to always be an array
    const sessions = Array.isArray(campaign.session_ids)
        ? campaign.session_ids
        : [];

    return (
        <div className="p-6 flex flex-col items-center bg-gray-900 text-white min-h-screen">
            <h1 className="text-4xl font-bold mb-4">
                {campaign.campaign_name}
            </h1>
            <p className="mb-2">Campaign ID: {campaign.campaign_id}</p>
            <p className="mb-4">Sessions: {sessions.length}</p>

            {campaign.campaign_image_url && (
                <img
                    src={campaign.campaign_image_url}
                    alt={campaign.campaign_name}
                    className="w-96 h-auto rounded shadow-lg mb-6"
                />
            )}

            <h2 className="text-2xl font-semibold mb-2">Sessions</h2>
            {sessions.length === 0 ? (
                <p className="text-gray-400">No sessions yet.</p>
            ) : (
                <ul className="space-y-1">
                    {sessions.map((id) => (
                        <li
                            key={id}
                            className="bg-gray-800 px-4 py-2 rounded shadow-sm"
                        >
                            {id}
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}
