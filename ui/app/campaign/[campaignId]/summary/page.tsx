"use client";

import { useEffect, useState } from "react";

export default function CampaignSummary() {
    const [campaigns, setCampaigns] = useState<string[]>([]);
    const [error, setError] = useState<string | null>(null);

    const baseUrl = "/api";

    async function fetchCampaignsFromSessions() {
        try {
            const res = await fetch(`${baseUrl}/sessions`, {
                cache: "no-store",
            });
            if (!res.ok) throw new Error(`GET failed: ${res.status}`);
            const data = await res.json();

            const ids: string[] = data.metadatas.map(
                (md: any) => md.campaign_id as string
            );

            const uniqueIds: string[] = Array.from(new Set(ids));
            setCampaigns(uniqueIds);
        } catch (e: any) {
            setError(e.message);
        }
    }

    // ✅ useEffect must be at the top level of the component
    useEffect(() => {
        fetchCampaignsFromSessions();
    }, []);

    return (
        <div className="card bg-gray-900 shadow-xl p-6">
            <h1 className="card-title text-white">Campaign Summary</h1>

            {error && <p className="text-error mt-2">{error}</p>}

            {campaigns.length === 0 ? (
                <p className="text-gray-400 mt-2">No campaigns found.</p>
            ) : (
                <ul className="mt-4 space-y-2">
                    {campaigns.map((id: string) => (
                        <li
                            key={id}
                            className="p-2 bg-gray-800 rounded text-white"
                        >
                            Campaign ID: {id}
                        </li>
                    ))}
                </ul>
            )}
        </div>
    );
}
