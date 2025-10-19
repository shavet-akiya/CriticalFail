"use client";

import React, { createContext, useContext, useState, useEffect, ReactNode } from "react";
import { useParams } from "next/navigation";

interface Campaign {
    campaign_id: string;
    campaign_name: string;
    campaign_image_url?: string;
    session_ids: string[];
}

interface CampaignContextType {
    selectedCampaign: Campaign | null;
    setSelectedCampaign: (campaign: Campaign | null) => void;
    sessions: any[];
    loading: boolean;
    error: string | null;
}

const CampaignContext = createContext<CampaignContextType | undefined>(undefined);

export function CampaignProvider({ children }: { children: ReactNode }) {
    const [selectedCampaign, setSelectedCampaign] = useState<Campaign | null>(null);
    const [sessions, setSessions] = useState<any[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const { campaignId } = useParams(); // Works for routes like /campaign/[campaignId]/summary

    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

    useEffect(() => {
        async function fetchCampaignAndSessions() {
            if (!campaignId) return;
            try {
                setLoading(true);
                setError(null);

                // Fetch campaign
                const resCampaign = await fetch(`${baseUrl}/campaign/${campaignId}`);
                if (!resCampaign.ok) throw new Error(`Campaign not found`);
                const raw = await resCampaign.json();
                const data: Campaign = Array.isArray(raw) ? raw[0] : raw;

                // Normalize sessions
                const sessionIds = Array.isArray(data.session_ids)
                    ? data.session_ids
                    : typeof data.session_ids === "string"
                        ? JSON.parse(data.session_ids)
                        : [];

                setSelectedCampaign({ ...data, session_ids: sessionIds });

                // Fetch sessions in parallel
                const sessionData = await Promise.all(
                    sessionIds.map(async (id: string) => {
                        const res = await fetch(`${baseUrl}/sessions/${id}`);
                        if (!res.ok) throw new Error(`Session ${id} not found`);
                        return res.json();
                    })
                );

                setSessions(sessionData);
            } catch (e: any) {
                setError(e.message);
            } finally {
                setLoading(false);
            }
        }

        fetchCampaignAndSessions();
    }, [campaignId, baseUrl]);

    return (
        <CampaignContext.Provider
            value={{ selectedCampaign, setSelectedCampaign, sessions, loading, error }}
        >
            {children}
        </CampaignContext.Provider>
    );
}

export function useCampaign() {
    const context = useContext(CampaignContext);
    if (!context) throw new Error("useCampaign must be used within a CampaignProvider");
    return context;
}
