"use client";

import React, { createContext, useContext, useState, ReactNode } from "react";

interface Campaign {
    campaign_id: string;
    campaign_name: string;
    campaign_image_url?: string;
    session_ids: string[];
}

interface CampaignContextType {
    selectedCampaign: Campaign | null;
    setSelectedCampaign: (campaign: Campaign | null) => void;
}

const CampaignContext = createContext<CampaignContextType | undefined>(undefined);

export function CampaignProvider({ children }: { children: ReactNode }) {
    const [selectedCampaign, setSelectedCampaign] = useState<Campaign | null>(null);

    return (
        <CampaignContext.Provider value={{ selectedCampaign, setSelectedCampaign }}>
            {children}
        </CampaignContext.Provider>
    );
}

export function useCampaign() {
    const context = useContext(CampaignContext);
    if (!context) {
        throw new Error("useCampaign must be used within a CampaignProvider");
    }
    return context;
}
