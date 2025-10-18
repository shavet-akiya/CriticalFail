"use client";

import React, { createContext, useContext, useState, ReactNode } from "react";

interface CampaignContextType {
    campaignID: number | null;
    setCampaignID: (id: number | null) => void;
}

const CampaignContext = createContext<CampaignContextType | undefined>(undefined);

export function CampaignProvider({ children }: { children: ReactNode }) {
    const [campaignID, setCampaignID] = useState<number | null>(null);

    return (
        <CampaignContext.Provider value={{ campaignID, setCampaignID }}>
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
