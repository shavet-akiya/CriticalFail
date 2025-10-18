"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useCampaign } from "@/contexts/CampaignContext";

export default function CamapignCard() {
    const { campaignID, setCampaignID } = useCampaign();
    const router = useRouter();

    useEffect(() => {
        if (campaignID !== null) {
            router.push(`/campaign/${campaignID}/summary`);
        }
    }, [campaignID, router]);


    return (
        <>
            <div className="card lg:card-side shadow-sm">
                <figure>
                    <img
                        src="https://img.daisyui.com/images/stock/photo-1494232410401-ad00d5433cfa.webp"
                        alt="Album" />
                </figure>
                <div className="card-body">
                    <h2 className="card-title">Campaign Title</h2>
                    <p>This is a description - we don't need this necessarily</p>
                    <div className="card-actions justify-end">
                        <button className="btn btn-primary" onClick={() => { setCampaignID(10); router.push(`/campaign/${campaignID}/summary`); }}>Go to campaign</button>
                    </div>
                </div>
            </div>
        </>
    )
}