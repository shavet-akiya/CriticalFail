"use client";

import { useRouter } from "next/navigation";

interface CampaignCardProps {
    campaignID: string;
    campaignName: string;
    sessionCount?: number;
}

export default function CampaignCard({
    campaignID,
    campaignName,
    sessionCount = 0,
}: CampaignCardProps) {
    const router = useRouter();

    return (
        <div className="card lg:card-side shadow-sm">
            <figure>
                <img
                    src="https://img.daisyui.com/images/stock/photo-1494232410401-ad00d5433cfa.webp"
                    alt="Campaign"
                />
            </figure>
            <div className="card-body">
                <h2 className="card-title">{campaignName}</h2>
                <p>Sessions: {sessionCount}</p>
                <div className="card-actions justify-end">
                    <button
                        className="btn btn-primary"
                        onClick={() =>
                            router.push(`/campaigns/${campaignID}/summary`)
                        }
                    >
                        Go to campaign
                    </button>
                </div>
            </div>
        </div>
    );
}
