"use client";

import { useRouter } from "next/navigation";

interface CampaignCardProps {
    campaignID: string;
    campaignName: string;
    sessionCount?: number;
    imageUrl?: string;
}

export default function CampaignCard({
    campaignID,
    campaignName,
    sessionCount = 0,
    imageUrl,
}: CampaignCardProps) {
    const router = useRouter();

    return (
        <div className="card lg:card-side shadow-sm">
            <figure>
                <img
                    src={imageUrl || "/default-placeholder.png"}
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
                            router.push(`/campaign/${campaignID}/summary`)
                        }
                    >
                        Go to campaign
                    </button>
                </div>
            </div>
        </div>
    );
}
