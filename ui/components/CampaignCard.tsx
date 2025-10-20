"use client";

import { useRouter } from "next/navigation";
import { useCampaign } from "@/contexts/CampaignContext";

interface CampaignCardProps {
    campaignID: string;
    campaignName: string;
    sessionCount: number;
    imageUrl?: string;
    onClick?: () => void;
    extend?: boolean;
}

export default function CampaignCard({
    campaignID,
    campaignName,
    sessionCount,
    imageUrl,
    onClick,
    extend = false,
}: CampaignCardProps) {
    const router = useRouter();
    const { selectedCampaign, sessions, loading, error } = useCampaign();

    return (
        <div
            onClick={onClick}
            className="cursor-pointer rounded shadow-lg hover:shadow-xl transition"
        >
            <div className="card w-96 shadow-sm">
                <div className="card-body">
                    <h2 className="card-title text-2xl">{campaignName}</h2>
                    <p>Sessions thus far: {sessionCount}</p>
                    <figure>
                        <img
                            src={imageUrl || "/default-placeholder.png"}
                            alt="Campaign Image"
                        />
                    </figure>
                    <div className="card-actions justify-end">
                        <button
                            className="btn btn-primary"
                            onClick={() => {
                                router.push(`/campaign/${campaignID}/summary`)
                            }
                            }
                        >
                            Continue the story
                        </button>                    </div>
                </div>
            </div>

        </div>

    );
}