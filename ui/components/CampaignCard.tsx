"use client";

import { useRouter } from "next/navigation";
import { useCampaign } from "@/contexts/CampaignContext";

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
    const { setCampaignID } = useCampaign();

    return (
        <>
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
                                setCampaignID(campaignID);
                                router.push(`/campaign/${campaignID}/summary`)
                            }
                            }
                        >
                            Continue the story
                        </button>                    </div>
                </div>
            </div>
        </>
    );
}