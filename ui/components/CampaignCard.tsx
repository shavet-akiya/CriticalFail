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
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

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
                            src={
                                imageUrl
                                    ? `${baseUrl}${imageUrl}`
                                    : "/images/campaign-placeholder.jpg"
                            }
                            alt="Campaign Image"
                            className="rounded-lg max-h-64 object-contain"
                            onError={(e) => {
                                (e.target as HTMLImageElement).src =
                                    "/images/campaign-placeholder.jpg";
                            }}
                        />
                    </figure>
                    <div className="card-actions justify-end">
                        <button
                            className="btn btn-primary"
                            onClick={() => {
                                router.push(`/campaign/${campaignID}/summary`);
                            }}
                        >
                            Continue the story
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
