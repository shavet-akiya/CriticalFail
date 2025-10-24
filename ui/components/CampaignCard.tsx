"use client";

import { useRouter } from "next/navigation";
import { deleteCampaign } from "@/helpers/api";

interface CampaignCardProps {
    campaignID: string;
    campaignName: string;
    sessionCount: number;
    imageUrl?: string;
    onClick?: () => void;
    extend?: boolean;
    onDelete?: () => void;
}

export default function CampaignCard({
    campaignID,
    campaignName,
    sessionCount,
    imageUrl,
    onClick,
    extend = false,
    onDelete,
}: CampaignCardProps) {
    const router = useRouter();
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

    return (
        <div
            onClick={() => {
                router.push(`/campaign/${campaignID}/summary`);
            }}
            className="cursor-pointer rounded-lg transition bg-white-colour transform hover:scale-102 
               hover:shadow-lg border border-purple-300"
        >
            <div className="card-body flex flex-col items-center text-center gap-4">
                <div className="card-body">
                    <h2 className="card-title text-2xl purple-colour">
                        {campaignName}
                    </h2>
                    <p className="purple-colour text-lg text-left pb-4">
                        Total Sessions: {sessionCount}
                    </p>
                    <figure>
                        <img
                            src={
                                imageUrl
                                    ? `${baseUrl}${imageUrl}`
                                    : "/images/campaign-placeholder.jpg"
                            }
                            alt="Campaign Image"
                            className="rounded-lg max-h-64 object-contain border-2 border-purple mb-4"
                            onError={(e) => {
                                (e.target as HTMLImageElement).src =
                                    "/images/campaign-placeholder.jpg";
                            }}
                        />
                    </figure>
                    <div className="card-actions justify-center">
                        <button
                            className="btn btn-primary"
                            onClick={() => {
                                router.push(`/campaign/${campaignID}/summary`);
                            }}
                        >
                            Enter Campaign
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
