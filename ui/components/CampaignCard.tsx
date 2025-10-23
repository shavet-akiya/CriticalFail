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
                    <div className="card-actions justify-end flex flex-row items-center">
                        <button
                            onClick={async (e) => {
                                e.stopPropagation();
                                const success = await deleteCampaign(campaignID);
                                if (success && onDelete) onDelete();
                            }}
                            className="btn btn-warning white-colour"
                        >
                            Delete
                        </button>
                        <button
                            className="btn btn-primary"
                            onClick={() => {
                                router.push(`/campaign/${campaignID}/summary`);
                            }}
                        >
                            Go
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
