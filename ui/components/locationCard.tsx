"use client";

import { useParams } from "next/navigation";
import Link from "next/link";
import type { Location } from "@/types/types"; 

export function LocationCard({ location }: { location: Location }) {
    const { campaignId } = useParams<{ campaignId: string }>();

    return (
        <Link
            href={`/campaign/${campaignId}/locations/${location.location_id}`}
            className="block"
        >
            <div className="card bg-base-100 shadow-sm hover:bg-gray-200 rounded-lg cursor-pointer p-4">
                <h2 className="text-xl font-bold">{location.location_name}</h2>
                {location.location_description && (
                    <p className="text-gray-600 mt-1">
                        {location.location_description}
                    </p>
                )}
            </div>
        </Link>
    );
}
