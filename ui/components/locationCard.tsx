"use client";

import { useParams } from "next/navigation";
import Link from "next/link";
import { Location } from "@/app/campaign/[campaignId]/locations/page"; // or define a shared type

export function LocationCard({ location }: { location: Location }) {
    const { campaignId } = useParams<{ campaignId: string }>();

    return (
        <Link
            href={`/campaign/${campaignId}/locations/${location.location_id}`}
            className="block"
        >
            <div className="card bg-base-100 shadow-sm hover:bg-gray-200 rounded-lg cursor-pointer p-4">
                <h2 className="text-xl font-bold">{location.location_name}</h2>
                {location.description && (
                    <p className="text-gray-600 mt-1">{location.description}</p>
                )}
            </div>
        </Link>
    );
}
