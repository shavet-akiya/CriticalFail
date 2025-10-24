"use client";

import { useParams } from "next/navigation";
import Link from "next/link";
import type { Location } from "@/types/types";
import { formatSessionDate } from "@/helpers/helper_functions";

export function LocationCard({ location }: { location: Location }) {
    const { campaignId } = useParams<{ campaignId: string }>();

    return (
        <Link
            href={`/campaign/${campaignId}/locations/${location.location_id}`}
            className="block"
        >
            <div className="bg-white rounded-2xl shadow-md hover:shadow-lg hover:scale-105 transition-transform duration-300 cursor-pointer p-6 flex flex-col justify-between gap-3 border border-gray-100">
                <h2 className="text-xl font-bold obsidian-colour">
                    {location.location_name}
                </h2>
                {location.location_description && (
                    <p className="text-gray-600 line-clamp-3">
                        {location.location_description}
                    </p>
                )}

                {/* Session badges - AI was used for mapping between tags and badges*/}
                {location.session_ids && location.session_ids.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-3">
                        {location.session_ids.map((sessionId, idx) => (
                            <span
                                key={idx}
                                className="bg-purple-200 text-purple-800 text-xs font-semibold px-2 py-1 rounded-full"
                            >
                                {formatSessionDate(sessionId)}
                            </span>
                        ))}
                    </div>
                )}
            </div>
        </Link>
    );
}
