"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import SessionCard from "@/components/SessionCard";
import Link from "next/link";

interface Campaign {
    campaign_id: string;
    campaign_name: string;
    campaign_description?: string;
    session_ids: string[];
    campaign_image_url?: string;
}

interface Character {
    character_id: string;
    name: string;
    class: string;
    race?: string;
    HP: number;
    AC: number;
    STR: number;
    DEX: number;
    CON: number;
    INT: number;
    WIS: number;
    CHA: number;
    npc?: boolean;
    session_id: string;
    campaign_id: string;
    imageURL?: string;
}

interface Location {
    location_id: string;
    location_name: string;
    description: string;
    session_id: string;
}

interface Event {
    event_id: string;
    event: string;
    event_summary: string;
    participants?: string;
    location?: string;
    event_tags?: string;
    session_id: string;
}

interface Session {
    session_id: string;
    processed_at?: string;
    characters?: Character[];
    locations?: Location[];
    events?: Event[];
    campaign_id: string;
}

function formatSessionDate(sessionId: string): string {
    const year = sessionId.substring(0, 4);
    const month = sessionId.substring(4, 6);
    const day = sessionId.substring(6, 8);
    return `${day}/${month}/${year}`;
}

export default function CampaignSummaryPage() {
    const { campaignId } = useParams();
    const router = useRouter();

    const [campaign, setCampaign] = useState<Campaign | null>(null);
    const [sessions, setSessions] = useState<Session[]>([]);
    const [characters, setCharacters] = useState<Character[]>([]);
    const [locations, setLocations] = useState<Location[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);

    const [editing, setEditing] = useState(false);
    const [newName, setNewName] = useState("");
    const [newDescription, setNewDescription] = useState("");
    const [newImageUrl, setNewImageUrl] = useState("");
    const [saving, setSaving] = useState(false);

    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

    useEffect(() => {
        async function fetchData() {
            try {
                const resCampaign = await fetch(
                    `${baseUrl}/campaign/${campaignId}`
                );
                if (!resCampaign.ok)
                    throw new Error(
                        `Campaign not found: ${resCampaign.status}`
                    );
                const rawCampaign = await resCampaign.json();
                const data: Campaign = Array.isArray(rawCampaign)
                    ? rawCampaign[0]
                    : rawCampaign;

                setCampaign(data);
                setNewName(data.campaign_name || "");
                setNewDescription(data.campaign_description || "");
                setNewImageUrl(data.campaign_image_url || "");

                // Fetch sessions
                let sessionData: Session[] = [];
                if (data.session_ids && data.session_ids.length > 0) {
                    const sessionPromises = data.session_ids.map(async (id) => {
                        const res = await fetch(`${baseUrl}/sessions/${id}`);
                        if (!res.ok) throw new Error(`Session ${id} not found`);
                        return res.json() as Promise<Session>;
                    });
                    sessionData = await Promise.all(sessionPromises);
                    sessionData.sort((a, b) => {
                        const dateA = a.processed_at
                            ? new Date(a.processed_at).getTime()
                            : 0;
                        const dateB = b.processed_at
                            ? new Date(b.processed_at).getTime()
                            : 0;
                        return dateA - dateB;
                    });
                    const mostRecent = sessionData.pop();
                    if (mostRecent) {
                        setSessions([mostRecent]);
                        setLocations(mostRecent.locations || []);
                    }
                }

                // Fetch characters
                const resChars = await fetch(
                    `${baseUrl}/characters/${campaignId}`
                );
                if (!resChars.ok)
                    throw new Error(`Characters not found: ${resChars.status}`);
                const charJson = await resChars.json();
                const charData: Character[] = charJson.characters || [];
                setCharacters(charData);
            } catch (e: any) {
                setError(e.message);
            } finally {
                setLoading(false);
            }
        }

        if (campaignId) fetchData();
    }, [campaignId, baseUrl]);

    async function handleSave() {
        if (!campaign) return;
        setSaving(true);
        try {
            const res = await fetch(`${baseUrl}/campaign/${campaignId}`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    campaign_id: campaign.campaign_id,
                    campaign_name: newName,
                    campaign_description: newDescription,
                    campaign_image_url: newImageUrl,
                }),
            });
            if (!res.ok) throw new Error(`Failed to update: ${res.status}`);
            const updated = await res.json();
            setCampaign({ ...campaign, ...updated });
            setEditing(false);
        } catch (e: any) {
            alert(e.message);
        } finally {
            setSaving(false);
        }
    }

    async function handleDelete() {
        if (!campaign) return;
        if (
            !confirm(
                `Are you sure you want to delete "${campaign.campaign_name}"? This cannot be undone.`
            )
        )
            return;

        try {
            const res = await fetch(
                `${baseUrl}/campaign/${campaign.campaign_id}`,
                { method: "DELETE" }
            );
            if (!res.ok) throw new Error(`Failed to delete: ${res.status}`);
            router.push("/");
        } catch (e: any) {
            alert(e.message);
        }
    }

    if (loading)
        return <div className="p-6">Loading campaign and sessions...</div>;
    if (error) return <div className="p-6 text-red-500">{error}</div>;
    if (!campaign) return <div className="p-6">Campaign not found.</div>;

    return (
        <div className="p-6 flex flex-col items-center obsidian-colour min-h-screen w-full select-none gap-8">
            <div className="border-2 border-purple rounded-xl w-full max-w-4xl flex flex-col justify-center items-center p-4">
                <img
                    src={
                        campaign.campaign_image_url?.startsWith("http")
                            ? campaign.campaign_image_url
                            : `${baseUrl}${campaign.campaign_image_url}`
                    }
                    onError={(e) => {
                        (e.target as HTMLImageElement).src =
                            "/images/campaign-placeholder.jpg";
                    }}
                    alt={campaign.campaign_name}
                    className="rounded-lg max-w-xs max-h-48 object-contain"
                />
                <h1 className="text-4xl font-bold text-center mb-2">
                    {campaign.campaign_name}
                </h1>
                {campaign.campaign_description && (
                    <p className="text-center text-gray-200 max-w-2xl mb-4 whitespace-pre-line">
                        {campaign.campaign_description}
                    </p>
                )}
                <div className="flex gap-4 mt-2">
                    <button
                        onClick={() => setEditing(true)}
                        className="btn btn-primary"
                    >
                        Edit Campaign
                    </button>
                    <button onClick={handleDelete} className="btn btn-danger">
                        Delete Campaign
                    </button>
                </div>
            </div>

            {/* Edit Modal */}
            {editing && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
                    <div className="bg-gray-800 text-white rounded-lg shadow-lg p-6 w-11/12 max-w-md">
                        <h2 className="text-xl font-bold mb-4">
                            Edit Campaign
                        </h2>

                        <input
                            type="text"
                            value={newName}
                            onChange={(e) => setNewName(e.target.value)}
                            className="border p-2 rounded w-full mb-3 text-black"
                            placeholder="Campaign Name"
                        />

                        <textarea
                            value={newDescription}
                            onChange={(e) => setNewDescription(e.target.value)}
                            className="border p-2 rounded w-full h-24 mb-3 text-black"
                            placeholder="Campaign Description"
                        />

                        <input
                            type="file"
                            accept="image/*"
                            className="file-input w-full mb-3"
                            onChange={async (e) => {
                                const file = e.target.files?.[0];
                                if (!file) return;

                                // Preview immediately
                                const reader = new FileReader();
                                reader.onloadend = () =>
                                    setNewImageUrl(reader.result as string);
                                reader.readAsDataURL(file);

                                // Upload to backend
                                const formData = new FormData();
                                formData.append("campaign_image", file);

                                try {
                                    const res = await fetch(
                                        `${baseUrl}/campaign/${campaignId}/image`,
                                        { method: "POST", body: formData }
                                    );
                                    const data = await res.json();
                                    if (data.campaign_image_url)
                                        setNewImageUrl(data.campaign_image_url);
                                } catch (err) {
                                    alert("Image upload failed: " + err);
                                }
                            }}
                        />

                        {/* Preview */}
                        {newImageUrl && (
                            <div className="relative mb-3">
                                <img
                                    src={
                                        newImageUrl.startsWith("http")
                                            ? newImageUrl
                                            : `${baseUrl}${newImageUrl}`
                                    }
                                    alt="Campaign Preview"
                                    className="max-h-48 object-contain rounded border border-gray-300"
                                />
                                <button
                                    onClick={async () => {
                                        // Remove image locally
                                        setNewImageUrl("");

                                        // Remove from backend
                                        try {
                                            const res = await fetch(
                                                `${baseUrl}/campaign/${campaignId}/image`,
                                                {
                                                    method: "DELETE",
                                                }
                                            );
                                            if (!res.ok)
                                                throw new Error(
                                                    "Failed to remove image"
                                                );
                                        } catch (err) {
                                            alert(err);
                                        }
                                    }}
                                    className="absolute top-1 right-1 bg-red-600 px-2 py-1 rounded text-white hover:bg-red-700"
                                >
                                    Remove
                                </button>
                            </div>
                        )}

                        <div className="flex justify-end gap-2 mt-2">
                            <button
                                onClick={handleSave}
                                disabled={saving}
                                className="px-3 py-1 bg-green-600 rounded hover:bg-green-700"
                            >
                                {saving ? "Saving..." : "Save"}
                            </button>
                            <button
                                onClick={() => setEditing(false)}
                                className="px-3 py-1 bg-gray-500 rounded hover:bg-gray-600"
                            >
                                Cancel
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Sessions */}
            {sessions.length === 0 ? (
                <div className="flex flex-col justify-center items-center gap-8 pt-16">
                    <p className="text-xl obsidian-colour text-center">
                        No recent sessions! Press New Session to start your
                        session.
                    </p>
                    <button className="btn btn-primary mt-4">
                        <Link href={`/campaign/${campaignId}/new_session`}>
                            New Session
                        </Link>
                    </button>
                </div>
            ) : (
                <>
                    <div className="flex flex-col gap-6 w-full max-w-4xl mt-8">
                        <h2 className="text-2xl font-bold mb-4">
                            Most Recent Session
                        </h2>
                        <SessionCard
                            session={sessions[0]}
                            formatSessionDate={formatSessionDate}
                        />
                    </div>

                    {/* Characters */}
                    <div className="mt-8 w-full max-w-4xl">
                        <h2 className="text-2xl font-bold mb-4">Characters</h2>
                        {characters.length === 0 ? (
                            <p>No characters found.</p>
                        ) : (
                            <ul className="grid grid-cols-2 md:grid-cols-3 gap-4">
                                {characters.map((c) => (
                                    <li
                                        key={c.character_id}
                                        className="border p-2 rounded"
                                    >
                                        <p className="font-semibold">
                                            {c.name}
                                        </p>
                                        <p>
                                            {c.race} {c.class}
                                        </p>
                                    </li>
                                ))}
                            </ul>
                        )}

                        {/* Locations */}
                        <h2 className="text-2xl font-bold mt-8 mb-4">
                            Locations
                        </h2>
                        {locations.length === 0 ? (
                            <p>No locations found.</p>
                        ) : (
                            <ul className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {locations.map((l) => (
                                    <li
                                        key={l.location_id}
                                        className="border p-2 rounded"
                                    >
                                        <p className="font-semibold">
                                            {l.location_name}
                                        </p>
                                        <p>{l.description}</p>
                                    </li>
                                ))}
                            </ul>
                        )}
                    </div>
                </>
            )}
        </div>
    );
}
