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
    location_description: string;
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

    const [activePage, setActivePage] = useState<
        "sessions" | "characters" | "locations"
    >("sessions");

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
        <div className="p-6 flex flex-col items-center bg-white-colour min-h-screen w-full select-none gap-8">
            <div className="rounded-xl w-full max-w-4xl relative flex flex-col md:flex-row items-center md:items-start p-4 bg-purple-colour gap-6">
                {/* Campaign Image */}
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
                    className="rounded-lg w-48 h-48 object-contain"
                />

                {/* Name and Description */}
                <div className="flex-1 flex flex-col justify-center">
                    <h1 className="text-4xl font-bold mb-2">
                        {campaign.campaign_name}
                    </h1>
                    {campaign.campaign_description && (
                        <p className="text-gray-200 whitespace-pre-line">
                            {campaign.campaign_description}
                        </p>
                    )}
                </div>
                {/* Edit Button */}
                <button
                    onClick={() => setEditing(true)}
                    className="btn btn-primary absolute top-4 right-4"
                >
                    Edit Campaign
                </button>
            </div>

            {/* Edit Modal */}
            {editing && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-purple-colour bg-opacity-50">
                    <div className="flex flex-col gap-4 p-8 max-w-lg w-full bg-white rounded-xl shadow-lg relative">
                        <h1 className="text-4xl text-center pb-4 obsidian-colour">
                            Edit Campaign
                        </h1>
                        <label className="block text-lg purple-colour font-semibold mb-2">
                            Campaign Name
                        </label>
                        <input
                            type="text"
                            value={newName}
                            onChange={(e) => setNewName(e.target.value)}
                            className="border p-2 rounded w-full mb-3 text-black"
                            placeholder="Campaign Name"
                        />
                        <label className="block text-lg purple-colour font-semibold mb-2">
                            Campaign Description
                        </label>
                        <textarea
                            value={newDescription}
                            onChange={(e) => setNewDescription(e.target.value)}
                            className="border p-2 rounded w-full h-24 mb-3 text-black"
                            placeholder="Campaign Description"
                        />
                        <label className="block text-lg purple-colour font-semibold mb-2">
                            Upload New Campaign Image
                        </label>
                        <input
                            type="file"
                            accept="image/*"
                            className="file-input w-full mb-3"
                            onChange={async (e) => {
                                const file = e.target.files?.[0];
                                if (!file) return;

                                const reader = new FileReader();
                                reader.onloadend = () =>
                                    setNewImageUrl(reader.result as string);
                                reader.readAsDataURL(file);

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
                                        setNewImageUrl("");
                                        try {
                                            const res = await fetch(
                                                `${baseUrl}/campaign/${campaignId}/image`,
                                                { method: "DELETE" }
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

                        <div className="flex justify-between mt-2">
                            <button
                                onClick={handleDelete}
                                className="px-3 py-1 bg-red-600 rounded hover:bg-red-700"
                            >
                                Delete Campaign
                            </button>

                            <div className="flex gap-2">
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
                </div>
            )}

            {/* Carousel Buttons */}
            <div className="flex gap-20 w-full max-w-4xl">
                <button
                    onClick={() => setActivePage("sessions")}
                    className={`flex-1 py-2 rounded ${activePage === "sessions"
                        ? "bg-green-600 text-white"
                        : "bg-gray-300"
                        }`}
                >
                    Sessions
                </button>
                <button
                    onClick={() => setActivePage("characters")}
                    className={`flex-1 py-2 rounded ${activePage === "characters"
                        ? "bg-green-600 text-white"
                        : "bg-gray-300"
                        }`}
                >
                    Characters
                </button>
                <button
                    onClick={() => setActivePage("locations")}
                    className={`flex-1 py-2 rounded ${activePage === "locations"
                        ? "bg-green-600 text-white"
                        : "bg-gray-300"
                        }`}
                >
                    Locations
                </button>
            </div>

            {/* Carousel Pages */}
            <div className="flex flex-col gap-6 w-full max-w-4xl mt-8">
                {activePage === "sessions" && (
                    <div className="bg-[#e0d6cb] p-6 rounded-lg shadow-md hover:shadow-lg transition-all duration-200">
                        <h2 className="text-2xl font-bold mb-4 obsidian-colour">
                            Recent Sessions
                        </h2>
                        {sessions.length > 0 ? (
                            <SessionCard
                                session={sessions[0]}
                                formatSessionDate={formatSessionDate}
                            />
                        ) : (
                            <div>
                                <p className="obsidian-colour text-lg font-medium">
                                    No recent sessions found.
                                </p>
                                <p className="obsidian-colour text-sm">
                                    Complete a new session and it’ll appear here
                                    once saved!
                                </p>
                            </div>
                        )}
                    </div>
                )}
                {activePage === "characters" && (
                    <div className="bg-[#e0d6cb] p-6 rounded-lg shadow-md hover:shadow-lg transition-all duration-200">
                        <h2 className="text-2xl font-bold mb-4 obsidian-colour">
                            Characters
                        </h2>
                        {characters.length === 0 ? (
                            <div>
                                <p className="obsidian-colour">
                                    The tavern is quiet... no adventurers have
                                    gathered yet.
                                </p>
                                <p className="obsidian-colour">
                                    Add a character, and they will appear here!
                                </p>
                            </div>
                        ) : (
                            <ul className="grid grid-cols-2 md:grid-cols-3 gap-4 obsidian-colour">
                                {characters.map((c) => (
                                    <li
                                        key={c.character_id}
                                        className="border p-2 rounded cursor-pointer bg-white-colour flex items-center gap-3 hover:bg-gray-100 transition"
                                        onClick={() =>
                                            router.push(
                                                `/campaign/${campaignId}/characters/${c.character_id}`
                                            )
                                        }
                                    >
                                        {/* Character Image */}
                                        <img
                                            src={
                                                c.imageURL?.startsWith("http")
                                                    ? c.imageURL
                                                    : c.imageURL
                                                        ? `${baseUrl}${c.imageURL}`
                                                        : "/images/character-placeholder.png"
                                            }
                                            onError={(e) => {
                                                (
                                                    e.target as HTMLImageElement
                                                ).src =
                                                    "/images/character-placeholder.png";
                                            }}
                                            alt={c.name}
                                            className="w-12 h-12 object-cover rounded-full border border-gray-300"
                                        />

                                        {/* Character Info */}
                                        <div>
                                            <p className="font-semibold">
                                                {c.name}
                                            </p>
                                            <p className="text-sm text-gray-700">
                                                {c.race} {c.class}
                                            </p>
                                        </div>
                                    </li>
                                ))}
                            </ul>
                        )}
                    </div>
                )}

                {activePage === "locations" && (
                    <div className="bg-[#e0d6cb] p-6 rounded-lg shadow-md hover:shadow-lg transition-all duration-200">
                        <h2 className="text-2xl font-bold mb-4 obsidian-colour">
                            Locations
                        </h2>
                        {locations.length === 0 ? (
                            <div>
                                <p className="obsidian-colour">
                                    The map is empty... no places have been
                                    discovered yet!
                                </p>
                                <p className="obsidian-colour">
                                    Add a location, and it will appear here!
                                </p>
                            </div>
                        ) : (
                            <ul className="grid grid-cols-1 md:grid-cols-2 gap-4 obsidian-colour">
                                {locations.map((l) => (
                                    <li
                                        key={l.location_id}
                                        className="border p-2 rounded bg-white-colour"
                                    >
                                        <p className="font-semibold obsidian-colour">
                                            {l.location_name}
                                        </p>
                                        <p>{l.location_description}</p>
                                    </li>
                                ))}
                            </ul>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
