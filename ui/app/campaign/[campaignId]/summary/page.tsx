"use client";

import { useEffect, useState, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import SessionCard from "@/components/SessionCard";
import Link from "next/link";
import Loading from "@/components/Loading";

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
    const fileInputRef = useRef<HTMLInputElement | null>(null);

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
                // Fetch campaign
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

                // Fetch all sessions
                let sessionData: Session[] = [];
                if (data.session_ids && data.session_ids.length > 0) {
                    const sessionPromises = data.session_ids.map(async (id) => {
                        const res = await fetch(`${baseUrl}/sessions/${id}`);
                        if (!res.ok) throw new Error(`Session ${id} not found`);
                        return res.json() as Promise<Session>;
                    });
                    sessionData = await Promise.all(sessionPromises);

                    // Sort by date ascending
                    sessionData.sort((a, b) => {
                        const dateA = a.processed_at
                            ? new Date(a.processed_at).getTime()
                            : 0;
                        const dateB = b.processed_at
                            ? new Date(b.processed_at).getTime()
                            : 0;
                        return dateA - dateB;
                    });

                    // Show latest session
                    const mostRecent = sessionData[sessionData.length - 1];
                    if (mostRecent) setSessions([mostRecent]);
                }

                // Fetch all characters
                const resChars = await fetch(
                    `${baseUrl}/characters/${campaignId}`
                );
                if (!resChars.ok)
                    throw new Error(`Characters not found: ${resChars.status}`);
                const charJson = await resChars.json();
                const charData: Character[] = charJson.characters || [];
                setCharacters(charData);

                // Fetch all locations
                const resLocs = await fetch(
                    `${baseUrl}/campaign/locations/${campaignId}`
                );
                if (!resLocs.ok) throw new Error("Locations not found");
                const locJson = await resLocs.json();
                const locData: Location[] = locJson.locations || [];

                // Optionally link session info to location
                const linkedLocs = locData.map((loc) => {
                    const session = sessionData.find(
                        (s) => s.session_id === loc.session_id
                    );
                    return {
                        ...loc,
                        session_name: session?.processed_at || "",
                    }; // or session?.session_name
                });

                setLocations(linkedLocs);
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

    if (loading) return <Loading />;
    if (error) return <div className="p-6 text-red-500">{error}</div>;
    if (!campaign) return <div className="p-6">Campaign not found.</div>;

    return (
        <div className="flex flex-col items-center bg-white-colour h-full w-[80vw] max-w-7xl select-none gap-5 overflow-hidden padding-box">
            <div className="rounded-b-xl lg:rounded-xl w-full max-w-6xl relative flex flex-col md:flex-row items-center md:items-start p-4 bg-purple-colour gap-6">
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
                    className="rounded-lg w-48 h-48 object-contain"
                />

                <div className="flex-1 flex flex-col justify-center">
                    <h1 className="text-4xl font-bold mb-2 mt-2">
                        {campaign.campaign_name}
                    </h1>
                    {campaign.campaign_description && (
                        <p className="text-gray-200 whitespace-pre-line pl-4 border-l-4 border-white italic">
                            {campaign.campaign_description}
                        </p>
                    )}
                </div>
                <button
                    onClick={() => setEditing(true)}
                    className="absolute top-8 right-8 bg-white obsidian-colour font-bold px-4 py-2 rounded shadow hover:bg-gray-300"
                >
                    Edit
                </button>
            </div>

            {editing && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
                    <div className="flex flex-col gap-2 p-8 max-w-2xl w-full bg-white rounded-xl shadow-lg relative border-2 border-purple">
                        {/* X button */}
                        <button
                            type="button"
                            onClick={() => setEditing(false)}
                            className="absolute top-4 right-4 text-gray-500 hover:text-gray-800 text-3xl font-bold"
                        >
                            ×
                        </button>
                        <h1 className="text-4xl text-center pb-4 obsidian-colour font-bold">
                            Edit Campaign
                        </h1>
                        <label className="block text-lg purple-colour font-semibold mb-1">
                            Campaign Name
                        </label>
                        <p className="text-sm text-gray-900 mb-1">
                            This will rename your campaign.
                        </p>
                        <input
                            type="text"
                            value={newName}
                            onChange={(e) => setNewName(e.target.value)}
                            className="border p-2 rounded w-full mb-3 obsidian-colour"
                            placeholder="e.g. The Wild Beyond Witchlight"
                        />
                        <label className="block text-lg purple-colour font-semibold mb-1">
                            Campaign Description
                        </label>
                        <p className="text-sm text-gray-900 mb-1">
                            Provide a short description of your campaign.
                        </p>
                        <textarea
                            value={newDescription}
                            onChange={(e) => setNewDescription(e.target.value)}
                            className="border p-2 rounded w-full h-24 mb-3 obsidian-colour"
                            placeholder="e.g. My first D&D game."
                        />
                        {/* Upload New Campaign Image */}
                        <label className="block text-lg purple-colour font-semibold mb-1">
                            Upload New Campaign Image
                        </label>
                        <p className="text-sm text-gray-900 mb-1">
                            Choose an image that represents your campaign. Click
                            the box to pick a new image.
                        </p>
                        <div className="flex flex-col gap-2 items-center">
                            {/* Clickable area */}
                            <div
                                onClick={() => fileInputRef.current?.click()}
                                className="w-64 h-64 border-2 border-dashed border-gray-700 rounded-xl flex items-center justify-center cursor-pointer overflow-hidden bg-gray-300 relative"
                            >
                                {/* Old image at opacity */}
                                {campaign?.campaign_image_url &&
                                    !newImageUrl && (
                                        <img
                                            src={
                                                campaign.campaign_image_url.startsWith(
                                                    "http"
                                                )
                                                    ? campaign.campaign_image_url
                                                    : `${baseUrl}${campaign.campaign_image_url}`
                                            }
                                            className="w-full h-full object-cover opacity-40"
                                        />
                                    )}

                                {/* New image preview */}
                                {newImageUrl && (
                                    <img
                                        src={
                                            newImageUrl.startsWith("http")
                                                ? newImageUrl
                                                : `${baseUrl}${newImageUrl}`
                                        }
                                        className="w-full h-full object-cover"
                                    />
                                )}

                                {/* Overlay text */}
                                {!newImageUrl && (
                                    <span
                                        className="absolute white-colour text-center px-2"
                                        style={{
                                            textShadow: `
                                                1px 1px 0 #353434ff,
                                                -1px 1px 0 #353434ff,
                                                1px -1px 0 #353434ff,
                                                -1px -1px 0 #353434ff,
                                                0 1px 0 #353434ff,
                                                0 -1px 0 #353434ff,
                                                1px 0 0 #353434ff,
                                                -1px 0 0 #353434ff
                                                `,
                                        }}
                                    >
                                        Click to pick an image
                                    </span>
                                )}

                                {/* Small Remove X */}
                                {newImageUrl && (
                                    <button
                                        type="button"
                                        onClick={async (e) => {
                                            e.stopPropagation(); // prevent triggering file picker
                                            setNewImageUrl("");
                                            if (fileInputRef.current)
                                                fileInputRef.current.value = "";

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
                                        className="absolute top-2 right-2 w-6 h-6 flex items-center justify-center bg-gray-200 white-colour font-bold rounded-full text-sm hover:bg-gray-400 cursor-pointer"
                                    >
                                        <img
                                            src="/svg/x-circle.svg"
                                            className="w-6 h-6" // adjust size as needed
                                        />
                                    </button>
                                )}
                            </div>

                            {/* Hidden file input */}
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="image/*"
                                className="hidden"
                                onChange={async (e) => {
                                    const file = e.target.files?.[0];
                                    if (!file) return;

                                    // Preview
                                    const reader = new FileReader();
                                    reader.onloadend = () =>
                                        setNewImageUrl(reader.result as string);
                                    reader.readAsDataURL(file);

                                    // Upload to server
                                    const formData = new FormData();
                                    formData.append("campaign_image", file);

                                    try {
                                        const res = await fetch(
                                            `${baseUrl}/campaign/${campaignId}/image`,
                                            { method: "POST", body: formData }
                                        );
                                        const data = await res.json();
                                        if (data.campaign_image_url)
                                            setNewImageUrl(
                                                data.campaign_image_url
                                            );
                                    } catch (err) {
                                        alert("Image upload failed: " + err);
                                    }
                                }}
                            />

                            {/* Hidden file input */}
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept="image/*"
                                className="hidden"
                                onChange={async (e) => {
                                    const file = e.target.files?.[0];
                                    if (!file) return;

                                    // Preview
                                    const reader = new FileReader();
                                    reader.onloadend = () =>
                                        setNewImageUrl(reader.result as string);
                                    reader.readAsDataURL(file);

                                    // Upload to server
                                    const formData = new FormData();
                                    formData.append("campaign_image", file);

                                    try {
                                        const res = await fetch(
                                            `${baseUrl}/campaign/${campaignId}/image`,
                                            { method: "POST", body: formData }
                                        );
                                        const data = await res.json();
                                        if (data.campaign_image_url)
                                            setNewImageUrl(
                                                data.campaign_image_url
                                            );
                                    } catch (err) {
                                        alert("Image upload failed: " + err);
                                    }
                                }}
                            />
                        </div>

                        <div className="flex justify-between mt-2">
                            <button
                                onClick={handleDelete}
                                className="delete-button delete-button:hover"
                            >
                                Delete Campaign
                            </button>

                            <div className="flex gap-2">
                                <button
                                    onClick={handleSave}
                                    disabled={saving}
                                    className="save-button save-button:hover"
                                >
                                    {saving ? "Saving..." : "Save"}
                                </button>
                                <button
                                    onClick={() => setEditing(false)}
                                    className="px-3 py-1 bg-gray-500 rounded hover:bg-gray-600 font-bold cursor-pointer"
                                >
                                    Cancel
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* AI was used for helping generate this carousel*/}
            <div className="flex w-full max-w-6xl border-b border-gray-300">
                <button
                    onClick={() => setActivePage("sessions")}
                    className={`flex-1 py-2 text-center font-semibold transition-colors text-lg ${
                        activePage === "sessions"
                            ? "border-b-4 border-red-600 text-red-600"
                            : "text-gray-600 hover:text-gray-800"
                    }`}
                >
                    Sessions
                </button>
                <button
                    onClick={() => setActivePage("characters")}
                    className={`flex-1 py-2 text-center font-semibold transition-colors text-lg ${
                        activePage === "characters"
                            ? "border-b-4 border-red-600 text-red-600"
                            : "text-gray-600 hover:text-gray-800"
                    }`}
                >
                    Characters
                </button>
                <button
                    onClick={() => setActivePage("locations")}
                    className={`flex-1 py-2 text-center font-semibold transition-colors text-lg ${
                        activePage === "locations"
                            ? "border-b-4 border-red-600 text-red-600"
                            : "text-gray-600 hover:text-gray-800"
                    }`}
                >
                    Locations
                </button>
            </div>

            {/* Carousel Pages */}
            <div className="p-6 flex flex-col rounded-xl w-full max-w-4xl mt-0 h-[50vh] overflow-auto">
                {activePage === "sessions" && (
                    <div className="bg-[#ded4ca] p-6 rounded-xl">
                        <h2 className="page-sub-heading">Latest Session</h2>
                        {sessions.length > 0 ? (
                            <SessionCard
                                session={sessions[0]}
                                formatSessionDate={formatSessionDate}
                            />
                        ) : (
                            <div>
                                <p className="obsidian-colour">
                                    No recent sessions found.
                                </p>
                                <p className="obsidian-colour">
                                    Venture forth! Record a new D&D session and
                                    watch your tale unfold here once chronicled.
                                </p>
                            </div>
                        )}
                    </div>
                )}
                {activePage === "characters" && (
                    <div className="bg-[#ded4ca] p-6 rounded-lg shadow-md hover:shadow-lg transition-all duration-200">
                        <h2 className="page-sub-heading">Characters</h2>
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
                                            className="w-12 h-12 object-cover rounded-full border-2 border-white"
                                        />

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
                    <div className="bg-[#e0d6cb] p-6 rounded-xl shadow-md hover:shadow-lg transition-all duration-200">
                        <h2 className="page-sub-heading">Locations</h2>
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
                                        className="border p-2 rounded bg-white-colour cursor-pointer"
                                        onClick={() =>
                                            router.push(
                                                `/campaign/${campaignId}/locations/${l.location_id}`
                                            )
                                        }
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
