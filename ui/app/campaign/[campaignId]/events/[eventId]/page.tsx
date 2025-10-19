"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";

export type Event = {
    event_id: string;
    session_id: string;
    campaign_id?: string;
    timeline_order?: number;
    event?: string;
    event_summary?: string;
    participants?: string[];
    location?: string;
    event_tags?: string[];
    type?: string;
};

const TAG_OPTIONS = [
    "combat",
    "exploration",
    "player-to-player interaction",
    "npc interaction",
    "resting",
    "investigation",
    "miscellaneous",
];

export default function EventDetail() {
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
    const { campaignId, eventId } = useParams<{
        campaignId: string;
        eventId: string;
    }>();
    const router = useRouter();

    const [form, setForm] = useState<Event | null>(null);
    const [saving, setSaving] = useState(false);
    const [deleting, setDeleting] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Fetch event
    useEffect(() => {
        if (!eventId || !campaignId) return;

        fetch(
            `${baseUrl}/sessions/${encodeURIComponent(
                campaignId
            )}/events/${encodeURIComponent(eventId)}`
        )
            .then((res) => res.json())
            .then((data) => {
                if (!data.event) throw new Error("Event not found");
                const e = data.event;

                setForm({
                    event_id: e.event_id,
                    session_id: e.session_id,
                    campaign_id: e.campaign_id,
                    timeline_order: e.timeline_order ?? 0,
                    event: e.event ?? "",
                    event_summary: e.event_summary ?? "",
                    participants: Array.isArray(e.participants)
                        ? e.participants
                        : typeof e.participants === "string"
                        ? e.participants.split(",").map((p: string) => p.trim())
                        : [],
                    location: e.location ?? "",
                    event_tags:
                        typeof e.event_tags === "string"
                            ? e.event_tags
                                  .split(",")
                                  .map((t: string) => t.trim())
                            : Array.isArray(e.event_tags)
                            ? e.event_tags
                            : [],
                    type: e.type ?? "event",
                });
            })
            .catch((e) => {
                console.error(e);
                setError(e instanceof Error ? e.message : String(e));
            });
    }, [eventId, campaignId]);

    // Form handlers
    function onChange<K extends keyof Event>(key: K, value: any) {
        if (!form) return;
        setForm({ ...form, [key]: value });
    }

    // Participants
    const addParticipant = () => {
        if (!form) return;
        setForm({ ...form, participants: [...(form.participants || []), ""] });
    };

    const updateParticipant = (index: number, value: string) => {
        if (!form) return;
        const updated = [...(form.participants || [])];
        updated[index] = value;
        setForm({ ...form, participants: updated });
    };

    const removeParticipant = (index: number) => {
        if (!form) return;
        const updated = [...(form.participants || [])];
        updated.splice(index, 1);
        setForm({ ...form, participants: updated });
    };

    // Tags
    const toggleTag = (tag: string) => {
        if (!form) return;
        const updatedTags = form.event_tags ? [...form.event_tags] : [];
        if (updatedTags.includes(tag)) {
            updatedTags.splice(updatedTags.indexOf(tag), 1);
        } else {
            updatedTags.push(tag);
        }
        setForm({ ...form, event_tags: updatedTags });
    };

    async function save() {
        if (!form) return;
        setSaving(true);
        setError(null);

        try {
            const payload = {
                ...form,
                // convert arrays to comma-separated strings
                participants: (form.participants || []).join(", "),
                event_tags: (form.event_tags || []).join(", "),
            };

            const res = await fetch(
                `${baseUrl}/sessions/${encodeURIComponent(
                    campaignId
                )}/events/${encodeURIComponent(form.event_id)}`,
                {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                }
            );

            if (!res.ok) throw new Error(`Save failed: ${res.status}`);
            const data = await res.json();

            // convert strings back to arrays for UI
            setForm({
                ...data.event,
                participants:
                    typeof data.event.participants === "string"
                        ? data.event.participants
                              .split(",")
                              .map((p: string) => p.trim())
                              .filter(Boolean)
                        : [],
                event_tags:
                    typeof data.event.event_tags === "string"
                        ? data.event.event_tags
                              .split(",")
                              .map((t: string) => t.trim())
                              .filter(Boolean)
                        : [],
            });
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setSaving(false);
        }
    }

    // Delete event
    async function remove() {
        if (!form) return;
        if (!confirm(`Are you sure you want to delete this event?`)) return;

        setDeleting(true);
        setError(null);

        try {
            const res = await fetch(
                `${baseUrl}/sessions/${encodeURIComponent(
                    campaignId
                )}/events/${encodeURIComponent(form.event_id)}`,
                { method: "DELETE" }
            );
            if (!res.ok) throw new Error(`Delete failed: ${res.status}`);
            router.push(`/campaign/${campaignId}/events`);
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setDeleting(false);
        }
    }

    if (error) return <div className="text-error">{error}</div>;
    if (!form) return <div>Loading…</div>;

    return (
        <div className="max-w-2xl space-y-4">
            <h1 className="text-2xl font-bold">{form.event}</h1>

            {/* Event Name */}
            <div>
                <label className="block">Event</label>
                <input
                    className="input input-bordered w-full"
                    value={form.event}
                    onChange={(e) => onChange("event", e.target.value)}
                />
            </div>

            {/* Summary */}
            <div>
                <label className="block">Summary</label>
                <textarea
                    className="textarea textarea-bordered w-full"
                    value={form.event_summary}
                    onChange={(e) => onChange("event_summary", e.target.value)}
                />
            </div>

            {/* Participants */}
            <div>
                <label className="block">Participants</label>
                {form.participants?.map((p, i) => (
                    <div key={i} className="flex gap-2 items-center my-1">
                        <input
                            type="text"
                            className="input input-bordered flex-1"
                            value={p}
                            onChange={(e) =>
                                updateParticipant(i, e.target.value)
                            }
                        />
                        <button
                            type="button"
                            className="btn btn-sm btn-error"
                            onClick={() => removeParticipant(i)}
                        >
                            Remove
                        </button>
                    </div>
                ))}
                <button
                    type="button"
                    className="btn btn-sm btn-primary mt-2"
                    onClick={addParticipant}
                >
                    Add Participant
                </button>
            </div>

            {/* Tags */}
            <div>
                <label className="block mb-1">Tags</label>
                <div className="flex flex-wrap gap-2">
                    {TAG_OPTIONS.map((tag) => (
                        <button
                            key={tag}
                            type="button"
                            className={`btn btn-sm ${
                                form.event_tags?.includes(tag)
                                    ? "btn-primary"
                                    : "btn-outline"
                            }`}
                            onClick={() => toggleTag(tag)}
                        >
                            {tag}
                        </button>
                    ))}
                </div>
            </div>

            {/* Location */}
            <div>
                <label className="block">Location</label>
                <input
                    className="input input-bordered w-full"
                    value={form.location || ""}
                    onChange={(e) => onChange("location", e.target.value)}
                />
            </div>

            {/* Timeline Order */}
            <div>
                <label className="block">Timeline Order</label>
                <input
                    type="number"
                    className="input input-bordered w-full"
                    value={form.timeline_order}
                    onChange={(e) =>
                        onChange("timeline_order", Number(e.target.value))
                    }
                />
            </div>

            <div className="flex gap-4">
                <button
                    onClick={save}
                    className="btn btn-primary"
                    disabled={saving}
                >
                    {saving ? "Saving..." : "Save"}
                </button>
                <button
                    onClick={remove}
                    className="btn btn-error"
                    disabled={deleting}
                >
                    {deleting ? "Deleting..." : "Delete"}
                </button>
            </div>
        </div>
    );
}
