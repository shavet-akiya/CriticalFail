"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Loading from "@/components/Loading";

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
    const { campaignId, eventId } = useParams<{
        campaignId: string;
        eventId: string;
    }>();
    const router = useRouter();
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;

    const [form, setForm] = useState<Event | null>(null);
    const [saving, setSaving] = useState(false);
    const [deleting, setDeleting] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (!eventId || !campaignId) return;

        const fetchEvent = async () => {
            try {
                const res = await fetch(
                    `${baseUrl}/events/${encodeURIComponent(
                        campaignId
                    )}/${encodeURIComponent(eventId)}`
                );
                const data: { event?: any } = await res.json();

                if (!data.event) throw new Error("Event not found");

                const e = data.event;

                // Normalize participants and tags into string arrays
                const participants: string[] = Array.isArray(e.participants)
                    ? e.participants
                    : typeof e.participants === "string"
                        ? e.participants.split(",").map((p: string) => p.trim())
                        : [];

                const event_tags: string[] = Array.isArray(e.event_tags)
                    ? e.event_tags
                    : typeof e.event_tags === "string"
                        ? e.event_tags.split(",").map((t: string) => t.trim())
                        : [];

                setForm({
                    event_id: e.event_id,
                    session_id: e.session_id,
                    campaign_id: e.campaign_id,
                    timeline_order: e.timeline_order ?? 0,
                    event: e.event ?? "",
                    event_summary: e.event_summary ?? "",
                    participants,
                    event_tags,
                    location: e.location ?? "",
                    type: e.type ?? "event",
                });
            } catch (err) {
                console.error(err);
                setError(err instanceof Error ? err.message : String(err));
            }
        };

        fetchEvent();
    }, [eventId, campaignId, baseUrl]);

    if (!form) return <Loading />;
    if (error) return <div className="text-error">{error}</div>;

    // Form helpers
    const onChange = <K extends keyof Event>(key: K, value: any) => {
        setForm({ ...form, [key]: value });
    };

    const addParticipant = () =>
        setForm({ ...form, participants: [...(form.participants || []), ""] });
    const updateParticipant = (i: number, val: string) => {
        const updated = [...(form.participants || [])];
        updated[i] = val;
        setForm({ ...form, participants: updated });
    };
    const removeParticipant = (i: number) => {
        const updated = [...(form.participants || [])];
        updated.splice(i, 1);
        setForm({ ...form, participants: updated });
    };

    const toggleTag = (tag: string) => {
        const updatedTags = form.event_tags ? [...form.event_tags] : [];
        if (updatedTags.includes(tag))
            updatedTags.splice(updatedTags.indexOf(tag), 1);
        else updatedTags.push(tag);
        setForm({ ...form, event_tags: updatedTags });
    };

    // Save event
    const save = async () => {
        setSaving(true);
        setError(null);
        try {
            const payload = {
                ...form,
                participants: (form.participants || []).join(", "),
                event_tags: (form.event_tags || []).join(", "),
            };
            const res = await fetch(
                `${baseUrl}/events/${encodeURIComponent(
                    campaignId
                )}/${encodeURIComponent(form.event_id)}`,
                {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                }
            );
            if (!res.ok) throw new Error(`Save failed: ${res.status}`);
            const data = await res.json();
            setForm({
                ...data.event,
                participants: (data.event.participants || "")
                    .split(",")
                    .map((p: string) => p.trim()),
                event_tags: (data.event.event_tags || "")
                    .split(",")
                    .map((t: string) => t.trim()),
            });
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setSaving(false);
        }
    };

    // Delete event
    const remove = async () => {
        if (!confirm("Are you sure?")) return;
        setDeleting(true);
        setError(null);
        try {
            const res = await fetch(
                `${baseUrl}/events/${encodeURIComponent(
                    campaignId
                )}/${encodeURIComponent(form.event_id)}`,
                { method: "DELETE" }
            );
            if (!res.ok) throw new Error(`Delete failed: ${res.status}`);
            router.push(`/campaign/${campaignId}/events`);
        } catch (err: unknown) {
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setDeleting(false);
        }
    };

    return (
        <div className="max-w-2xl space-y-4 obsidian-colour select-none pb-8 pt-8">
            <h1 className="text-center text-4xl pb-4 obsidian-colour">{form.event}</h1>

            <div>
                <label className="form-field">Event</label>
                <input
                    className="input input-bordered border input-field"

                    value={form.event}
                    onChange={(e) => onChange("event", e.target.value)}
                />
            </div>

            <div>
                <label className="form-field">Summary</label>
                <textarea
                    className="textarea textarea-bordered w-full input-field"
                    value={form.event_summary}
                    onChange={(e) => onChange("event_summary", e.target.value)}
                />
            </div>

            <div>
                <label className="form-field">Participants</label>
                {form.participants?.map((p, i) => (
                    <div key={i} className="flex gap-2 my-1">
                        <input
                            className="input input-bordered flex-1 input-field"
                            value={p}
                            onChange={(e) =>
                                updateParticipant(i, e.target.value)
                            }
                        />
                        <button
                            className="btn btn-warning"
                            onClick={() => removeParticipant(i)}
                        >
                            Remove
                        </button>
                    </div>
                ))}
                <button
                    className="btn btn-sm btn-primary mt-2"
                    onClick={addParticipant}
                >
                    Add Participant
                </button>
            </div>

            <div>
                <label className="form-field">Tags</label>
                <div className="flex flex-wrap gap-2">
                    {TAG_OPTIONS.map((tag) => {
                        const formattedTag = tag.charAt(0).toUpperCase() + tag.slice(1);

                        return (
                            <button
                                key={tag}
                                className={`btn btn-sm ${form.event_tags?.includes(tag) ? "btn-info" : "btn-outline"
                                    }`}
                                onClick={() => toggleTag(tag)}
                            >
                                {formattedTag}
                            </button>
                        );
                    })}

                </div>
            </div>

            <div>
                <label className="form-field">Location</label>
                <input
                    className="input input-bordered w-full input-field"
                    value={form.location || ""}
                    onChange={(e) => onChange("location", e.target.value)}
                />
            </div>

            <div>
                <label className="form-field">Timeline Order</label>
                <input
                    type="number"
                    className="input input-bordered w-full input-field"
                    value={form.timeline_order}
                    onChange={(e) =>
                        onChange("timeline_order", Number(e.target.value))
                    }
                />
            </div>

            <div className="flex gap-4">
                <button
                    className="btn btn-primary"
                    onClick={save}
                    disabled={saving}
                >
                    {saving ? "Saving..." : "Save"}
                </button>
                <button
                    className="btn btn-warning"
                    onClick={remove}
                    disabled={deleting}
                >
                    {deleting ? "Deleting..." : "Delete"}
                </button>
            </div>
        </div>
    );
}
