"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import type { Location } from "@/types/types";

export default function LocationProfile() {
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
    const { campaignId, locationId } = useParams<{
        campaignId: string;
        locationId: string;
    }>();
    const router = useRouter();

    const [form, setForm] = useState<Location | null>(null);
    const [saving, setSaving] = useState(false);
    const [deleting, setDeleting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [isEditing, setIsEditing] = useState(false);

    useEffect(() => {
        if (!campaignId || !locationId) return;

        fetch(`${baseUrl}/campaign/locations/${campaignId}/${locationId}`)
            .then((res) => {
                if (!res.ok) throw new Error(`Error ${res.status}`);
                return res.json();
            })
            .then((data) => {
                if (!data.location) throw new Error("Location not found");

                setForm({
                    location_id: data.location.location_id,
                    location_name: data.location.location_name,
                    location_description: data.location.location_description,
                    session_ids: data.location.session_ids || [],
                });
            })
            .catch((e) => setError(e instanceof Error ? e.message : String(e)));
    }, [campaignId, locationId]);

    function onChange<K extends keyof Location>(key: K, value: any) {
        if (!form) return;
        setForm((prev) => ({ ...prev!, [key]: value }));
    }

    async function save() {
        if (!form) return;
        setSaving(true);
        setError(null);

        try {
            const res = await fetch(
                `${baseUrl}/locations/${campaignId}/${form.location_id}`,
                {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(form),
                }
            );
            if (!res.ok) throw new Error(`Save failed: ${res.status}`);
            const data = await res.json();
            setForm({
                ...data.location,
                session_ids: data.location.session_ids || [],
            });
            setIsEditing(false);
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setSaving(false);
        }
    }

    async function remove() {
        if (!form) return;
        if (!confirm(`Are you sure you want to delete ${form.location_name}?`))
            return;

        setDeleting(true);
        setError(null);

        try {
            const res = await fetch(
                `${baseUrl}/locations/${campaignId}/${form.location_id}`,
                { method: "DELETE" }
            );
            if (!res.ok) throw new Error(`Delete failed: ${res.status}`);
            router.push(`/campaign/${campaignId}/locations`);
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setDeleting(false);
        }
    }

    if (error) return <div className="text-red-500">{error}</div>;
    if (!form) return <div>Loading…</div>;

    return (
        <div className="w-screen h-screen flex items-center justify-center overflow-hidden bg-gray-100 p-4">
            <div className="max-w-4xl mx-auto p-6 bg-white-colour rounded-xl shadow-md flex flex-col md:flex-row gap-6">
                <div className="flex-1 flex flex-col justify-between">
                    <div>
                        <div className="flex justify-between items-center mb-4">
                            <h1 className="text-3xl font-bold">
                                {form.location_name}
                            </h1>
                            <button
                                className="btn btn-outline btn-sm"
                                onClick={() => setIsEditing(!isEditing)}
                            >
                                {isEditing ? "Cancel" : "Edit"}
                            </button>
                        </div>

                        {!isEditing ? (
                            <div className="space-y-2 text-gray-700">
                                <p>
                                    <strong>Description:</strong>{" "}
                                    {form.location_description}
                                </p>
                                <p>
                                    <strong>Sessions:</strong>{" "}
                                    {(form.session_ids || []).join(", ")}
                                </p>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                <div className="grid grid-cols-1 gap-4">
                                    <div>
                                        <label className="block text-sm font-semibold">
                                            Name
                                        </label>
                                        <input
                                            className="input input-bordered w-full"
                                            value={form.location_name}
                                            onChange={(e) =>
                                                onChange(
                                                    "location_name",
                                                    e.target.value
                                                )
                                            }
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-semibold">
                                            Description
                                        </label>
                                        <textarea
                                            className="input input-bordered w-full"
                                            value={form.location_description}
                                            onChange={(e) =>
                                                onChange(
                                                    "location_description",
                                                    e.target.value
                                                )
                                            }
                                        />
                                    </div>
                                </div>

                                <div className="flex gap-4 mt-4">
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
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
