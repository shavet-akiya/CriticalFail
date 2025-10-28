"use client";


// AI was used on this page to refactor the layout of the edit modal to better suit laptop screens
import { useEffect, useState, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import type { Character } from "@/types/types";
import Loading from "@/components/Loading";
import BackButton from "@/components/BackButton";

export default function CharacterProfile() {
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
    const { campaignId, characterId } = useParams<{
        campaignId: string;
        characterId: string;
    }>();
    const router = useRouter();

    const [form, setForm] = useState<Character | null>(null);
    const [saving, setSaving] = useState(false);
    const [deleting, setDeleting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [isEditing, setIsEditing] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (!characterId || !campaignId) return;

        fetch(
            `${baseUrl}/characters/${encodeURIComponent(
                campaignId
            )}/${encodeURIComponent(characterId)}`
        )
            .then((res) => res.json())
            .then((data) => {
                if (!data.character) throw new Error("Character not found");
                setForm({
                    characterId: data.character.character_id,
                    name: data.character.name,
                    race: data.character.race,
                    class: data.character.class,
                    npc: data.character.npc ?? false,
                    AC: data.character.AC ?? 0,
                    HP: data.character.HP ?? 0,
                    STR: data.character.STR ?? 0,
                    DEX: data.character.DEX ?? 0,
                    CON: data.character.CON ?? 0,
                    INT: data.character.INT ?? 0,
                    WIS: data.character.WIS ?? 0,
                    CHA: data.character.CHA ?? 0,
                    imageURL: data.character.imageURL ?? "",
                });
            })
            .catch((e) => setError(e instanceof Error ? e.message : String(e)));
    }, [characterId, campaignId]);

    function onChange<K extends keyof Character>(k: K, v: any) {
        if (!form) return;
        setForm((s) => ({ ...s!, [k]: v }));
    }

    async function save() {
        if (!form) return;
        setSaving(true);
        setError(null);

        try {
            const res = await fetch(
                `${baseUrl}/characters/${campaignId}/${form.characterId}`,
                {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(form),
                }
            );
            if (!res.ok) throw new Error(`Save failed: ${res.status}`);
            const data = await res.json();
            setForm((prev) => ({ ...prev!, ...data.character }));
            setIsEditing(false);
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setSaving(false);
        }
    }

    async function remove() {
        if (!form) return;
        if (!confirm(`Are you sure you want to delete ${form.name}?`)) return;

        setDeleting(true);
        setError(null);

        try {
            const res = await fetch(
                `${baseUrl}/characters/${campaignId}/${form.characterId}`,
                { method: "DELETE" }
            );
            if (!res.ok) throw new Error(`Delete failed: ${res.status}`);
            router.push(`/campaign/${campaignId}/characters`);
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setDeleting(false);
        }
    }

    async function uploadImage(file: File) {
        if (!form) return;
        const formData = new FormData();
        formData.append("character_image", file);

        try {
            const res = await fetch(
                `${baseUrl}/characters/${campaignId}/${form.characterId}/image`,
                { method: "POST", body: formData }
            );
            const data = await res.json();
            if (data.imageURL) onChange("imageURL", data.imageURL);
        } catch (err) {
            alert("Image upload failed: " + err);
        }
    }

    if (error) return <div className="text-red-500">{error}</div>;
    if (!form) return <Loading />;

    return (
        <div className="w-screen h-screen flex flex-col items-center justify-center overflow-hidden bg-gray-100">
            <div className="absolute top-8 left-8">
                <BackButton />
            </div>

            {/* Character Card */}
            <div className="max-w-6xl w-full p-8 bg-white-colour rounded-3xl border-3 border-purple shadow-lg flex flex-col md:flex-row gap-8">
                <div className="flex-shrink-0 w-48 h-48 rounded-xl overflow-hidden relative">
                    <img
                        src={
                            form.imageURL
                                ? form.imageURL.startsWith("http")
                                    ? form.imageURL
                                    : `${baseUrl}${form.imageURL}`
                                : "/images/character-placeholder.png"
                        }
                        alt={form.name}
                        className="w-full rounded-2xl h-full object-cover border-2 border-purple"
                    />
                </div>

                <div className="flex-1 flex flex-col justify-between">
                    <div>
                        <div className="flex justify-between items-center mb-4">
                            <h1 className="text-3xl font-bold obsidian-colour">{form.name}</h1>
                            <button
                                className="btn btn-primary"
                                onClick={() => setIsEditing(true)}
                            >
                                Edit
                            </button>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-gray-700">
                            {/* General Info */}
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                                {/* Race */}
                                <div className="flex flex-col space-y-1">
                                    <span className="font-bold text-purple-colour">Race</span>
                                    <span className="text-lg font-semibold obsidian-colour">{form.race}</span>
                                </div>

                                {/* Class */}
                                <div className="flex flex-col space-y-1">
                                    <span className="font-bold text-purple-colour">Class</span>
                                    <span className="text-lg font-semibold obsidian-colour">{form.class ?? "N/A"}</span>
                                </div>

                                {/* AC */}
                                <div className="flex flex-col space-y-1">
                                    <span className="font-bold text-purple-colour">AC</span>
                                    <span className="text-lg font-semibold obsidian-colour">{form.AC}</span>
                                </div>

                                {/* HP */}
                                <div className="flex flex-col space-y-1">
                                    <span className="font-bold text-purple-colour">HP</span>
                                    <span className="text-lg font-semibold obsidian-colour">{form.HP}</span>
                                </div>

                                {/* NPC */}
                                <div className="flex flex-col space-y-1">
                                    <span className="font-bold text-purple-colour">NPC</span>
                                    <span className="text-lg font-semibold obsidian-colour">{form.npc ? "Yes" : "No"}</span>
                                </div>
                            </div>



                            {/* Stats */}
                            <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                                {[
                                    { label: "STR", value: form.STR },
                                    { label: "DEX", value: form.DEX },
                                    { label: "CON", value: form.CON },
                                    { label: "INT", value: form.INT },
                                    { label: "WIS", value: form.WIS },
                                    { label: "CHA", value: form.CHA },
                                ].map((stat) => (
                                    <div
                                        key={stat.label}
                                        className="flex flex-col items-center justify-center bg-purple/5 border-2 border-purple rounded-xl py-2"
                                    >
                                        <span className="font-bold text-purple-colour">{stat.label}</span>
                                        <span className="text-lg font-semibold obsidian-colour">{stat.value}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                    </div>
                </div>
            </div>

            {/* Edit Modal - Where AI was used for restructuring*/}
            {isEditing && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
                    <div className="flex flex-col gap-4 p-8 max-w-4xl w-full bg-white rounded-xl shadow-lg border-2 border-purple relative">
                        <button
                            type="button"
                            onClick={() => setIsEditing(false)}
                            className="absolute top-4 right-4 text-gray-500 hover:text-gray-800 text-3xl font-bold"
                        >
                            ×
                        </button>

                        <h1 className="text-4xl text-center obsidian-colour font-bold">
                            Edit Character
                        </h1>

                        {error && <p className="text-red-500">{error}</p>}

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
                            <div className="flex flex-col gap-4">
                                <div
                                    onClick={() => fileInputRef.current?.click()}
                                    className="w-full h-64 border-2 border-dashed border-gray-700 rounded-xl flex items-center justify-center cursor-pointer overflow-hidden bg-gray-300 relative"
                                >
                                    {form.imageURL && (
                                        <img
                                            src={
                                                form.imageURL.startsWith("http")
                                                    ? form.imageURL
                                                    : `${baseUrl}${form.imageURL}`
                                            }
                                            className="w-full h-full object-cover opacity-40"
                                        />
                                    )}
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
                                </div>
                                <input
                                    ref={fileInputRef}
                                    type="file"
                                    accept="image/*"
                                    className="hidden"
                                    onChange={(e) => {
                                        const file = e.target.files?.[0];
                                        if (!file) return;

                                        const reader = new FileReader();
                                        reader.onload = (event) => {
                                            if (event.target?.result) {
                                                onChange("imageURL", event.target.result as string);
                                            }
                                        };
                                        reader.readAsDataURL(file);

                                        uploadImage(file);
                                    }}
                                />

                                <div className="flex flex-col gap-3">
                                    {["name", "race", "class"].map((field) => (
                                        <div key={field}>
                                            <label className="form-field capitalize">
                                                {field}
                                            </label>
                                            <input
                                                className="border p-2 rounded w-full obsidian-colour"
                                                value={(form as any)[field] || ""}
                                                onChange={(e) => onChange(field as keyof Character, e.target.value)}
                                                placeholder={`Enter ${field}`}
                                            />
                                        </div>
                                    ))}

                                    <div className="flex items-center gap-2 mt-2">
                                        <input
                                            type="checkbox"
                                            checked={!!form.npc}
                                            onChange={(e) => onChange("npc", e.target.checked)}
                                        />
                                        <label className="form-field">Is NPC?</label>
                                    </div>
                                </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                                {["AC", "HP"].map((stat) => (
                                    <div key={stat}>
                                        <label className="block text-lg purple-colour font-semibold mb-1">{stat}</label>
                                        <input
                                            type="number"
                                            className="border p-2 rounded w-full obsidian-colour"
                                            value={(form as any)[stat]}
                                            onChange={(e) => onChange(stat as keyof Character, Number(e.target.value))}
                                        />
                                    </div>
                                ))}

                                {["STR", "DEX", "CON", "INT", "WIS", "CHA"].map((stat) => (
                                    <div key={stat}>
                                        <label className="block text-lg purple-colour font-semibold mb-1">{stat}</label>
                                        <input
                                            type="number"
                                            className="border p-2 rounded w-full obsidian-colour"
                                            value={(form as any)[stat]}
                                            onChange={(e) => onChange(stat as keyof Character, Number(e.target.value))}
                                        />
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="flex justify-between mt-4">
                            <button
                                onClick={remove}
                                disabled={deleting}
                                className="px-4 py-2 bg-red-600 rounded white-colour hover:bg-red-700"
                            >
                                {deleting ? "Deleting..." : "Delete"}
                            </button>
                            <div className="flex gap-2">
                                <button
                                    onClick={save}
                                    disabled={saving}
                                    className="save-button save-button:hover"
                                >
                                    {saving ? "Saving..." : "Save"}
                                </button>
                                <button
                                    onClick={() => setIsEditing(false)}
                                    className="cancel-button cancel-button:hover"
                                >
                                    Cancel
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
