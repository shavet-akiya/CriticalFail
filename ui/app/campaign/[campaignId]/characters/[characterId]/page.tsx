"use client";

import { useEffect, useState, useRef } from "react";
import { useParams, useRouter } from "next/navigation";
import type { Character } from "@/types/types";
import Loading from "@/components/Loading";

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
        <div className="w-screen h-screen flex flex-col items-center justify-start overflow-auto bg-gray-100 p-8">
            {/* Back Button */}
            <div className="w-full max-w-6xl mb-4">
                <button
                    onClick={() => router.back()}
                    className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600"
                >
                    ← Back
                </button>
            </div>
            <div className="max-w-6xl w-full p-8 bg-white-colour rounded-2xl shadow-lg flex flex-col md:flex-row gap-8">
                {/* Character Image */}
                <div className="flex-shrink-0 w-48 h-48 rounded-xl overflow-hidden border border-gray-300 relative">
                    <img
                        src={
                            form.imageURL
                                ? form.imageURL.startsWith("http")
                                    ? form.imageURL
                                    : `${baseUrl}${form.imageURL}`
                                : "/images/character-placeholder.png"
                        }
                        alt={form.name}
                        className="w-full h-full object-cover"
                    />
                </div>

                {/* Character Info */}
                <div className="flex-1 flex flex-col justify-between">
                    <div>
                        <div className="flex justify-between items-center mb-4">
                            <h1 className="text-3xl font-bold obsidian-colour">
                                {form.name}
                            </h1>
                            <button
                                className="px-3 py-1 bg-purple-colour rounded text-white hover:bg-purple-700"
                                onClick={() => setIsEditing(true)}
                            >
                                Edit
                            </button>
                        </div>

                        {!isEditing ? (
                            <div className="space-y-2 text-gray-700">
                                <p>
                                    <strong>Race:</strong> {form.race}
                                </p>
                                <p>
                                    <strong>Class:</strong> {form.class}
                                </p>
                                <p>
                                    <strong>AC:</strong> {form.AC} |{" "}
                                    <strong>HP:</strong> {form.HP} |{" "}
                                    <strong>NPC:</strong>{" "}
                                    {form.npc ? "Yes" : "No"}
                                </p>
                                <p>
                                    <strong>Stats:</strong> STR {form.STR}, DEX{" "}
                                    {form.DEX}, CON {form.CON}, INT {form.INT},
                                    WIS {form.WIS}, CHA {form.CHA}
                                </p>
                            </div>
                        ) : (
                            <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
                                <div className="flex flex-col gap-4 p-8 max-w-2xl w-full bg-white rounded-xl shadow-lg border-2 border-purple relative">
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

                                    {error && (
                                        <p className="text-red-500">{error}</p>
                                    )}

                                    {/* Editable Fields */}
                                    {/* Editable Fields */}
                                    <div className="grid grid-cols-1 gap-4 mt-4">
                                        {/* Name full width */}
                                        <div>
                                            <label className="block text-lg purple-colour font-semibold mb-1">
                                                Name
                                            </label>
                                            <input
                                                className="border p-2 rounded w-full mb-3 obsidian-colour"
                                                value={form.name}
                                                placeholder="e.g. Aragon"
                                                onChange={(e) =>
                                                    onChange(
                                                        "name",
                                                        e.target.value
                                                    )
                                                }
                                            />
                                        </div>

                                        {/* Race and Class side by side */}
                                        <div className="grid grid-cols-2 gap-4">
                                            <div>
                                                <label className="block text-lg purple-colour font-semibold mb-1">
                                                    Race
                                                </label>
                                                <input
                                                    className="border p-2 rounded w-full mb-3 obsidian-colour"
                                                    value={form.race || ""}
                                                    placeholder="e.g. Half-Elf"
                                                    onChange={(e) =>
                                                        onChange(
                                                            "race",
                                                            e.target.value
                                                        )
                                                    }
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-lg purple-colour font-semibold mb-1">
                                                    Class
                                                </label>
                                                <input
                                                    className="border p-2 rounded w-full mb-3 obsidian-colour"
                                                    placeholder="e.g. Warlock"
                                                    value={form.class || ""}
                                                    onChange={(e) =>
                                                        onChange(
                                                            "class",
                                                            e.target.value
                                                        )
                                                    }
                                                />
                                            </div>
                                        </div>

                                        {/* AC, HP, NPC as before */}
                                        <div className="grid grid-cols-3 gap-4">
                                            <div>
                                                <label className="block text-lg purple-colour font-semibold mb-1">
                                                    AC
                                                </label>
                                                <input
                                                    type="number"
                                                    className="border p-2 rounded w-full mb-3 obsidian-colour"
                                                    value={form.AC}
                                                    onChange={(e) =>
                                                        onChange(
                                                            "AC",
                                                            Number(
                                                                e.target.value
                                                            )
                                                        )
                                                    }
                                                />
                                            </div>
                                            <div>
                                                <label className="block text-lg purple-colour font-semibold mb-1">
                                                    HP
                                                </label>
                                                <input
                                                    type="number"
                                                    className="border p-2 rounded w-full mb-3 obsidian-colour"
                                                    value={form.HP}
                                                    onChange={(e) =>
                                                        onChange(
                                                            "HP",
                                                            Number(
                                                                e.target.value
                                                            )
                                                        )
                                                    }
                                                />
                                            </div>
                                            <div className="flex items-center mt-6">
                                                <label className="flex items-center gap-2 text-lg purple-colour font-semibold">
                                                    NPC
                                                    <input
                                                        type="checkbox"
                                                        checked={!!form.npc}
                                                        className="w-6 h-6 purple-colour"
                                                        onChange={(e) =>
                                                            onChange(
                                                                "npc",
                                                                e.target.checked
                                                            )
                                                        }
                                                    />
                                                </label>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Stats */}
                                    <div className="grid grid-cols-3 gap-4 mt-4">
                                        {[
                                            "STR",
                                            "DEX",
                                            "CON",
                                            "INT",
                                            "WIS",
                                            "CHA",
                                        ].map((stat) => (
                                            <div key={stat}>
                                                <label className="block text-lg purple-colour font-semibold mb-1">
                                                    {stat}
                                                </label>
                                                <input
                                                    type="number"
                                                    className="border p-2 rounded w-full mb-3 obsidian-colour"
                                                    value={
                                                        (form as any)[stat] ?? 0
                                                    }
                                                    onChange={(e) =>
                                                        onChange(
                                                            stat as keyof Character,
                                                            Number(
                                                                e.target.value
                                                            )
                                                        )
                                                    }
                                                />
                                            </div>
                                        ))}
                                    </div>
                                    {/* Image Upload */}
                                    <div className="mt-4 flex flex-col items-center gap-2">
                                        <div
                                            onClick={() =>
                                                fileInputRef.current?.click()
                                            }
                                            className="w-64 h-64 border-2 border-dashed border-gray-700 rounded-xl flex items-center justify-center cursor-pointer overflow-hidden bg-gray-300 relative"
                                        >
                                            {form.imageURL && (
                                                <img
                                                    src={
                                                        form.imageURL.startsWith(
                                                            "http"
                                                        )
                                                            ? form.imageURL
                                                            : `${baseUrl}${form.imageURL}`
                                                    }
                                                    className="w-full h-full object-cover opacity-40"
                                                />
                                            )}
                                            <span
                                                className="absolute text-white text-center px-2"
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
                                                const file =
                                                    e.target.files?.[0];
                                                if (!file) return;

                                                const reader = new FileReader();
                                                reader.onload = (event) => {
                                                    if (event.target?.result) {
                                                        onChange(
                                                            "imageURL",
                                                            event.target
                                                                .result as string
                                                        );
                                                    }
                                                };
                                                reader.readAsDataURL(file);

                                                uploadImage(file);
                                            }}
                                        />
                                    </div>
                                    {/* Buttons */}
                                    <div className="flex justify-between mt-2">
                                        <button
                                            onClick={remove}
                                            disabled={deleting}
                                            className="px-4 py-2 bg-red-600 rounded text-white hover:bg-red-700"
                                        >
                                            {deleting
                                                ? "Deleting..."
                                                : "Delete"}
                                        </button>
                                        <div className="flex gap-2">
                                            <button
                                                onClick={save}
                                                disabled={saving}
                                                className="px-4 py-2 bg-green-600 rounded text-white hover:bg-green-700"
                                            >
                                                {saving ? "Saving..." : "Save"}
                                            </button>
                                            <button
                                                onClick={() =>
                                                    setIsEditing(false)
                                                }
                                                className="px-4 py-2 bg-gray-500 rounded text-white hover:bg-gray-600"
                                            >
                                                Cancel
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
