"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import type { Character } from "@/types/types";

export default function CharacterDetail() {
    const { characterId } = useParams<{ characterId: string }>();
    const baseUrl = "/localhost:9000/";
    const [form, setForm] = useState<Character | null>(null);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);

    // Fetch character
    useEffect(() => {
        if (!characterId) return;
        console.log("Fetching character", characterId);

        fetch(`${baseUrl}/characters/${encodeURIComponent(characterId)}`)
            .then((res) => {
                console.log("Response status", res.status);
                return res.json();
            })
            .then((data) => {
                console.log("API data", data);
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
                });
            })
            .catch((e) => {
                console.error(e);
                setError(e instanceof Error ? e.message : String(e));
            });
    }, [characterId]);

    // Handle changes
    function onChange<K extends keyof Character>(k: K, v: any) {
        if (!form) return;
        setForm((s) => ({ ...s!, [k]: v }));
    }

    // Save updates
    async function save() {
        if (!form) return;
        setSaving(true);
        setError(null);

        try {
            const res = await fetch(
                `${baseUrl}/characters/${encodeURIComponent(form.characterId)}`,
                {
                    method: "PUT",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(form),
                }
            );

            if (!res.ok) throw new Error(`Save failed: ${res.status}`);
            const data = await res.json();
            const c = data.character;

            setForm({
                characterId: c.character_id,
                name: c.name,
                race: c.race,
                class: c.class,
                npc: c.npc ?? false,
                AC: c.AC ?? 0,
                HP: c.HP ?? 0,
                STR: c.STR ?? 0,
                DEX: c.DEX ?? 0,
                CON: c.CON ?? 0,
                INT: c.INT ?? 0,
                WIS: c.WIS ?? 0,
                CHA: c.CHA ?? 0,
            });
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setSaving(false);
        }
    }

    if (error) return <div className="text-error">{error}</div>;
    if (!form) return <div>Loading…</div>;

    return (
        <div className="max-w-2xl space-y-4">
            <h1 className="text-2xl font-bold">{form.name}</h1>

            {/* Name */}
            <div>
                <label className="block">Name</label>
                <input
                    className="input input-bordered w-full"
                    value={form.name}
                    onChange={(e) => onChange("name", e.target.value)}
                />
            </div>

            {/* Race/Class */}
            <div className="grid grid-cols-2 gap-4">
                <div>
                    <label className="block">Race</label>
                    <input
                        className="input input-bordered w-full"
                        value={form.race || ""}
                        onChange={(e) => onChange("race", e.target.value)}
                    />
                </div>
                <div>
                    <label className="block">Class</label>
                    <input
                        className="input input-bordered w-full"
                        value={form.class || ""}
                        onChange={(e) => onChange("class", e.target.value)}
                    />
                </div>
            </div>

            {/* AC / HP / NPC */}
            <div className="grid grid-cols-3 gap-4">
                <div>
                    <label>AC</label>
                    <input
                        type="number"
                        className="input input-bordered w-full"
                        value={form.AC}
                        onChange={(e) => onChange("AC", Number(e.target.value))}
                    />
                </div>
                <div>
                    <label>HP</label>
                    <input
                        type="number"
                        className="input input-bordered w-full"
                        value={form.HP}
                        onChange={(e) => onChange("HP", Number(e.target.value))}
                    />
                </div>
                <div className="flex items-end">
                    <label className="flex items-center gap-2">
                        <input
                            type="checkbox"
                            checked={!!form.npc}
                            onChange={(e) => onChange("npc", e.target.checked)}
                        />
                        NPC
                    </label>
                </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-3 gap-4">
                {["STR", "DEX", "CON", "INT", "WIS", "CHA"].map((stat) => (
                    <div key={stat}>
                        <label>{stat}</label>
                        <input
                            type="number"
                            className="input input-bordered w-full"
                            value={(form as any)[stat] ?? 0}
                            onChange={(e) =>
                                onChange(
                                    stat as keyof Character,
                                    Number(e.target.value)
                                )
                            }
                        />
                    </div>
                ))}
            </div>

            <button
                onClick={save}
                className="btn btn-primary"
                disabled={saving}
            >
                {saving ? "Saving..." : "Save"}
            </button>
        </div>
    );
}
