"use client";
import { useState } from "react";
import { useCharacter } from "@/contexts/CharacterContext";
import type { Character } from "@/types/types";

export default function CharacterPage() {
    const { currentCharacter: initial, setCurrentCharacter } = useCharacter();
    if (!initial) return <p>No character selected</p>;

    const [form, setForm] = useState<Character>({ ...initial } as Character);
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);

    function onChange<K extends keyof Character>(k: K, v: any) {
        setForm((s) => ({ ...s, [k]: v }));
    }

    async function save() {
        setSaving(true);
        setError(null);
        try {
            // map client keys to server expected fields (char_class, armour_class)
            const payload = {
                id: form.id,
                name: form.name,
                race: form.race,
                char_class: form.class,
                armour_class: Number(form.armourClass || 0),
                hp: Number(form.hp || 0),
                str: Number(form.str || 0),
                dex: Number(form.dex || 0),
                con: Number(form.con || 0),
                int: Number(form.int || 0),
                wis: Number(form.wis || 0),
                cha: Number(form.cha || 0),
                npc: Boolean(form.npc),
            };
            const res = await fetch("/api/characters", {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            if (!res.ok) throw new Error(`save failed: ${res.status}`);
            const data = await res.json();
            const updated = data.character ?? payload;
            // normalize back to client shape (camelCase field names)
            const clientChar: Character = {
                id: updated.id,
                name: updated.name,
                race: updated.race,
                class: updated.class || updated.char_class || "",
                armourClass: updated.armourClass ?? updated.armour_class ?? 0,
                hp: updated.hp ?? 0,
                str: updated.str ?? 0,
                dex: updated.dex ?? 0,
                con: updated.con ?? 0,
                int: updated.int ?? 0,
                wis: updated.wis ?? 0,
                cha: updated.cha ?? 0,
                npc: !!updated.npc,
            } as Character;

            setCurrentCharacter(clientChar);
            setForm(clientChar);
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setSaving(false);
        }
    }

    return (
        <div className="max-w-2xl">
            {error && <div className="text-red-600 mb-2">{error}</div>}
            <div className="mb-4">
                <label className="block">Name</label>
                <input
                    value={form.name}
                    onChange={(e) => onChange("name", e.target.value)}
                    className="input input-bordered w-full"
                />
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                    <label className="block">Race</label>
                    <input
                        value={form.race || ""}
                        onChange={(e) => onChange("race", e.target.value)}
                        className="input input-bordered w-full"
                    />
                </div>
                <div>
                    <label className="block">Class</label>
                    <input
                        value={form.class || ""}
                        onChange={(e) => onChange("class", e.target.value)}
                        className="input input-bordered w-full"
                    />
                </div>
            </div>

            <div className="grid grid-cols-3 gap-4 mb-4">
                <div>
                    <label className="block">AC</label>
                    <input
                        type="number"
                        value={form.armourClass ?? 0}
                        onChange={(e) =>
                            onChange("armourClass", Number(e.target.value))
                        }
                        className="input input-bordered w-full"
                    />
                </div>
                <div>
                    <label className="block">HP</label>
                    <input
                        type="number"
                        value={form.hp ?? 0}
                        onChange={(e) => onChange("hp", Number(e.target.value))}
                        className="input input-bordered w-full"
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

            <div className="grid grid-cols-3 gap-4 mb-4">
                <div>
                    <label>STR</label>
                    <input
                        type="number"
                        value={form.str ?? 0}
                        onChange={(e) =>
                            onChange("str", Number(e.target.value))
                        }
                        className="input input-bordered w-full"
                    />
                </div>
                <div>
                    <label>DEX</label>
                    <input
                        type="number"
                        value={form.dex ?? 0}
                        onChange={(e) =>
                            onChange("dex", Number(e.target.value))
                        }
                        className="input input-bordered w-full"
                    />
                </div>
                <div>
                    <label>CON</label>
                    <input
                        type="number"
                        value={form.con ?? 0}
                        onChange={(e) =>
                            onChange("con", Number(e.target.value))
                        }
                        className="input input-bordered w-full"
                    />
                </div>
                <div>
                    <label>INT</label>
                    <input
                        type="number"
                        value={form.int ?? 0}
                        onChange={(e) =>
                            onChange("int", Number(e.target.value))
                        }
                        className="input input-bordered w-full"
                    />
                </div>
                <div>
                    <label>WIS</label>
                    <input
                        type="number"
                        value={form.wis ?? 0}
                        onChange={(e) =>
                            onChange("wis", Number(e.target.value))
                        }
                        className="input input-bordered w-full"
                    />
                </div>
                <div>
                    <label>CHA</label>
                    <input
                        type="number"
                        value={form.cha ?? 0}
                        onChange={(e) =>
                            onChange("cha", Number(e.target.value))
                        }
                        className="input input-bordered w-full"
                    />
                </div>
            </div>

            <div className="flex gap-2">
                <button
                    onClick={save}
                    className="btn btn-primary"
                    disabled={saving}
                >
                    {saving ? "Saving..." : "Save"}
                </button>
            </div>
        </div>
    );
}
