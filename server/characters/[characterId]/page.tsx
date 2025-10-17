"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import type { Character } from "@/types/types";

export default function CharacterDetail() {
    const { characterId } = useParams<{ characterId: string }>();
    const [originalCharacter, setOriginalCharacter] =
        useState<Character | null>(null);
    const [character, setCharacter] = useState<Character | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [saving, setSaving] = useState(false);

    const baseUrl = "http://localhost:9000"; // FastAPI backend

    // Fetch character
    useEffect(() => {
        if (!characterId) return;

        fetch(`${baseUrl}/characters/${encodeURIComponent(characterId)}`)
            .then(async (res) => {
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                return res.json();
            })
            .then((data) => {
                if (!data.character) throw new Error("Character not found");

                setCharacter(data.character);
                setOriginalCharacter(data.character); // include character_id
            })
            .catch((e) => {
                console.error("Error fetching character:", e);
                setError(e instanceof Error ? e.message : String(e));
            });
    }, [characterId]);

    // Update form state
    const onChange = <K extends keyof Character>(key: K, value: any) => {
        if (!character) return;
        setCharacter({ ...character, [key]: value });
    };

    const saveCharacter = async () => {
        if (!character || !originalCharacter) return;
        if (!characterId) {
            setError("Character ID missing in URL. Cannot save.");
            return;
        }

        setSaving(true);
        setError(null);

        // Build a patch of only changed fields
        const patch: Partial<Character> = {};
        for (const key in character) {
            if ((character as any)[key] !== (originalCharacter as any)[key]) {
                (patch as any)[key] = (character as any)[key];
            }
        }

        if (Object.keys(patch).length === 0) {
            setSaving(false);
            return; // nothing changed
        }

        try {
            const res = await fetch(
                `${baseUrl}/characters/${encodeURIComponent(characterId)}`,
                {
                    method: "PATCH",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(patch),
                }
            );

            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            setCharacter(data.character);
            setOriginalCharacter(data.character); // update original
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : String(e));
        } finally {
            setSaving(false);
        }
    };

    if (error) return <div className="text-error">{error}</div>;
    if (!character) return <div>Loading...</div>;

    return (
        <div className="max-w-2xl space-y-4 p-6">
            <h1 className="text-3xl font-bold">Edit Character</h1>

            {/* Name */}
            <div>
                <label className="block">Name</label>
                <input
                    className="input input-bordered w-full"
                    value={character.name || ""}
                    onChange={(e) => onChange("name", e.target.value)}
                />
            </div>

            {/* Race/Class */}
            <div className="grid grid-cols-2 gap-4">
                <div>
                    <label>Race</label>
                    <input
                        className="input input-bordered w-full"
                        value={character.race || ""}
                        onChange={(e) => onChange("race", e.target.value)}
                    />
                </div>
                <div>
                    <label>Class</label>
                    <input
                        className="input input-bordered w-full"
                        value={character.class || ""}
                        onChange={(e) => onChange("class", e.target.value)}
                    />
                </div>
            </div>

            {/* NPC / AC / HP */}
            <div className="grid grid-cols-3 gap-4">
                <div>
                    <label>AC</label>
                    <input
                        type="number"
                        className="input input-bordered w-full"
                        value={character.AC ?? 0}
                        onChange={(e) => onChange("AC", Number(e.target.value))}
                    />
                </div>
                <div>
                    <label>HP</label>
                    <input
                        type="number"
                        className="input input-bordered w-full"
                        value={character.HP ?? 0}
                        onChange={(e) => onChange("HP", Number(e.target.value))}
                    />
                </div>
                <div className="flex items-end">
                    <label className="flex items-center gap-2">
                        <input
                            type="checkbox"
                            checked={!!character.npc}
                            onChange={(e) => onChange("npc", e.target.checked)}
                        />
                        NPC
                    </label>
                </div>
            </div>

            {/* Stats */}
            <div>
                <h2 className="text-xl font-semibold mt-4">Stats</h2>
                <div className="grid grid-cols-3 gap-4 mt-2">
                    {["STR", "DEX", "CON", "INT", "WIS", "CHA"].map((stat) => (
                        <div key={stat}>
                            <label>{stat}</label>
                            <input
                                type="number"
                                className="input input-bordered w-full"
                                value={(character as any)[stat] ?? 0}
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
            </div>

            <button
                className="btn btn-primary mt-4"
                onClick={saveCharacter}
                disabled={saving}
            >
                {saving ? "Saving..." : "Save Changes"}
            </button>
        </div>
    );
}
