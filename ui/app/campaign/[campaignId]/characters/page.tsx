"use client";

import CharacterCard from "@/components/characterCard";
import { useFilter } from "@/contexts/FilterContext";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import type { Character } from "@/types/types";

export default function Characters() {
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
    const { campaignId } = useParams<{ campaignId: string }>();
    const { filter } = useFilter();

    const [characters, setCharacters] = useState<Character[]>([]);
    const [sessions, setSessions] = useState<
        { session_id: string; name: string }[]
    >([]);
    const [selectedSessions, setSelectedSessions] = useState<string[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [showModal, setShowModal] = useState(false);

    const [newCharacter, setNewCharacter] = useState({
        name: "",
        race: "Human",
        class: "Fighter",
        npc: false,
        AC: 10,
        HP: 10,
        STR: 10,
        DEX: 10,
        CON: 10,
        INT: 10,
        WIS: 10,
        CHA: 10,
    });

    // --- Fetch all characters ---
    const fetchCharacters = async (): Promise<Character[]> => {
        if (!campaignId) return [];
        const res = await fetch(`${baseUrl}/characters/${campaignId}`, {
            cache: "no-store",
        });
        if (!res.ok)
            throw new Error(`GET /api/characters failed: ${res.status}`);
        const data = await res.json();

        return (data.characters ?? []).map((char: any) => ({
            characterId: char.character_id,
            name: char.name,
            race: char.race,
            class: char.class,
            npc: char.npc ?? false,
            AC: char.AC ?? 0,
            HP: char.HP ?? 0,
            STR: char.STR ?? 0,
            DEX: char.DEX ?? 0,
            CON: char.CON ?? 0,
            INT: char.INT ?? 0,
            WIS: char.WIS ?? 0,
            CHA: char.CHA ?? 0,
        }));
    };

    // --- Fetch sessions for dropdown ---
    const fetchSessions = async () => {
        if (!campaignId) return;
        try {
            const res = await fetch(
                `${baseUrl}/campaign/${campaignId}/sessions`
            );
            if (!res.ok)
                throw new Error(`GET /api/sessions failed: ${res.status}`);
            const data = await res.json();
            setSessions(data.sessions ?? []);
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
    };

    useEffect(() => {
        fetchCharacters()
            .then(setCharacters)
            .catch((e) => setError(e instanceof Error ? e.message : String(e)));
        fetchSessions();
    }, [campaignId]);

    const filteredCharacters = characters.filter((character) => {
        if (filter === "all") return true;
        if (filter === "players") return !character.npc;
        if (filter === "npc") return character.npc;
        return true;
    });

    const handleCreateCharacter = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!campaignId) return;

        try {
            const payload = {
                ...newCharacter,
                session_ids: [],
                campaign_id: campaignId,
            };

            const res = await fetch(`${baseUrl}/characters/${campaignId}`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!res.ok) throw new Error(`POST failed: ${res.status}`);
            await res.json();

            // Refresh characters
            const updatedCharacters = await fetchCharacters();
            setCharacters(updatedCharacters);

            // Close modal & reset form
            setShowModal(false);
            setNewCharacter({
                name: "",
                race: "Human",
                class: "Fighter",
                npc: false,
                AC: 10,
                HP: 10,
                STR: 10,
                DEX: 10,
                CON: 10,
                INT: 10,
                WIS: 10,
                CHA: 10,
            });
            setSelectedSessions([]);
            setError(null);

            // Scroll to character grid
            const grid = document.querySelector(".grid.grid-cols-1");
            grid?.scrollIntoView({ behavior: "smooth" });
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        }
    };

    return (
        <div className="pl-16 pr-16 pt-16 text-black">
            {error && <div className="text-red-500 mb-4">{error}</div>}

            {/* --- Header + Add Button --- */}
            <div className="mb-8 flex justify-between items-center">
                <h1 className="text-3xl font-bold">Characters</h1>
                <button
                    onClick={() => setShowModal(true)}
                    className="bg-black text-white px-4 py-2 rounded-lg hover:bg-gray-800 transition"
                >
                    + Add Character
                </button>
            </div>

            {/* --- Character Grid --- */}
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-16">
                {filteredCharacters.map((character) => (
                    <div key={character.characterId}>
                        <CharacterCard character={character} />
                    </div>
                ))}
            </div>

            {/* --- Modal for Character Creation --- */}
            {showModal && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="bg-white text-black rounded-2xl shadow-xl p-8 w-full max-w-lg">
                        <h2 className="text-2xl font-bold mb-4">
                            Create New Character
                        </h2>

                        <form
                            onSubmit={handleCreateCharacter}
                            className="space-y-4"
                        >
                            {/* --- Name / Race / Class --- */}
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <label className="block text-sm font-semibold mb-1">
                                        Name
                                    </label>
                                    <input
                                        type="text"
                                        value={newCharacter.name}
                                        onChange={(e) =>
                                            setNewCharacter({
                                                ...newCharacter,
                                                name: e.target.value,
                                            })
                                        }
                                        className="w-full border rounded-lg px-3 py-2"
                                        required
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-semibold mb-1">
                                        Race
                                    </label>
                                    <input
                                        type="text"
                                        value={newCharacter.race}
                                        onChange={(e) =>
                                            setNewCharacter({
                                                ...newCharacter,
                                                race: e.target.value,
                                            })
                                        }
                                        className="w-full border rounded-lg px-3 py-2"
                                    />
                                </div>
                                <div>
                                    <label className="block text-sm font-semibold mb-1">
                                        Class
                                    </label>
                                    <input
                                        type="text"
                                        value={newCharacter.class}
                                        onChange={(e) =>
                                            setNewCharacter({
                                                ...newCharacter,
                                                class: e.target.value,
                                            })
                                        }
                                        className="w-full border rounded-lg px-3 py-2"
                                    />
                                </div>
                                <div className="flex items-center gap-2">
                                    <input
                                        type="checkbox"
                                        checked={newCharacter.npc}
                                        onChange={(e) =>
                                            setNewCharacter({
                                                ...newCharacter,
                                                npc: e.target.checked,
                                            })
                                        }
                                    />
                                    <label className="text-sm font-semibold">
                                        Is NPC?
                                    </label>
                                </div>
                            </div>

                            {/* --- Session Selector --- */}
                            <div>
                                <label className="block text-sm font-semibold mb-1">
                                    Assign to Session(s)
                                </label>
                                <select
                                    multiple
                                    required
                                    value={selectedSessions}
                                    onChange={(e) =>
                                        setSelectedSessions(
                                            Array.from(
                                                e.target.selectedOptions,
                                                (opt) => opt.value
                                            )
                                        )
                                    }
                                    className="w-full border rounded-lg px-3 py-2"
                                >
                                    {sessions.length > 0 ? (
                                        sessions.map((s) => (
                                            <option
                                                key={s.session_id}
                                                value={s.session_id}
                                            >
                                                {s.name}
                                            </option>
                                        ))
                                    ) : (
                                        <option disabled>
                                            No sessions available
                                        </option>
                                    )}
                                </select>
                                <p className="text-xs text-gray-500 mt-1">
                                    Hold Ctrl/Cmd to select multiple sessions.
                                </p>
                            </div>

                            {/* --- Stats Section --- */}
                            <div className="grid grid-cols-3 gap-3">
                                {[
                                    "AC",
                                    "HP",
                                    "STR",
                                    "DEX",
                                    "CON",
                                    "INT",
                                    "WIS",
                                    "CHA",
                                ].map((stat) => (
                                    <div key={stat}>
                                        <label className="block text-xs font-semibold mb-1">
                                            {stat}
                                        </label>
                                        <input
                                            type="number"
                                            value={(newCharacter as any)[stat]}
                                            onChange={(e) =>
                                                setNewCharacter({
                                                    ...newCharacter,
                                                    [stat]: Number(
                                                        e.target.value
                                                    ),
                                                })
                                            }
                                            className="w-full border rounded-lg px-2 py-1"
                                        />
                                    </div>
                                ))}
                            </div>

                            {/* --- Buttons --- */}
                            <div className="flex justify-end gap-4 pt-4">
                                <button
                                    type="button"
                                    onClick={() => setShowModal(false)}
                                    className="px-4 py-2 rounded-lg border border-gray-400 hover:bg-gray-200 transition"
                                >
                                    Cancel
                                </button>
                                <button
                                    type="submit"
                                    className="bg-black text-white px-4 py-2 rounded-lg hover:bg-gray-800 transition"
                                >
                                    Save
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
}
