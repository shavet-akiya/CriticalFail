"use client";

import CharacterCard from "@/components/characterCard";
import { useFilter } from "@/contexts/FilterContext";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import type { Character } from "@/helpers/types";

export default function Characters() {
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
    const { campaignId } = useParams<{ campaignId: string }>();
    const { filter } = useFilter();

    const [characters, setCharacters] = useState<Character[]>([]);
    const [sessions, setSessions] = useState<
        { session_id: string; name: string }[]
    >([]);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [showModal, setShowModal] = useState(false);

    const [newCharacter, setNewCharacter] = useState({
        campaign_id: campaignId ?? "9d31fe",
        character_id: "0375a9",
        name: "Alice",
        race: "Unknown",
        class: "Unknown",
        npc: false,
        AC: 0,
        HP: 0,
        STR: 0,
        DEX: 0,
        CON: 0,
        INT: 0,
        WIS: 0,
        CHA: 0,
        session_id: "",
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

    // --- Fetch sessions ---
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

    // Automatically assign most recent session
    useEffect(() => {
        if (sessions.length > 0) {
            const mostRecent = sessions[sessions.length - 1].session_id;
            setSessionId(mostRecent);
            setNewCharacter((prev) => ({ ...prev, session_id: mostRecent }));
        }
    }, [sessions]);

    const filteredCharacters = characters.filter((character) => {
        if (filter === "all") return true;
        if (filter === "players") return !character.npc;
        if (filter === "npc") return character.npc;
        return true;
    });

    const handleCreateCharacter = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!campaignId || !sessionId) {
            setError("No session assigned. Please create a session first.");
            return;
        }

        // Required fields
        const requiredFields = ["name", "race", "class"];
        for (const field of requiredFields) {
            if (!(newCharacter as any)[field]) {
                setError(`Please fill in the ${field} field.`);
                return;
            }
        }

        try {
            const payload = {
                ...newCharacter,
                campaign_id: campaignId,
                session_id: sessionId,
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

            // Reset form
            setShowModal(false);
            setNewCharacter({
                campaign_id: campaignId,
                character_id: "0375a9",
                name: "Alice",
                race: "Unknown",
                class: "Unknown",
                npc: false,
                AC: 0,
                HP: 0,
                STR: 0,
                DEX: 0,
                CON: 0,
                INT: 0,
                WIS: 0,
                CHA: 0,
                session_id: sessionId,
            });
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        }
    };

    return (
        <div className="pl-16 pr-16 pt-16 text-black">
            {error && <div className="text-red-500 mb-4">{error}</div>}

            <div className="mb-8 flex justify-between items-center">
                <h1 className="text-3xl font-bold">Characters</h1>
                <button
                    onClick={() => setShowModal(true)}
                    className="bg-black text-white px-4 py-2 rounded-lg hover:bg-gray-800 transition"
                >
                    + Add Character
                </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-16">
                {filteredCharacters.map((character) => (
                    <div key={character.character_id}>
                        <CharacterCard character={character} />
                    </div>
                ))}
            </div>

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
                                        required
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
                                        required
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
                                            required
                                        />
                                    </div>
                                ))}
                            </div>

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
