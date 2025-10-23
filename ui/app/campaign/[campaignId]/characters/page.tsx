"use client";

import CharacterCard from "@/components/characterCard";
import { useFilter } from "@/contexts/CharacterFilterContext";
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

    const [searchQuery, setSearchQuery] = useState("");

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

    const [newCharacterImage, setNewCharacterImage] = useState<File | null>(
        null
    );
    const [newCharacterImagePreview, setNewCharacterImagePreview] = useState<
        string | null
    >(null);

    // --- Fetch all characters ---
    const fetchCharacters = async (): Promise<Character[]> => {
        if (!campaignId) return [];
        const res = await fetch(`${baseUrl}/characters/${campaignId}`, {
            cache: "no-store",
        });
        if (!res.ok) throw new Error(`GET /characters failed: ${res.status}`);
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
            imageURL: char.imageURL || "",
        }));
    };

    // --- Fetch sessions for dropdown ---
    const fetchSessions = async () => {
        if (!campaignId) return;
        try {
            const res = await fetch(
                `${baseUrl}/campaign/${campaignId}/sessions`
            );
            if (!res.ok) throw new Error(`Failed to fetch sessions`);
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

    const filteredCharacters = characters
        .filter((character) => {
            if (filter === "all") return true;
            if (filter === "players") return !character.npc;
            if (filter === "npc") return character.npc;
            return true;
        })
        .filter((character) =>
            character.name.toLowerCase().includes(searchQuery.toLowerCase())
        );

    // --- Handle character creation and image upload ---
    const handleCreateCharacter = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!campaignId) return;

        try {
            // Step 1: Create character
            const createRes = await fetch(
                `${baseUrl}/characters/${campaignId}`,
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        ...newCharacter,
                        campaign_id: campaignId,
                        session_ids: selectedSessions,
                    }),
                }
            );

            if (!createRes.ok)
                throw new Error(`Create character failed: ${createRes.status}`);
            const characterData = await createRes.json();
            const createdCharacter: Character = characterData.character;

            // Step 2: Upload image if exists
            if (newCharacterImage) {
                const formData = new FormData();
                formData.append("character_image", newCharacterImage);

                const imageRes = await fetch(
                    `${baseUrl}/characters/${campaignId}/${createdCharacter.characterId}/image`,
                    {
                        method: "POST",
                        body: formData,
                    }
                );

                if (!imageRes.ok)
                    throw new Error(`Image upload failed: ${imageRes.status}`);
                const imageData = await imageRes.json();
                if (imageData.imageURL) {
                    createdCharacter.imageURL = imageData.imageURL;
                }
            }

            // Step 3: Update state immediately
            setCharacters((prev) => [...prev, createdCharacter]);

            // Step 4: Reset modal
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
            setNewCharacterImage(null);
            setNewCharacterImagePreview(null);
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        }
    };

    return (
        <div className="pl-16 pr-16 pt-16 text-black">
            <h1 className="text-3xl font-bold">Characters</h1>
            <div className="mb-8 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
                <div className="flex gap-2 flex-wrap items-center w-full sm:w-auto">
                    <input
                        type="text"
                        placeholder="Search by name..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className="border px-3 py-2 rounded-lg w-full sm:w-72"
                    />
                    <button
                        onClick={() => setShowModal(true)}
                        className="bg-black text-white px-4 py-2 rounded-lg hover:bg-gray-800 transition"
                    >
                        + Add Character
                    </button>
                </div>
            </div>

            {/* --- No Characters / Sessions Message --- */}
            {characters.length === 0 && (
                <div className="text-gray-500 text-center mt-16">
                    <p className="mb-2">No characters or sessions available.</p>
                    <p>Create a character or a session to get started.</p>
                </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-16">
                {filteredCharacters.map((character) => {
                    const imageSrc = character.imageURL
                        ? character.imageURL.startsWith("http")
                            ? character.imageURL
                            : `${baseUrl}${character.imageURL}`
                        : "/images/character-placeholder.png";

                    return (
                        <CharacterCard
                            key={character.characterId}
                            character={character}
                            imageSrc={imageSrc}
                        />
                    );
                })}
            </div>

            {showModal && (
                <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                    <div className="bg-white text-black rounded-2xl shadow-xl p-8 w-full max-w-lg overflow-y-auto max-h-[90vh]">
                        <h2 className="text-2xl font-bold mb-4">
                            Create New Character
                        </h2>

                        <form
                            onSubmit={handleCreateCharacter}
                            className="space-y-4"
                        >
                            {/* --- Name / Race / Class / NPC --- */}
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

                            {/* --- Image Picker --- */}
                            <div>
                                <label className="block text-sm font-semibold mb-1">
                                    Character Image
                                </label>
                                <input
                                    type="file"
                                    accept="image/*"
                                    onChange={(e) => {
                                        const file =
                                            e.target.files?.[0] || null;
                                        setNewCharacterImage(file);
                                        setNewCharacterImagePreview(
                                            file
                                                ? URL.createObjectURL(file)
                                                : null
                                        );
                                    }}
                                    className="w-full border rounded-lg px-3 py-2"
                                />
                                {newCharacterImagePreview && (
                                    <img
                                        src={newCharacterImagePreview}
                                        alt="Preview"
                                        className="mt-2 w-32 h-32 object-cover rounded-lg border"
                                    />
                                )}
                            </div>

                            {/* --- Session Selector --- */}
                            <div>
                                <label className="block text-sm font-semibold mb-1">
                                    Assign to Session(s)
                                </label>
                                <select
                                    multiple
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

                            <div className="flex justify-end gap-4 pt-4">
                                <button
                                    type="button"
                                    onClick={() => {
                                        setShowModal(false);
                                        setNewCharacterImage(null);
                                        setNewCharacterImagePreview(null);
                                    }}
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
