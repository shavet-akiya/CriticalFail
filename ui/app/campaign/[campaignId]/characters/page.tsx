"use client";

import CharacterCard from "@/components/characterCard";
import { useFilter } from "@/contexts/CharacterFilterContext";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import type { Character } from "@/types/types";
import Loading from "@/components/Loading";
import { Search } from "lucide-react";

export default function Characters() {
    const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
    const { campaignId } = useParams<{ campaignId: string }>();
    const { filter } = useFilter();

    const [characters, setCharacters] = useState<Character[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [showModal, setShowModal] = useState(false);
    const [searchQuery, setSearchQuery] = useState("");
    const [loading, setLoading] = useState(true);

    const [newCharacter, setNewCharacter] = useState({
        name: "",
        race: "",
        class: "",
        npc: false,
        AC: 0,
        HP: 0,
        STR: 0,
        DEX: 0,
        CON: 0,
        INT: 0,
        WIS: 0,
        CHA: 0,
    });

    const [newCharacterImage, setNewCharacterImage] = useState<File | null>(
        null
    );
    const [newCharacterImagePreview, setNewCharacterImagePreview] = useState<
        string | null
    >(null);

    // ---- Fetch Characters Only ----
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

    useEffect(() => {
        if (!campaignId) return;

        setLoading(true);
        setError(null);

        fetchCharacters()
            .then((chars) => setCharacters(chars))
            .catch((e) => setError(e instanceof Error ? e.message : String(e)))
            .finally(() => setLoading(false));
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

    const handleCreateCharacter = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!campaignId) return;

        try {
            setLoading(true);
            const createRes = await fetch(
                `${baseUrl}/characters/${campaignId}`,
                {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        ...newCharacter,
                        campaign_id: campaignId,
                    }),
                }
            );

            if (!createRes.ok)
                throw new Error(`Create character failed: ${createRes.status}`);
            const characterData = await createRes.json();
            const createdCharacter: Character = characterData.character;

            if (newCharacterImage) {
                const formData = new FormData();
                formData.append("character_image", newCharacterImage);

                const imageRes = await fetch(
                    `${baseUrl}/characters/${campaignId}/${createdCharacter.characterId}/image`,
                    { method: "POST", body: formData }
                );

                if (!imageRes.ok)
                    throw new Error(`Image upload failed: ${imageRes.status}`);
                const imageData = await imageRes.json();
                if (imageData.imageURL)
                    createdCharacter.imageURL = imageData.imageURL;
            }

            setCharacters((prev) => [...prev, createdCharacter]);
            setShowModal(false);
            setNewCharacter({
                name: "",
                race: "",
                class: "",
                npc: false,
                AC: 0,
                HP: 0,
                STR: 0,
                DEX: 0,
                CON: 0,
                INT: 0,
                WIS: 0,
                CHA: 0,
            });
            setNewCharacterImage(null);
            setNewCharacterImagePreview(null);
            setError(null);
        } catch (err) {
            setError(err instanceof Error ? err.message : String(err));
        } finally {
            setLoading(false);
        }
    };

    if (loading) return <Loading />;

    return (
        <div className="h-full w-[80vw] select-none gap-5 overflow-hidden padding-box">
            <div className="heading-banner obsidian-colour px-8 select-none">
                <h1 className="page-heading pb-4">Characters</h1>

                <div className="mb-4 flex flex-col sm:flex-row gap-4 px-2">
                    <div className="relative flex-1">
                        <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 w-5 h-5 pointer-events-none" />
                        <input
                            type="text"
                            placeholder="Search by name..."
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            className="border bg-white border-gray-300 pl-10 pr-3 py-2 rounded-xl w-full obsidian-colour focus:outline-none focus:ring-2 focus:ring-purple-400 focus:border-purple-400 transition shadow-sm"
                        />
                    </div>
                    <button
                        onClick={() => setShowModal(true)}
                        className="bg-[#a80d18] white-colour px-4 py-2 rounded-lg font-semibold hover:bg-purple-700 transition shadow-md w-full sm:w-auto"
                    >
                        + Add Character
                    </button>
                </div>
            </div>

            {/* --- Character Grid --- */}
            <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-8 w-full">
                {filteredCharacters.length === 0 ? (
                    <div className="text-gray-500 text-center col-span-full mt-16">
                        <p className="mb-2">No characters yet.</p>
                        <p>Create a character to get started!</p>
                    </div>
                ) : (
                    filteredCharacters.map((character) => {
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
                    })
                )}
            </div>

            {/* --- Modal --- */}
            {showModal && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                    <div className="bg-white obsidian-colour rounded-2xl shadow-xl p-8 w-full max-w-lg overflow-y-auto max-h-[90vh]">
                        <h2 className="text-2xl font-bold mb-4">
                            Create New Character
                        </h2>

                        <form
                            onSubmit={handleCreateCharacter}
                            className="space-y-4"
                        >
                            {/* Name / Race / Class / NPC */}
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
                                        placeholder="e.g. Tiefling"
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
                                        placeholder="e.g. Rogue"
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

                            {/* Character Stats */}
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
                                    className="bg-black white-colour px-4 py-2 rounded-lg hover:bg-gray-800 transition"
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
