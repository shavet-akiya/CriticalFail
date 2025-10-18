"use client";
import CharacterCard from "@/components/characterCard";
import { useFilter } from "@/contexts/FilterContext";
import { useEffect, useState } from "react";
import Link from "next/link";
import type { Character } from "@/types/types";

const fetchCharacters = async (): Promise<Character[]> => {
    const res = await fetch("/api/characters", { cache: "no-store" });
    if (!res.ok) throw new Error(`GET /api/characters failed: ${res.status}`);
    const data = await res.json();

    // Map snake_case from backend → camelCase for frontend
    return (data.characters ?? []).map((char: any) => ({
        characterId: char.character_id, // 👈 map here
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

export default function Characters() {
    const { filter } = useFilter();
    const [characters, setCharacters] = useState<Character[]>([]);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetchCharacters()
            .then(setCharacters)
            .catch((e: unknown) =>
                setError(e instanceof Error ? e.message : String(e))
            );
    }, []);

    const filteredCharacters = characters.filter((character) => {
        if (filter === "all") return true;
        if (filter === "players") return !character.npc;
        if (filter === "npc") return character.npc;
        return true;
    });

    return (
        <div className="pl-16 pr-16 pt-16">
            {error && <div className="text-error mb-4">{error}</div>}
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-16">
                {filteredCharacters.map((character) => (
                    <div key={character.characterId}>
                        <CharacterCard character={character} />
                        <Link
                            href={`/characters/${character.characterId}`}
                            className="btn btn-sm mt-2"
                        >
                            Edit Stats
                        </Link>
                    </div>
                ))}
            </div>
        </div>
    );
}
