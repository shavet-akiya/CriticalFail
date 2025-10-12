"use client";
import CharacterCard from "@/components/characterCard";
import { useFilter } from "@/contexts/FilterContext";
import { useEffect, useState } from "react";
import type { Character } from "@/types/types";

const fetchCharacters = async (): Promise<Partial<Character>[]> => {
    const res = await fetch("/api/characters", { cache: "no-store" });
    if (!res.ok) throw new Error(`GET /api/characters failed: ${res.status}`);
    const data = await res.json();
    const chars: Partial<Character>[] = data.characters ?? [];

    // Client-side dedupe by name (stable: first occurrence wins)
    const map = new Map<string, Partial<Character>>();
    for (const c of chars) {
        if (!c || !("name" in c)) continue;
        const name = (c as any).name as string;
        if (!map.has(name)) map.set(name, c);
    }
    return Array.from(map.values());
};

export default function Characters() {
    const { filter } = useFilter();
    const [characters, setCharacters] = useState<Partial<Character>[]>([]);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        fetchCharacters()
            .then(setCharacters)
            .catch((e: unknown) =>
                setError(e instanceof Error ? e.message : String(e))
            );
    }, []);

    const filteredCharacters = characters.filter((character: any) => {
        if (filter === "all") return true;
        if (filter === "players") return !character.npc;
        if (filter === "npc") return character.npc;
        return true;
    });

    return (
        <>
            <div className="p-16">
                {error && <div className="text-error mb-4">{error}</div>}
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-16">
                    {filteredCharacters.map((character: any) => (
                        <CharacterCard
                            key={character.name}
                            character={character as Character}
                        />
                    ))}
                </div>
            </div>
        </>
    );
}
