"use client"
import CharacterCard from "@/components/characterCard";
import { useFilter } from "@/contexts/FilterContext";
import { characters } from "@/types/mockData"
import { Character } from "@/types/types";
import { Key } from "react";

// ids for when we inevitably need to click into characters

export default function Characters() {
  const { filter } = useFilter();

  const filteredCharacters = characters.filter((character: { npc: any; }) => {
    if (filter === "all") return true;
    if (filter === "players") return !character.npc;
    if (filter === "npc") return character.npc;
    return true;
  });

  return (
    <>
      <div className="p-16 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-16">
        {filteredCharacters.map((character: Character, index: Key | null | undefined) => (
          <CharacterCard key={index} character={character} />
        ))}
      </div>
    </>
  );
}
