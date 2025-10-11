"use client"
import CharacterCard from "@/components/characterCard";
import { useFilter } from "@/contexts/FilterContext";

// ids for when we inevitably need to click into characters
const characters = [
  {
    id: 1,
    name: "Thalindra Moonshadow",
    class: "Rogue",
    race: "Half-Elf",
    armourClass: 16,
    npc: false,
    hp: 1,
    str: 1,
    dex: 1,
    con: 1,
    int: 1,
    wis: 1,
    cha: 1,
  },
  {
    id: 2,
    name: "Brom Ironfist",
    class: "Cleric",
    race: "Dwarf",
    armourClass: 18,
    npc: true,
    hp: 1,
    str: 1,
    dex: 1,
    con: 1,
    int: 1,
    wis: 1,
    cha: 1,
  },
  {
    id: 3,
    name: "Seraphina Dawnsworn",
    class: "Paladin",
    race: "Human",
    armourClass: 20,
    npc: true,
    hp: 1,
    str: 1,
    dex: 1,
    con: 1,
    int: 1,
    wis: 1,
    cha: 1,
  },
  {
    id: 4,
    name: "Zyren the Ashborn",
    class: "Warlock",
    race: "Tiefling",
    armourClass: 14,
    npc: false,
    hp: 1,
    str: 1,
    dex: 1,
    con: 1,
    int: 1,
    wis: 1,
    cha: 1,
  }
]

export default function Characters() {
  const { filter } = useFilter();

  const filteredCharacters = characters.filter((character) => {
    if (filter === "all") return true;
    if (filter === "players") return !character.npc;
    if (filter === "npc") return character.npc;
    return true;
  });

  return (
    <>
      <div className="p-16 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-16">
        {filteredCharacters.map((character, index) => (
          <CharacterCard key={index} character={character} />
        ))}
      </div>
    </>
  );
}
