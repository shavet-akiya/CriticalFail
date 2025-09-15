"use client"
import CharacterCard from "@/components/characterCard";
import { useFilter } from "@/components/handlers/FilterContext";


const characters = [
  {
    "name": "Thalindra Moonshadow",
    "class": "Rogue",
    "race": "Half-Elf",
    "armourClass": 16,
    "npc": false
  },
  {
    "name": "Brom Ironfist",
    "class": "Cleric",
    "race": "Dwarf",
    "armourClass": 18,
    "npc": true
  },
  {
    "name": "Seraphina Dawnsworn",
    "class": "Paladin",
    "race": "Human",
    "armourClass": 20,
    "npc": true
  },
  {
    "name": "Zyren the Ashborn",
    "class": "Warlock",
    "race": "Tiefling",
    "armourClass": 14,
    "npc": false
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
      <div className="p-16 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-16">
        {filteredCharacters.map((character, index) => (
          <CharacterCard key={index} character={character} />
        ))}
      </div>
    </>
  );
}
