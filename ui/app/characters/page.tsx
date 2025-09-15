import CharacterCard from "@/components/characterCard";

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
  return (
    <div className="p-16 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-16">
      {characters.map((character, index) => (
        <CharacterCard key={index} character={character} />
      ))}
    </div>
  );
}
