"use client";
import { useCharacter } from "@/contexts/CharacterContext";

export default function CharacterPage() {
    const { currentCharacter: character } = useCharacter();

    if (!character) return <p>No character selected</p>;

    return (
        <div className="p-16 max-w-3xl mx-auto">
            <h1 className="text-3xl font-bold mb-4">{character.name}</h1>
            <p className="italic mb-4">{character.race} / {character.class}</p>
            <p>AC: {character.armourClass}</p>
            <p>HP: {character.hp}</p>

            <div className="grid grid-cols-3 gap-4 mt-4">
                <p>STR: {character.str}</p>
                <p>DEX: {character.dex}</p>
                <p>CON: {character.con}</p>
                <p>INT: {character.int}</p>
                <p>WIS: {character.wis}</p>
                <p>CHA: {character.cha}</p>
            </div>
        </div>
    );
}
