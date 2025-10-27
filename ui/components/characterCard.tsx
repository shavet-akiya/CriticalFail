"use client";

import { useCharacter } from "@/contexts/CharacterContext";
import Link from "next/link";
import { useParams } from "next/navigation";
import { Character } from "@/types/types";

interface CharacterCardProps {
    character: Character;
    imageSrc: string;
}

export default function CharacterCard({
    character,
    imageSrc,
}: CharacterCardProps) {
    const { setCurrentCharacter } = useCharacter();
    const { campaignId } = useParams<{ campaignId: string }>();

    return (
        <Link
            href={`/campaign/${campaignId}/characters/${character.characterId}`}
            onClick={() => setCurrentCharacter(character)}
            className="block"
        >
            <div className="card p-6 bg-white border-3 border-purple w-full max-w-sm shadow-sm hover:bg-gray-300 duration-300 rounded-3xl cursor-pointer group relative">
                <div className=" bg-purple-colour rounded-lg white-colour absolute bottom-4 right-4 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button className=" font-semibold  px-3 py-1 shadow-md hover:shadow-lg hover:bg-red-900 hover:rounded-lg transition-all flex items-center gap-1">
                        View
                    </button>
                </div>

                <figure>
                    <img
                        className="rounded-2xl w-70 h-70 object-cover border-2 border-purple"
                        src={imageSrc}
                        alt={character.name || "Character Image"}
                        onError={(e) => {
                            (e.target as HTMLImageElement).src =
                                "./images/character-placeholder.png";
                        }}
                    />
                </figure>

                <div className="card-body obsidian-colour">
                    <div className="flex justify-between">
                        <div>
                            <h2 className="text-xl font-bold">
                                {character.name || "Unnamed"}
                            </h2>
                            <p>
                                {character.race || "Unknown"} /{" "}
                                {character.class || "Unknown"}
                            </p>
                        </div>
                        <div>
                            <p>AC {character.AC ?? 0}</p>
                            <p>HP {character.HP ?? 0}</p>
                        </div>
                    </div>

                    <div className="grid grid-cols-3 gap-2 mt-4 text-sm">
                        <div>
                            <strong>STR</strong> {character.STR ?? 0}
                        </div>
                        <div>
                            <strong>DEX</strong> {character.DEX ?? 0}
                        </div>
                        <div>
                            <strong>CON</strong> {character.CON ?? 0}
                        </div>
                        <div>
                            <strong>INT</strong> {character.INT ?? 0}
                        </div>
                        <div>
                            <strong>WIS</strong> {character.WIS ?? 0}
                        </div>
                        <div>
                            <strong>CHA</strong> {character.CHA ?? 0}
                        </div>
                    </div>

                    <div className="mt-2">
                        NPC: {character.npc ? "Yes" : "No"}
                    </div>
                </div>
            </div>
        </Link>
    );
}
