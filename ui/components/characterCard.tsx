"use client";

import { useCharacter } from "@/contexts/CharacterContext";
import Link from "next/link";
import { useParams } from "next/navigation"; // ✅ import useParams
import { Character } from "@/helpers/types";

export default function CharacterCard({ character }: { character: Character }) {
    const { setCurrentCharacter } = useCharacter();
    const { campaignId } = useParams<{ campaignId: string }>(); // ✅ get campaignId from route

    return (
        <Link
            href={`/campaign/${campaignId}/characters/${character.characterId}`}
            onClick={() => setCurrentCharacter(character)}
            className="block"
        >
            <div className="card bg-base-100 w-full max-w-sm shadow-sm hover:bg-gray-700 rounded-lg cursor-pointer group relative">
                {/* Edit button */}
                <div className="absolute top-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg">
                    <button className="btn btn-primary rounded-full w-auto flex items-center gap-1">
                        <img
                            src="/svg/edit.svg"
                            alt="Edit"
                            className="w-4 h-4"
                        />
                        Edit
                    </button>
                </div>

                {/* Character image */}
                <div>
                    <img
                        className="rounded-t-lg"
                        src="/images/picrew.png"
                        alt="Character Image"
                    />
                </div>

                {/* Character info */}
                <div className="card-body text-white">
                    <div className="flex items-center justify-between">
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

                    {/* Stats grid */}
                    <div className="grid grid-cols-4 gap-2 mt-4 text-sm">
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
