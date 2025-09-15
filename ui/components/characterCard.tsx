"useClient"
import { useCharacter } from "@/contexts/CharacterContext";
import Link from "next/link"
import { Character } from "@/types";

export default function CharacterCard({ character }: { character: Character }) {
  const { setCurrentCharacter } = useCharacter();

  return (
    <Link
      href={`/characters/${character.id}`}
      onClick={() => setCurrentCharacter(character)}
      className="block">
      <div className="card bg-base-100 w-full max-w-sm shadow-sm hover:bg-gray-700 rounded-lg cursor-pointer">
        <div>
          <img
            className="rounded-t-lg"
            src="/images/picrew.png" // public is automatically served from the root (/). You don't include /public.
            alt="Character Image"
          />
        </div>
        <div className="card-body">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold">{character.name}</h2>
              <p> {character.race} / {character.class} </p>
            </div>
            <div>
              <p>AC {character.armourClass}</p>
            </div>
          </div>
        </div>
      </div>
    </Link>

  );
}