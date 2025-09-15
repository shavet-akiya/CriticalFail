type Character = {
  name: string;
  class: string;
  race: string;
  armourClass: number;
  npc: boolean;
};


export default function CharacterCard({ character }: { character: Character }) {
  return (
    <div className="card bg-base-100 w-full max-w-sm shadow-sm hover:bg-gray-700 rounded-lg">
      <div className="">
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
            <p className="italic">{character.class} / {character.race} </p>
          </div>
          <div>
            <p>AC {character.armourClass}</p>
          </div>
        </div>
      </div>
    </div>
  );
}