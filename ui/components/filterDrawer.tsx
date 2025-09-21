// ui/components/filterDrawer.tsx
import { CampaignTags } from "@/types/types";

const allTags: CampaignTags[] = [
    "combat", "exploring", "player-to-player interaction", "npc interaction",
    "resting", "investigating", "world expansion", "character expansion",
    "lore expansion", "misc"
];

type FilterDrawerProps = {
    characterFilter: "all" | "players" | "npc";
    setCharacterFilter: (value: "all" | "players" | "npc") => void;
    tagFilter: string[];
    setTagFilter: (value: string[]) => void;
};

export default function FilterDrawer({
    characterFilter,
    setCharacterFilter,
    tagFilter,
    setTagFilter,
}: FilterDrawerProps) {
    const toggleTag = (tag: string) => {
        if (tagFilter.includes(tag)) {
            setTagFilter(tagFilter.filter((t) => t !== tag));
        } else {
            setTagFilter([...tagFilter, tag]);
        }
    };

    return (
        <aside className="bg-base-200 p-4 border-r border-base-300">
            <h2 className="text-lg font-bold mb-4">Filters</h2>

            {/* Character Filter */}
            <div className="mb-6">
                <h3 className="font-semibold mb-2">Characters</h3>
                <div className="flex flex-col gap-2">
                    {["all", "players", "npc"].map((option) => (
                        <label key={option} className="cursor-pointer">
                            <input
                                type="radio"
                                name="characterFilter"
                                className="radio mr-2"
                                checked={characterFilter === option}
                                onChange={() => setCharacterFilter(option as "all" | "players" | "npc")}
                            />
                            {option}
                        </label>
                    ))}
                </div>
            </div>

            {/* Tags Filter */}
            <div className="mb-6">
                <h3 className="font-semibold mb-2">Tags</h3>
                <div className="flex flex-col gap-1">
                    {allTags.map((tag) => (
                        <label key={tag} className="cursor-pointer">
                            <input
                                type="checkbox"
                                className="checkbox mr-2"
                                checked={tagFilter.includes(tag)}
                                onChange={() => toggleTag(tag)}
                            />
                            {tag}
                        </label>
                    ))}
                </div>
            </div>

            <div>
                <h3 className="font-semibold mb-2">Themes</h3>
                <p className="text-sm opacity-70">Coming soon</p>
            </div>
        </aside>
    );
}
