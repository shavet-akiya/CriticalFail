// ui/timeline/page.tsx
"use client";
import { useState } from "react";
import { events } from "@/types/mockData";
import EventCard from "@/components/eventCard";
import FilterDrawer from "@/components/filterDrawer";

export default function Timeline() {
  const [characterFilter, setCharacterFilter] = useState<"all" | "players" | "npc">("all");
  const [tagFilter, setTagFilter] = useState<string[]>([]);
  const [themeFilter, setThemeFilter] = useState<string[]>([]);

  // filtering 
  const filteredEvents = events.filter((event) => {
    if (tagFilter.length > 0 && !event.tags.some((tag) => tagFilter.includes(tag))) {
      return false;
    }
    return true;
  });

  return (
    <div className="w-full">
      <div className="grid grid-cols-[250px_1fr] min-h-screen">
        <FilterDrawer
          characterFilter={characterFilter}
          setCharacterFilter={setCharacterFilter}
          tagFilter={tagFilter}
          setTagFilter={setTagFilter}
        />

        <main className="p-6">
          <h1 className="text-2xl font-bold mb-6">Timeline</h1>
          {filteredEvents.map((event) => (
            <EventCard key={event.id} event={event} />
          ))}
        </main>
      </div>

    </div>
  );
}
