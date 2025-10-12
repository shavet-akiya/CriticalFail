"use client";
import { useState } from "react";
import { events } from "@/types/mockData";
import EventCard from "@/components/eventCard";
import FilterDrawer from "@/components/filterDrawer";

export default function Timeline() {
    const [characterFilter, setCharacterFilter] = useState<
        "all" | "players" | "npc"
    >("all");
    const [tagFilter, setTagFilter] = useState<string[]>([]);
    const [themeFilter, setThemeFilter] = useState<string[]>([]);

    return (
        <div className="w-full">
            <div className="grid grid-cols-[250px_1fr] min-h-screen"></div>
        </div>
    );
}
