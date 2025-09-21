"use client";
import { createContext, useContext, useState, ReactNode } from "react";
import { CharacterFilter } from "@/types/types"

interface FilterContextType {
    filter: CharacterFilter;
    setFilter: (filter: CharacterFilter) => void;
}

const FilterContext = createContext<FilterContextType | undefined>(undefined);

export function FilterProvider({ children }: { children: ReactNode }) {
    const [filter, setFilter] = useState<CharacterFilter>("all");
    return (
        <FilterContext.Provider value={{ filter, setFilter }}>
            {children}
        </FilterContext.Provider>
    );
}

export function useFilter() {
    const context = useContext(FilterContext);
    if (!context) throw new Error("useFilter must be used within FilterProvider");
    return context;
}
