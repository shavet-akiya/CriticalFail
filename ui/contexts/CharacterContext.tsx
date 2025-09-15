"use client";
import { createContext, useContext, ReactNode, useState } from "react";

type Character = {
    id: number;
    name: string;
    class: string;
    race: string;
    armourClass: number;
    npc: boolean;
    hp: number;
    str: number;
    dex: number;
    con: number;
    int: number;
    wis: number;
    cha: number;
};

interface CharacterContextType {
    currentCharacter: Character | null;
    setCurrentCharacter: (c: Character) => void;
}

const CharacterContext = createContext<CharacterContextType | undefined>(undefined);

export function CharacterProvider({ children }: { children: ReactNode }) {
    const [currentCharacter, setCurrentCharacter] = useState<Character | null>(null);
    return (
        <CharacterContext.Provider value={{ currentCharacter, setCurrentCharacter }}>
            {children}
        </CharacterContext.Provider>
    );
}

export function useCharacter() {
    const context = useContext(CharacterContext);
    if (!context) throw new Error("useCharacter must be used within CharacterProvider");
    return context;
}
