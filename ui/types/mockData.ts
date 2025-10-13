import { Session, Location, Character, CampaignTags, Event, Campaign, Landmark } from "@/types/types"

// Example mock Characters
export const characters: Character[] = [
  {
    id: 1,
    name: "Thalindra Bob",
    class: "Rogue",
    race: "Half-Elf",
    armourClass: 16,
    npc: false,
    enemy: false,
    hp: 1,
    str: 1,
    dex: 1,
    con: 1,
    int: 1,
    wis: 1,
    cha: 1,
    img: "",
    currentLocationID: 1,
    lastAppearanceID: 1,
  },
  {
    id: 2,
    name: "Brom Ironfist",
    class: "Cleric",
    race: "Dwarf",
    armourClass: 18,
    npc: true,
    enemy: false,
    hp: 1,
    str: 1,
    dex: 1,
    con: 1,
    int: 1,
    wis: 1,
    cha: 1,
    img: "",
    currentLocationID: 1,
    lastAppearanceID: 1,
  },
  {
    id: 3,
    name: "Seraphina Dawnsworn",
    class: "Paladin",
    race: "Human",
    armourClass: 20,
    npc: true,
    enemy: false,
    hp: 1,
    str: 1,
    dex: 1,
    con: 1,
    int: 1,
    wis: 1,
    cha: 1,
    img: "",
    currentLocationID: 2,
    lastAppearanceID: 2,
  },
  {
    id: 4,
    name: "Zyren the Ashborn",
    class: "Warlock",
    race: "Tiefling",
    armourClass: 14,
    npc: false,
    enemy: false,
    hp: 1,
    str: 1,
    dex: 1,
    con: 1,
    int: 1,
    wis: 1,
    cha: 1,
    img: "",
    currentLocationID: 2,
    lastAppearanceID: 2,
  },
]

// Example mock Locations
export const locations: Location[] = [
  {
    id: 1,
    name: "Neverwinter",
    description: "A bustling city known for trade, politics, and hidden intrigue.",
    type: "city",
    npcs: [2], // Brom Ironfist
    enemyIDs: [],
    landmarkIDs: [1, 2],
    tags: ["npc interaction", "investigating", "world expansion"],
  },
  {
    id: 2,
    name: "Cragmaw Hideout",
    description: "A small cave system serving as a goblin outpost.",
    type: "dungeon",
    npcs: [],
    enemyIDs: [],
    landmarkIDs: [3],
    tags: ["combat", "exploring"],
  },
]

// Example mock Landmarks
export const landmarks: Landmark[] = [
  {
    id: 1,
    locationID: 1,
    name: "Hall of Justice",
    description: "A grand temple dedicated to Tyr, god of justice.",
    type: "building",
  },
  {
    id: 2,
    locationID: 1,
    name: "Blacklake District",
    description: "Once wealthy, now decayed and full of crime.",
    type: "cultural",
  },
  {
    id: 3,
    locationID: 2,
    name: "Underground Stream",
    description: "A narrow but strong current running through the hideout.",
    type: "natural",
  },
]

// Example mock Events
export const events: Event[] = [
  {
    id: 1,
    sessionId: 1,
    summary: "The party arrives in Neverwinter and meets Brom Ironfist.",
    characterIDs: [1, 2], // Thalindra + Brom
    locationIDs: [1], // Neverwinter
    tags: ["npc interaction", "world expansion"],
    themes: ["introduction"],
  },
  {
    id: 2,
    sessionId: 2,
    summary: "The party investigates Cragmaw Hideout and fights goblins.",
    characterIDs: [3, 4], // Seraphina + Zyren
    locationIDs: [2], // Cragmaw Hideout
    tags: ["combat", "exploring"],
    themes: ["danger", "teamwork"],
  },
]

// Example mock Sessions
export const sessions: Session[] = [
  {
    id: 1,
    campaignID: 1,
    summary: "Session 1: The party enters Neverwinter and meets allies.",
    timestamp: new Date("2025-09-01T19:00:00"),
  },
  {
    id: 2,
    campaignID: 1,
    summary: "Session 2: The party travels to Cragmaw Hideout and battles goblins.",
    timestamp: new Date("2025-09-08T19:00:00"),
  },
]

// Example mock Campaign
export const campaigns: Campaign[] = [
  {
    id: 1,
    title: "The Lost Relics",
    summary: "A campaign where adventurers uncover relics of a forgotten age.",
    startedAt: new Date("2025-09-01T18:00:00"),
    completedStatus: false,
    archived: false,
  },
]