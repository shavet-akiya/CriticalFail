// need to be able to edit this.
export type Character = {
  id: number;
  name: string;
  class: string;
  race: string;
  armourClass: number;
  npc: boolean;
  enemy: boolean; // monster/enemy --> set true
  hp: number;
  str: number;
  dex: number;
  con: number;
  int: number;
  wis: number;
  cha: number;
  img?: Base64URLString; // this may just be done via pathing if the user uploads it anyways 
  currentLocationID?: number; // may not need right now
  lastAppearanceID?: number; // session ID
};

// No monster table for now. Define monsters/enemies under Character.npc = true.

export type Event = {
  id: number;
  sessionId: number; // session ID will be linked to Campaign via Session
  summary: string;
  characterIDs: number[];
  locationIDs: number[];
  tags: CampaignTags[];
  themes: string[];
  // order in session timeline comes from ID or array position
};


export type CharacterFilter = "all" | "players" | "npc" ;

// checked with an interviewee & Oscar
export type CampaignTags= 
  "combat" | 
  "exploring" | 
  "player-to-player interaction" | 
  "npc interaction" | 
  "resting" | 
  "investigating" | 
  "world expansion" | 
  "character expansion" | 
  "lore expansion" |
  "misc";

export type Location = {
  id: number;
  name: string;
  description: string;

  // Broad category for quick filtering/searching
  type: 
    | "city"
    | "town"
    | "village"
    | "dungeon"
    | "wilderness"
    | "stronghold"
    | "ruins"
    | "other";

  npcs: number[]; //character ID for monsters/NPC
  enemyIDs: number[]; // monsters as Characters with enemy=true


  landmarkIDs: number[];

  tags: CampaignTags[];
  subLocations?: Location[];
};

export type Landmark = {
  id: number;
  locationID: number;
  name: string;
  description: string;
  type: "natural" | "building" | "infrastructure" | "cultural";
};


export type Session = {
  id: number;
  campaignID: number;
  summary: string;
  timestamp: Date;
}

export type Campaign = {
  id: number;
  title: string;
  summary: string;
  startedAt: Date;
  completedStatus: boolean;
  archived: boolean;
}

// encounter table?