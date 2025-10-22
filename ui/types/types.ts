// need to be able to edit this.
export type Character = {
  characterId: string // uuid
  name: string;
  race?: string;
  class?: string;
  npc?: boolean;

  // new stats
  AC: number;
  HP: number;
  STR: number;
  DEX: number;
  CON: number;
  INT: number;
  WIS: number;
  CHA: number;
};

export interface CampaignEvents { events: Event[]; }

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