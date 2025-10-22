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


export interface SessionCharacter {
    name: string;
}

export interface SessionLocation {
    name?: string;
    location_name?: string;
}

export interface SessionMetadata {
    session_id: string;
    campaign_id?: string;
    characters?: SessionCharacter[];
    locations?: SessionLocation[];
    events?: Event[];
}

export interface Character {
    character_id: string;
    name: string;
    class: string;
    race?: string;
    HP: number;
    AC: number;
    npc?: boolean;
    STR: number;
    DEX: number;
    CON: number;
    INT: number;
    WIS: number;
    CHA: number;
}

export interface Location {
    location_id: string;
    location_name: string;
    description: string;
}

export interface Event {
    event_id: string;
    event: string;
    event_summary: string;
    participants?: string;
    location?: string;
    event_tags?: string;
}

export interface Session {
    session_id: string;
    processed_at?: string;
    characters?: Character[];
    locations?: Location[];
    events?: Event[];
}

export interface SessionCardProps {
    session: Session;
    formatSessionDate: (id: string) => string;
}

export interface Campaign {
    campaign_id: string;
    campaign_name: string;
    session_ids: string[];
    campaign_image_url?: string;
}