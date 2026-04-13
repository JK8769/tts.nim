// Types matching our JSONL script format

export interface ScriptHeader {
  type: "header";
  title: string;
  format: string;
  cast: Record<string, string>;
  defaults: {
    pause?: number;
    speed?: number;
    narrator_voice?: string;
  };
}

export interface ScriptLine {
  type: "line" | "narration" | "scene" | "chapter" | "pause";
  text?: string;
  name?: string;
  voice?: string;
  setting?: string;
  characters?: string[];
  mood?: string;
  narrate?: boolean;
  duration?: number;
  pause?: number;
}

export interface TimelineEntry {
  index: number;
  name: string;
  voice: string;
  type?: string;
  start: number;
  duration: number;
  text: string;
  // Scene metadata (embedded by prep)
  setting?: string;
  characters?: string[];
  mood?: string;
}

export interface ScriptData {
  header: ScriptHeader;
  lines: ScriptLine[];
  timeline: TimelineEntry[];
  audioDuration: number;
}

// Generate a stable color from a string
export function nameToColor(name: string): string {
  const colors = [
    "#ff6b9d", "#64b5f6", "#ce93d8", "#69f0ae", "#ffab40",
    "#4dd0e1", "#ff8a65", "#aed581", "#f48fb1", "#80deea",
  ];
  // FNV-1a hash — better distribution for short strings
  let hash = 2166136261;
  for (let i = 0; i < name.length; i++) {
    hash ^= name.charCodeAt(i);
    hash = (hash * 16777619) >>> 0;
  }
  return colors[hash % colors.length];
}

// Mood → color/style mapping
export const moodColors: Record<string, { bg: string; accent: string }> = {
  excited: { bg: "#1a0a2e", accent: "#ff6b9d" },
  casual: { bg: "#0a1628", accent: "#64b5f6" },
  cool: { bg: "#0d1117", accent: "#7c8aff" },
  warm: { bg: "#1a0f0a", accent: "#ffab40" },
  intimidating: { bg: "#1a0a0a", accent: "#ff5252" },
  curious: { bg: "#0a1a1a", accent: "#4dd0e1" },
  playful: { bg: "#1a0a20", accent: "#ce93d8" },
  chaotic: { bg: "#1a1a0a", accent: "#ffd740" },
  anticipation: { bg: "#0a1a10", accent: "#69f0ae" },
  default: { bg: "#0d1117", accent: "#8b949e" },
};
