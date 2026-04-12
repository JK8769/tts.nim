import {
  AbsoluteFill,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  Audio,
  staticFile,
} from "remotion";
import { useAudioData, visualizeAudio } from "@remotion/media-utils";
import type { TimelineEntry } from "./types";
import { nameToColor, moodColors } from "./types";
import { MorphOrb } from "./MorphOrb";

const PANEL_Y = 180;
const BAR_COUNT = 64;

// Find which timeline entry is active at a given time
function getActiveSpeaker(
  timeline: TimelineEntry[],
  timeSec: number
): TimelineEntry | null {
  for (let i = timeline.length - 1; i >= 0; i--) {
    const e = timeline[i];
    if (timeSec >= e.start && timeSec < e.start + e.duration) {
      return e;
    }
  }
  return null;
}

// Get unique cast members from timeline
function getCast(timeline: TimelineEntry[]): string[] {
  const seen = new Set<string>();
  const names: string[] = [];
  for (const e of timeline) {
    const n = e.name;
    if (n && n !== "narrator" && !seen.has(n)) {
      seen.add(n);
      names.push(n);
    }
  }
  return names;
}

// Speaker panel circle
const SpeakerCircle: React.FC<{
  name: string;
  isActive: boolean;
  x: number;
  y: number;
}> = ({ name, isActive, x, y }) => {
  const frame = useCurrentFrame();
  const color = nameToColor(name);

  const pulse = isActive
    ? 1 + 0.04 * Math.sin(frame * 0.3)
    : 1;
  const glowOpacity = isActive ? 0.8 : 0;
  const circleOpacity = isActive ? 1 : 0.35;

  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y,
        transform: `translate(-50%, -50%) scale(${pulse})`,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 12,
      }}
    >
      {/* Glow ring */}
      <div
        style={{
          width: 110,
          height: 110,
          borderRadius: "50%",
          border: `3px solid ${color}`,
          opacity: circleOpacity,
          boxShadow: isActive
            ? `0 0 30px ${color}80, 0 0 60px ${color}40, inset 0 0 20px ${color}20`
            : "none",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: isActive ? `${color}15` : `${color}08`,
          transition: "all 0.2s ease",
        }}
      >
        {/* Initial letter */}
        <span
          style={{
            fontSize: 44,
            fontWeight: 700,
            color: isActive ? color : `${color}90`,
            fontFamily: "sans-serif",
          }}
        >
          {name[0]}
        </span>
      </div>
      {/* Name label */}
      <span
        style={{
          fontSize: 18,
          fontWeight: isActive ? 700 : 400,
          color: isActive ? color : "#666",
          fontFamily: "sans-serif",
          letterSpacing: 1,
          textTransform: "uppercase",
        }}
      >
        {name}
      </span>
    </div>
  );
};

// ON AIR indicator
const OnAirBadge: React.FC = () => {
  const frame = useCurrentFrame();
  const blink = Math.sin(frame * 0.08) > 0 ? 1 : 0.6;

  return (
    <div
      style={{
        position: "absolute",
        top: 40,
        left: 60,
        display: "flex",
        alignItems: "center",
        gap: 12,
        opacity: blink,
      }}
    >
      <div
        style={{
          width: 14,
          height: 14,
          borderRadius: "50%",
          backgroundColor: "#ff3333",
          boxShadow: "0 0 12px #ff3333, 0 0 24px #ff333380",
        }}
      />
      <span
        style={{
          fontSize: 22,
          fontWeight: 700,
          color: "#ff3333",
          fontFamily: "sans-serif",
          letterSpacing: 4,
        }}
      >
        ON AIR
      </span>
    </div>
  );
};

// Audio equalizer bars
const Equalizer: React.FC<{
  audioSrc: string;
  speakerColor: string;
}> = ({ audioSrc, speakerColor }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const audioData = useAudioData(audioSrc);

  if (!audioData) return null;

  const visualization = visualizeAudio({
    fps,
    frame,
    audioData,
    numberOfSamples: BAR_COUNT * 2, // oversample for smoother look
  });

  // Take first BAR_COUNT samples, mirror them
  const bars = visualization.slice(0, BAR_COUNT);
  const mirrored = [...bars].reverse();
  const allBars = [...mirrored, ...bars];

  return (
    <div
      style={{
        position: "absolute",
        bottom: 0,
        left: 0,
        right: 0,
        height: 200,
        display: "flex",
        alignItems: "flex-end",
        justifyContent: "center",
        gap: 2,
        padding: "0 80px",
      }}
    >
      {allBars.map((amp, i) => {
        const height = Math.max(3, amp * 180);
        const dist = Math.abs(i - allBars.length / 2) / (allBars.length / 2);
        const opacity = interpolate(dist, [0, 1], [0.9, 0.3]);
        return (
          <div
            key={i}
            style={{
              width: Math.max(2, (1920 - 160) / allBars.length - 2),
              height,
              backgroundColor: speakerColor,
              opacity,
              borderRadius: 2,
              boxShadow: amp > 0.3 ? `0 0 8px ${speakerColor}60` : "none",
            }}
          />
        );
      })}
    </div>
  );
};

// Title / show name
const ShowTitle: React.FC<{ title: string }> = ({ title }) => {
  return (
    <div
      style={{
        position: "absolute",
        top: 38,
        right: 60,
        fontSize: 20,
        color: "#444",
        fontFamily: "sans-serif",
        fontWeight: 300,
        letterSpacing: 2,
        textTransform: "uppercase",
      }}
    >
      {title}
    </div>
  );
};

export const RadioStudio: React.FC<{
  timeline: TimelineEntry[];
  title: string;
}> = ({ timeline, title }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const timeSec = frame / fps;

  const cast = getCast(timeline);
  const active = getActiveSpeaker(timeline, timeSec);
  const activeName = active?.name ?? "";
  const activeColor = activeName ? nameToColor(activeName) : "#8b949e";

  // Find current mood from the most recent scene entry
  let currentMood = "default";
  for (const entry of timeline) {
    if (entry.type === "scene" && entry.mood && timeSec >= entry.start) {
      currentMood = entry.mood;
    }
  }
  const moodColor = (moodColors[currentMood] ?? moodColors.default).accent;

  // Distribute cast panels evenly
  const panelWidth = 1920 / (cast.length + 1);

  return (
    <AbsoluteFill
      style={{
        background: "linear-gradient(180deg, #0a0a0f 0%, #0f1018 40%, #111320 100%)",
      }}
    >
      {/* Subtle grid lines */}
      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage:
            "linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px), " +
            "linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)",
          backgroundSize: "60px 60px",
        }}
      />

      {/* Morphing orb — centered */}
      <MorphOrb size={3.2} x={960} y={540} speakerColor={moodColor} />

      <OnAirBadge />
      <ShowTitle title={title} />

      {/* Speaker panels */}
      {cast.map((name, i) => (
        <SpeakerCircle
          key={name}
          name={name}
          isActive={activeName === name}
          x={panelWidth * (i + 1)}
          y={PANEL_Y}
        />
      ))}

      {/* Horizontal divider */}
      <div
        style={{
          position: "absolute",
          top: PANEL_Y + 90,
          left: 80,
          right: 80,
          height: 1,
          background: "linear-gradient(90deg, transparent, #ffffff10, transparent)",
        }}
      />

      {/* Equalizer */}
      <Equalizer audioSrc={staticFile("audio.mp3")} speakerColor={activeColor} />
    </AbsoluteFill>
  );
};
