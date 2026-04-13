import { AbsoluteFill, Sequence, useCurrentFrame, useVideoConfig } from "remotion";
import type { ScriptData, TimelineEntry } from "./types";
import { RadioStudio } from "./RadioStudio";
import { DialogueSubtitle } from "./DialogueSubtitle";
import { ChapterTitle } from "./ChapterTitle";
import { SceneCard } from "./SceneCard";

const FPS = 30;

/** Timeline entry with its own audio URL (for live streaming). */
export interface LiveEntry extends TimelineEntry {
  audioUrl?: string;
}

export interface LiveData {
  header: ScriptData["header"];
  lines: ScriptData["lines"];
  timeline: LiveEntry[];
  audioDuration: number;
}

/**
 * Live version of Storybook — each line has its own audio track
 * instead of a single pre-rendered audio.mp3.
 * Used by the Remotion Player for real-time streaming.
 */
export const LiveStorybook: React.FC<{
  data: LiveData;
}> = ({ data }) => {
  if (!data) return null;
  const { header, lines, timeline, audioDuration } = data;

  // Chapter titles from lines array
  const chapterSequences: { text: string; startFrame: number }[] = [];
  if (lines) {
    for (let i = 0; i < lines.length; i++) {
      if (lines[i]?.type !== "chapter") continue;
      const nextEntry = timeline?.find((t) => t.index > i);
      if (nextEntry) {
        const startFrame = Math.max(0, Math.round(nextEntry.start * FPS) - 60);
        chapterSequences.push({ text: lines[i].text ?? "", startFrame });
      }
    }
  }

  return (
    <AbsoluteFill>
      {/* Radio studio background — no equalizer in live mode */}
      <RadioStudio timeline={timeline ?? []} title={header?.title ?? ""} />

      {/* Audio is played externally via Web Audio API — not through Remotion */}

      {/* Title card */}
      <Sequence from={0} durationInFrames={45}>
        <ChapterTitle text={header?.title ?? ""} durationInFrames={45} />
      </Sequence>

      {/* Chapter titles */}
      {chapterSequences.map((ch, i) => (
        <Sequence key={`ch-${i}`} from={ch.startFrame} durationInFrames={60}>
          <ChapterTitle text={ch.text} durationInFrames={60} />
        </Sequence>
      ))}

      {/* Timeline: scenes + dialogue */}
      {(timeline ?? []).map((entry, i) => {
        const startFrame = Math.round(entry.start * FPS);
        const durationFrames = Math.max(Math.round(entry.duration * FPS), FPS);

        if (entry.type === "scene") {
          return (
            <Sequence key={`s-${i}`} from={startFrame} durationInFrames={durationFrames}>
              <SceneCard
                text={entry.text ?? ""}
                setting={entry.setting}
                mood={entry.mood}
                characters={entry.characters}
                durationInFrames={durationFrames}
              />
            </Sequence>
          );
        }

        return (
          <Sequence key={`l-${i}`} from={startFrame} durationInFrames={durationFrames}>
            <DialogueSubtitle
              name={entry.name ?? ""}
              text={entry.text ?? ""}
              durationInFrames={durationFrames}
            />
          </Sequence>
        );
      })}
    </AbsoluteFill>
  );
};
