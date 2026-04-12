import { AbsoluteFill, Audio, Sequence, staticFile } from "remotion";
import type { ScriptData } from "./types";
import { RadioStudio } from "./RadioStudio";
import { DialogueSubtitle } from "./DialogueSubtitle";
import { ChapterTitle } from "./ChapterTitle";
import { SceneCard } from "./SceneCard";

const FPS = 30;

export const Storybook: React.FC<{
  data: ScriptData;
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
      {/* Radio studio background with speaker panels + equalizer */}
      <RadioStudio timeline={timeline ?? []} title={header?.title ?? ""} />

      {/* Audio track */}
      <Audio src={staticFile("audio.mp3")} />

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
