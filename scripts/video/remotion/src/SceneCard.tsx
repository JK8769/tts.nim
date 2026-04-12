import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { moodColors } from "./types";

export const SceneCard: React.FC<{
  text: string;
  setting?: string;
  mood?: string;
  characters?: string[];
  durationInFrames: number;
}> = ({ text, setting, mood, characters, durationInFrames }) => {
  const frame = useCurrentFrame();
  const colors = moodColors[mood ?? "default"] ?? moodColors.default;

  const fadeIn = interpolate(frame, [0, 20], [0, 1], {
    extrapolateRight: "clamp",
  });
  const fadeOut = interpolate(
    frame,
    [durationInFrames - 20, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp" }
  );
  const opacity = Math.min(fadeIn, fadeOut);

  const textReveal = interpolate(frame, [5, 40], [0, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: colors.bg,
        justifyContent: "center",
        alignItems: "center",
        opacity,
      }}
    >
      {setting && (
        <div
          style={{
            position: "absolute",
            top: 60,
            left: 80,
            color: colors.accent,
            fontSize: 28,
            fontFamily: "Georgia, serif",
            fontStyle: "italic",
            opacity: interpolate(frame, [0, 15], [0, 0.7], {
              extrapolateRight: "clamp",
            }),
          }}
        >
          {setting}
        </div>
      )}

      <div
        style={{
          maxWidth: "75%",
          textAlign: "center",
          opacity: textReveal,
          transform: `translateY(${interpolate(frame, [5, 40], [20, 0], { extrapolateRight: "clamp" })}px)`,
        }}
      >
        <p
          style={{
            color: "#e0e0e0",
            fontSize: 42,
            fontFamily: "Georgia, serif",
            fontStyle: "italic",
            lineHeight: 1.6,
            margin: 0,
          }}
        >
          {text}
        </p>
      </div>

      {characters && characters.length > 0 && (
        <div
          style={{
            position: "absolute",
            bottom: 60,
            display: "flex",
            gap: 20,
            opacity: interpolate(frame, [10, 30], [0, 0.6], {
              extrapolateRight: "clamp",
            }),
          }}
        >
          {characters.map((c) => (
            <span
              key={c}
              style={{
                color: colors.accent,
                fontSize: 22,
                fontFamily: "sans-serif",
                padding: "4px 16px",
                border: `1px solid ${colors.accent}40`,
                borderRadius: 20,
              }}
            >
              {c}
            </span>
          ))}
        </div>
      )}
    </AbsoluteFill>
  );
};
