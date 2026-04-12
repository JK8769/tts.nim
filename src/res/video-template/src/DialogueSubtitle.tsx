import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { nameToColor } from "./types";

export const DialogueSubtitle: React.FC<{
  name: string;
  text: string;
  durationInFrames: number;
}> = ({ name, text, durationInFrames }) => {
  const frame = useCurrentFrame();
  const color = nameToColor(name ?? "unknown");

  const fadeIn = interpolate(frame, [0, 2], [0, 1], {
    extrapolateRight: "clamp",
  });
  const fadeOut = interpolate(
    frame,
    [durationInFrames - 3, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp" }
  );
  const opacity = Math.min(fadeIn, fadeOut);

  return (
    <AbsoluteFill
      style={{
        justifyContent: "flex-end",
        alignItems: "center",
        paddingBottom: 80,
      }}
    >
      <div
        style={{
          opacity,
          maxWidth: "80%",
          textAlign: "center",
        }}
      >
        <div
          style={{
            color,
            fontSize: 26,
            fontFamily: "sans-serif",
            fontWeight: 700,
            marginBottom: 8,
            textTransform: "uppercase",
            letterSpacing: 2,
          }}
        >
          {name}
        </div>
        <div
          style={{
            color: "#ffffff",
            fontSize: 36,
            fontFamily: "Georgia, serif",
            lineHeight: 1.5,
            textShadow: "0 2px 8px rgba(0,0,0,0.8)",
            padding: "12px 24px",
            backgroundColor: "rgba(0,0,0,0.5)",
            borderRadius: 12,
            borderLeft: `3px solid ${color}`,
          }}
        >
          {text}
        </div>
      </div>
    </AbsoluteFill>
  );
};
