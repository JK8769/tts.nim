import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";

export const ChapterTitle: React.FC<{
  text: string;
  durationInFrames: number;
}> = ({ text, durationInFrames }) => {
  const frame = useCurrentFrame();

  const fadeIn = interpolate(frame, [0, 25], [0, 1], {
    extrapolateRight: "clamp",
  });
  const fadeOut = interpolate(
    frame,
    [durationInFrames - 25, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp" }
  );
  const opacity = Math.min(fadeIn, fadeOut);

  const scale = interpolate(frame, [0, 30], [0.9, 1], {
    extrapolateRight: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        backgroundColor: "#000000",
        justifyContent: "center",
        alignItems: "center",
        opacity,
      }}
    >
      <div
        style={{
          transform: `scale(${scale})`,
          textAlign: "center",
        }}
      >
        <h1
          style={{
            color: "#ffffff",
            fontSize: 64,
            fontFamily: "Georgia, serif",
            fontWeight: 400,
            letterSpacing: 4,
            margin: 0,
          }}
        >
          {text}
        </h1>
        <div
          style={{
            width: 120,
            height: 2,
            backgroundColor: "#ffffff40",
            margin: "20px auto 0",
          }}
        />
      </div>
    </AbsoluteFill>
  );
};
