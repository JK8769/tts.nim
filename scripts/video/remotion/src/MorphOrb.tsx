import { useCurrentFrame, useVideoConfig } from "remotion";

/**
 * Original CSS morphing orb from uiverse.io/andrew-manzyk/young-walrus-64
 * All animations paused and seeked via negative animation-delay
 * to sync with Remotion's frame clock.
 */

function colorShades(hex: string): { light: string; dark: string; lightA: string; darkA: string; lightA2: string } {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return {
    light: `rgb(${Math.min(255, r + 40)}, ${Math.min(255, g + 40)}, ${Math.min(255, b + 40)})`,
    dark: `rgb(${Math.max(0, r - 30)}, ${Math.max(0, g - 30)}, ${Math.max(0, b - 30)})`,
    lightA: `rgba(${Math.min(255, r + 40)}, ${Math.min(255, g + 40)}, ${Math.min(255, b + 40)}, 0.5)`,
    darkA: `rgba(${Math.max(0, r - 30)}, ${Math.max(0, g - 30)}, ${Math.max(0, b - 30)}, 0.5)`,
    lightA2: `rgba(${Math.min(255, r + 40)}, ${Math.min(255, g + 40)}, ${Math.min(255, b + 40)}, 0.25)`,
  };
}

export const MorphOrb: React.FC<{
  size?: number;
  x?: number;
  y?: number;
  speakerColor?: string;
}> = ({ size = 2.5, x = 960, y = 540, speakerColor = "#8b949e" }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps; // current time in seconds
  const { light, dark, lightA, darkA, lightA2 } = colorShades(speakerColor);
  const time = 2; // animation cycle in seconds

  // All animations: paused + negative delay to seek to correct time
  const css = `
    .morb-loader {
      --color-one: ${light};
      --color-two: ${dark};
      --color-three: ${lightA};
      --color-four: ${darkA};
      --color-five: ${lightA2};
      position: relative;
      border-radius: 50%;
      transform: scale(${size});
      box-shadow:
        0 0 25px 0 var(--color-three),
        0 20px 50px 0 var(--color-four);
      animation: morb-colorize ${time * 3}s ease-in-out infinite;
      animation-play-state: paused;
      animation-delay: -${t}s;
    }

    .morb-loader::before {
      content: "";
      position: absolute;
      top: 0; left: 0;
      width: 100px; height: 100px;
      border-radius: 50%;
      border-top: solid 1px var(--color-one);
      border-bottom: solid 1px var(--color-two);
      background: linear-gradient(180deg, var(--color-five), var(--color-four));
      box-shadow:
        inset 0 10px 10px 0 var(--color-three),
        inset 0 -10px 10px 0 var(--color-four);
    }

    .morb-loader .morb-box {
      width: 100px; height: 100px;
      background: linear-gradient(180deg, var(--color-one) 30%, var(--color-two) 70%);
      mask: url(#morb-clipping);
      -webkit-mask: url(#morb-clipping);
    }

    .morb-loader svg { position: absolute; }

    .morb-loader svg #morb-clipping {
      filter: contrast(15);
      animation: morb-roundness ${time / 2}s linear infinite;
      animation-play-state: paused;
      animation-delay: -${t}s;
    }

    .morb-loader svg #morb-clipping polygon {
      filter: blur(7px);
    }

    .morb-loader svg #morb-clipping polygon:nth-child(1) {
      transform-origin: 75% 25%;
      transform: rotate(90deg);
    }
    .morb-loader svg #morb-clipping polygon:nth-child(2) {
      transform-origin: 50% 50%;
      animation: morb-rotation ${time}s linear infinite reverse;
      animation-play-state: paused;
      animation-delay: -${t}s;
    }
    .morb-loader svg #morb-clipping polygon:nth-child(3) {
      transform-origin: 50% 60%;
      animation: morb-rotation ${time}s linear infinite;
      animation-play-state: paused;
      animation-delay: -${t - time / 3}s;
    }
    .morb-loader svg #morb-clipping polygon:nth-child(4) {
      transform-origin: 40% 40%;
      animation: morb-rotation ${time}s linear infinite reverse;
      animation-play-state: paused;
      animation-delay: -${t}s;
    }
    .morb-loader svg #morb-clipping polygon:nth-child(5) {
      transform-origin: 40% 40%;
      animation: morb-rotation ${time}s linear infinite reverse;
      animation-play-state: paused;
      animation-delay: -${t - time / 2}s;
    }
    .morb-loader svg #morb-clipping polygon:nth-child(6) {
      transform-origin: 60% 40%;
      animation: morb-rotation ${time}s linear infinite;
      animation-play-state: paused;
      animation-delay: -${t}s;
    }
    .morb-loader svg #morb-clipping polygon:nth-child(7) {
      transform-origin: 60% 40%;
      animation: morb-rotation ${time}s linear infinite;
      animation-play-state: paused;
      animation-delay: -${t - time / 1.5}s;
    }

    @keyframes morb-rotation {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    @keyframes morb-roundness {
      0% { filter: contrast(15); }
      20% { filter: contrast(3); }
      40% { filter: contrast(3); }
      60% { filter: contrast(15); }
      100% { filter: contrast(15); }
    }

    @keyframes morb-colorize {
      0% { filter: hue-rotate(0deg); }
      20% { filter: hue-rotate(-30deg); }
      40% { filter: hue-rotate(-60deg); }
      60% { filter: hue-rotate(-90deg); }
      80% { filter: hue-rotate(-45deg); }
      100% { filter: hue-rotate(0deg); }
    }
  `;

  return (
    <div
      style={{
        position: "absolute",
        left: x,
        top: y,
        transform: "translate(-50%, -50%)",
      }}
    >
      <style dangerouslySetInnerHTML={{ __html: css }} />
      <div className="morb-loader">
        <svg width="100" height="100" viewBox="0 0 100 100">
          <defs>
            <mask id="morb-clipping">
              <polygon points="0,0 100,0 100,100 0,100" fill="black" />
              <polygon points="25,25 75,25 50,75" fill="white" />
              <polygon points="50,25 75,75 25,75" fill="white" />
              <polygon points="35,35 65,35 50,65" fill="white" />
              <polygon points="35,35 65,35 50,65" fill="white" />
              <polygon points="35,35 65,35 50,65" fill="white" />
              <polygon points="35,35 65,35 50,65" fill="white" />
            </mask>
          </defs>
        </svg>
        <div className="morb-box" />
      </div>
    </div>
  );
};
