import React, { useRef, useEffect } from "react";
import { createRoot } from "react-dom/client";

const MorphOrb: React.FC = () => {
  const hostRef = useRef<HTMLDivElement>(null);
  const shadowRef = useRef<ShadowRoot | null>(null);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    if (!shadowRef.current) {
      shadowRef.current = host.attachShadow({ mode: "open" });
    }
    const shadow = shadowRef.current;

    shadow.innerHTML = `
<style>
.loader {
  --color-one: #ffbf48;
  --color-two: #be4a1d;
  --color-three: #ffbf4780;
  --color-four: #bf4a1d80;
  --color-five: #ffbf4740;
  --time-animation: 2s;
  --size: 1.5;
  position: relative;
  border-radius: 50%;
  transform: scale(var(--size));
  box-shadow:
    0 0 25px 0 var(--color-three),
    0 20px 50px 0 var(--color-four);
  animation: colorize calc(var(--time-animation) * 3) ease-in-out infinite;
  will-change: transform, filter;
  backface-visibility: hidden;
}
.loader::before {
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
  z-index: 1;
  pointer-events: none;
}
.loader svg {
  display: block;
  width: 100px; height: 100px;
  overflow: visible;
  shape-rendering: geometricPrecision;
  transform: translateZ(0);
}
.loader svg #orb-polys polygon:nth-child(1) { transform-origin: 75% 25%; transform: rotate(90deg); }
.loader svg #orb-polys polygon:nth-child(2) { transform-origin: 50% 50%; animation: rotation var(--time-animation) linear infinite reverse; }
.loader svg #orb-polys polygon:nth-child(3) { transform-origin: 50% 60%; animation: rotation var(--time-animation) linear infinite; animation-delay: calc(var(--time-animation) / -3); }
.loader svg #orb-polys polygon:nth-child(4) { transform-origin: 40% 40%; animation: rotation var(--time-animation) linear infinite reverse; }
.loader svg #orb-polys polygon:nth-child(5) { transform-origin: 40% 40%; animation: rotation var(--time-animation) linear infinite reverse; animation-delay: calc(var(--time-animation) / -2); }
.loader svg #orb-polys polygon:nth-child(6) { transform-origin: 60% 40%; animation: rotation var(--time-animation) linear infinite; }
.loader svg #orb-polys polygon:nth-child(7) { transform-origin: 60% 40%; animation: rotation var(--time-animation) linear infinite; animation-delay: calc(var(--time-animation) / -1.5); }

@keyframes rotation { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
@keyframes colorize { 0% { filter: hue-rotate(0deg); } 20% { filter: hue-rotate(-30deg); } 40% { filter: hue-rotate(-60deg); } 60% { filter: hue-rotate(-90deg); } 80% { filter: hue-rotate(-45deg); } 100% { filter: hue-rotate(0deg); } }
</style>

<div class="loader">
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
    <defs>
      <filter id="orb-goo" color-interpolation-filters="sRGB">
        <feGaussianBlur in="SourceGraphic" stdDeviation="5" result="blur" />
        <feColorMatrix in="blur" type="matrix"
          values="1 0 0 0 0
                  0 1 0 0 0
                  0 0 1 0 0
                  0 0 0 18 -7" />
      </filter>
      <mask id="orb-mask" style="mask-type: alpha">
        <g id="orb-polys" filter="url(#orb-goo)">
          <polygon points="25,25 75,25 50,75" fill="white" />
          <polygon points="50,25 75,75 25,75" fill="white" />
          <polygon points="35,35 65,35 50,65" fill="white" />
          <polygon points="35,35 65,35 50,65" fill="white" />
          <polygon points="35,35 65,35 50,65" fill="white" />
          <polygon points="35,35 65,35 50,65" fill="white" />
          <polygon points="35,35 65,35 50,65" fill="white" />
        </g>
      </mask>
      <linearGradient id="orb-grad" x1="0" y1="0" x2="0" y2="1">
        <stop offset="30%" stop-color="#ffbf48" />
        <stop offset="70%" stop-color="#be4a1d" />
      </linearGradient>
    </defs>
    <rect x="0" y="0" width="100" height="100"
      fill="url(#orb-grad)" mask="url(#orb-mask)"
      style="transform-box: fill-box;" />
  </svg>
</div>
`;
  }, []);

  return <div ref={hostRef} />;
};

const root = createRoot(document.getElementById("root")!);
root.render(
  <div style={{ width: "100vw", height: "100vh", background: "#0a0a0f", display: "flex", alignItems: "center", justifyContent: "center" }}>
    <MorphOrb />
  </div>
);
