import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import type { ScriptLine, TimelineEntry, ScriptData } from "./types";
import { nameToColor, moodColors } from "./types";

/** Timeline entry with its own audio URL (for live streaming). */
interface LiveEntry extends TimelineEntry {
  audioUrl?: string;
}

// ---- Types ----

type WSMessage =
  | { type: "init"; header: ScriptData["header"] }
  | { type: "line"; entry: LiveEntry; scriptLine?: ScriptLine }
  | { type: "chapter"; text: string; index: number }
  | { type: "scene"; entry: LiveEntry; scriptLine?: ScriptLine }
  | { type: "music"; action: string; url?: string; volume?: number; loop?: boolean; fade_ms?: number; start_at?: number; curve?: string | [number, number, number, number] }
  | { type: "sfx"; url: string; volume?: number }
  | { type: "cast"; cast: Record<string, string> }
  | { type: "stop" };

const emptyHeader: ScriptData["header"] = {
  type: "header", title: "", format: "show", cast: {}, defaults: {},
};

// ---- Audio queue ----

class AudioQueue {
  ctx: AudioContext | null = null;
  private speechGain: GainNode | null = null;
  private queue: string[] = [];
  private playing = false;
  onStart?: (url: string) => void;
  onEnd?: (url: string) => void;
  onSpeechStart?: () => void;
  onSpeechEnd?: () => void;

  ensure() {
    if (!this.ctx) {
      this.ctx = new AudioContext();
      this.speechGain = this.ctx.createGain();
      this.speechGain.connect(this.ctx.destination);
    }
    if (this.ctx.state === "suspended") this.ctx.resume();
  }

  play(url: string) {
    this.ensure();
    this.queue.push(url);
    if (!this.playing) this.pump();
  }

  private async pump() {
    if (!this.ctx || !this.speechGain || this.queue.length === 0) {
      this.playing = false;
      return;
    }
    this.playing = true;
    const url = this.queue.shift()!;
    this.onStart?.(url);
    this.onSpeechStart?.();
    try {
      const resp = await fetch(url);
      const buf = await resp.arrayBuffer();
      const audio = await this.ctx.decodeAudioData(buf);
      const src = this.ctx.createBufferSource();
      src.buffer = audio;
      src.connect(this.speechGain);
      await new Promise<void>((done) => {
        src.onended = () => done();
        src.start();
      });
    } catch (e) {
      console.error("Audio error:", e);
    }
    this.onSpeechEnd?.();
    this.onEnd?.(url);
    await new Promise((r) => setTimeout(r, 200));
    this.pump();
  }
}

// ---- Background music player with auto-ducking ----

/**
 * Solve cubic bezier at parameter t for one axis.
 * Control points: P0=0, P1=cp1, P2=cp2, P3=1
 */
function cubicBezier1D(t: number, cp1: number, cp2: number): number {
  const u = 1 - t;
  return 3 * u * u * t * cp1 + 3 * u * t * t * cp2 + t * t * t;
}

/** Find the bezier parameter t for a given x using Newton's method. */
function bezierTForX(x: number, x1: number, x2: number): number {
  let t = x;
  for (let i = 0; i < 8; i++) {
    const cx = cubicBezier1D(t, x1, x2) - x;
    if (Math.abs(cx) < 1e-6) break;
    // derivative: 3(1-t)^2*x1 + 6(1-t)t*x2 - 6(1-t)t*x1 + 3t^2*(1-x2) ... simplified:
    const u = 1 - t;
    const dx = 3 * u * u * x1 + 6 * u * t * (x2 - x1) + 3 * t * t * (1 - x2);
    if (Math.abs(dx) < 1e-6) break;
    t -= cx / dx;
    t = Math.max(0, Math.min(1, t));
  }
  return t;
}

/**
 * Generate a gain curve (Float32Array) from cubic-bezier control points.
 * Like CSS cubic-bezier(x1,y1,x2,y2) — maps normalized time [0,1] to value [from,to].
 */
function bezierCurve(
  steps: number, from: number, to: number,
  x1: number, y1: number, x2: number, y2: number
): Float32Array {
  const curve = new Float32Array(steps);
  for (let i = 0; i < steps; i++) {
    const x = i / (steps - 1);
    const t = bezierTForX(x, x1, x2);
    const y = cubicBezier1D(t, y1, y2);
    curve[i] = from + (to - from) * y;
  }
  return curve;
}

/**
 * Generate a linked crossfade pair from one curve.
 * The curve defines the incoming deck (0→inVol).
 * The outgoing deck is the complement (outVol→0), so at any time t:
 *   gainIn(t) + gainOut(t) = volume
 */
function crossfadePair(
  steps: number, outVol: number, inVol: number,
  x1: number, y1: number, x2: number, y2: number
): { fadeIn: Float32Array; fadeOut: Float32Array } {
  const fadeIn = new Float32Array(steps);
  const fadeOut = new Float32Array(steps);
  for (let i = 0; i < steps; i++) {
    const x = i / (steps - 1);
    const t = bezierTForX(x, x1, x2);
    const y = cubicBezier1D(t, y1, y2); // 0→1 normalized
    fadeIn[i] = inVol * y;
    fadeOut[i] = outVol * (1 - y);
  }
  return { fadeIn, fadeOut };
}

// Preset crossfade curves: [x1, y1, x2, y2]
const CURVES = {
  linear:     [0, 0, 1, 1] as const,
  ease:       [0.25, 0.1, 0.25, 1] as const,
  easeInOut:  [0.42, 0, 0.58, 1] as const,
  smooth:     [0.4, 0, 0.2, 1] as const,    // slow start, gentle landing
  cut:        [0.9, 0, 1, 0.1] as const,     // DJ hard cut
  equalPower: [0.5, 0, 0.5, 1] as const,     // equal-power-ish S-curve
};

type CurveName = keyof typeof CURVES;
type CurveSpec = CurveName | [number, number, number, number];

function resolveCurve(spec: CurveSpec): [number, number, number, number] {
  if (typeof spec === "string") return [...CURVES[spec]] as [number, number, number, number];
  return spec;
}

/** A single audio deck with its own source and gain node. */
class Deck {
  source: AudioBufferSourceNode | null = null;
  gain: GainNode;
  url = "";
  private ctx: AudioContext;
  private buffer: AudioBuffer | null = null;  // pre-loaded buffer (cued)
  cueLoop = false;
  private cueStartAt = 0;

  constructor(ctx: AudioContext) {
    this.ctx = ctx;
    this.gain = ctx.createGain();
    this.gain.gain.value = 0;
    this.gain.connect(ctx.destination);
  }

  /** Fetch, decode, and immediately start playback. */
  async load(url: string, loop: boolean, startAt: number, onEnded?: () => void) {
    this.stop();
    await this.cue(url, loop, startAt);
    this.start(onEnded);
  }

  /** Fetch and decode audio into the buffer — ready to start, but silent. */
  async cue(url: string, loop = true, startAt = 0) {
    this.stop();
    this.url = url;
    this.cueLoop = loop;
    this.cueStartAt = startAt;
    const resp = await fetch(url);
    const buf = await resp.arrayBuffer();
    this.buffer = await this.ctx.decodeAudioData(buf);
  }

  /** Start playback from a cued buffer. No-op if nothing is cued. */
  start(onEnded?: () => void) {
    if (!this.buffer) return;
    // Stop any existing source before creating a new one
    if (this.source) {
      try { this.source.stop(); } catch {}
      try { this.source.disconnect(); } catch {}
    }
    this.source = this.ctx.createBufferSource();
    this.source.buffer = this.buffer;
    this.source.loop = this.cueLoop;
    this.source.connect(this.gain);
    if (onEnded) this.source.onended = onEnded;
    this.source.start(0, this.cueStartAt);
  }

  /** Whether this deck has a track cued and ready to play. */
  get cued() { return this.buffer !== null && this.source === null; }

  /** Fade gain along a bezier curve from→to over durationMs. */
  curvedFade(from: number, to: number, durationMs: number, curve: CurveSpec = "easeInOut") {
    const [x1, y1, x2, y2] = resolveCurve(curve);
    const steps = Math.max(2, Math.round(durationMs / 10)); // ~100Hz resolution
    const values = bezierCurve(steps, from, to, x1, y1, x2, y2);
    this.applyCurve(values, durationMs);
  }

  /** Apply a pre-computed gain curve over durationMs. */
  applyCurve(values: Float32Array, durationMs: number) {
    const now = this.ctx.currentTime;
    this.gain.gain.cancelScheduledValues(now);
    this.gain.gain.setValueAtTime(values[0], now);
    this.gain.gain.setValueCurveAtTime(values, now, durationMs / 1000);
  }

  fadeTo(target: number, fadeMs: number, curve: CurveSpec = "easeInOut") {
    this.curvedFade(this.gain.gain.value, target, fadeMs, curve);
  }

  stop() {
    if (this.source) {
      try { this.source.stop(); } catch {}
      try { this.source.disconnect(); } catch {}
      this.source = null;
    }
    this.buffer = null;
    this.url = "";
    this.gain.gain.cancelScheduledValues(this.ctx.currentTime);
    this.gain.gain.value = 0;
  }

  fadeOutAndStop(fadeMs: number, curve: CurveSpec = "easeInOut") {
    if (!this.source) return;
    this.curvedFade(this.gain.gain.value, 0, fadeMs, curve);
    this.scheduleStop(fadeMs);
  }

  /** Apply a pre-computed fade-out curve and stop the source after it completes. */
  applyCurveAndStop(values: Float32Array, durationMs: number) {
    if (!this.source) return;
    this.applyCurve(values, durationMs);
    this.scheduleStop(durationMs);
  }

  private scheduleStop(fadeMs: number) {
    const src = this.source;
    this.source = null;
    this.buffer = null;
    this.url = "";
    if (!src) return;
    setTimeout(() => {
      try { src.stop(); } catch {}
      try { src.disconnect(); } catch {}
    }, fadeMs + 100);
  }

  get playing() { return this.source !== null; }
}

/**
 * Two-deck DJ mixer with cubic-bezier crossfade curves.
 *
 * One curve defines the crossfade shape. At any point t during the transition:
 *   incoming gain = target * curve(t)
 *   outgoing gain = target * (1 - curve(t))
 *
 * This guarantees gainIn + gainOut = target volume at all times,
 * just like a physical DJ crossfader.
 *
 * Use preset names ("linear", "ease", "easeInOut", "smooth", "cut", "equalPower")
 * or custom [x1,y1,x2,y2] control points like CSS cubic-bezier().
 */
class AudioMixer {
  private ctx: AudioContext;
  private deckA: Deck;
  private deckB: Deck;
  private active: "A" | "B" = "A";
  private sfxGain: GainNode;
  private baseVolume = 0.3;
  private duckVolume = 0.08;
  private ducked = false;
  private playlist: { url: string; volume?: number; loop?: boolean }[] = [];
  private defaultCurve: CurveSpec = "easeInOut";

  constructor(ctx: AudioContext) {
    this.ctx = ctx;
    this.deckA = new Deck(ctx);
    this.deckB = new Deck(ctx);
    this.sfxGain = ctx.createGain();
    this.sfxGain.gain.value = 1.0;
    this.sfxGain.connect(ctx.destination);
  }

  private get activeDeck() { return this.active === "A" ? this.deckA : this.deckB; }
  private get inactiveDeck() { return this.active === "A" ? this.deckB : this.deckA; }
  private flip() { this.active = this.active === "A" ? "B" : "A"; }

  /** Set the default crossfade curve for all transitions. */
  setCurve(curve: CurveSpec) { this.defaultCurve = curve; }

  /** Cue a track on the inactive deck — fetches, decodes, ready to crossfade. */
  async cue(url: string, loop = true, startAt = 0, volume?: number) {
    if (volume !== undefined) {
      this.baseVolume = volume;
      this.duckVolume = volume * 0.25;
    }
    try {
      await this.inactiveDeck.cue(url, loop, startAt);
    } catch (e) {
      console.error("Cue error:", e);
    }
  }

  /** Crossfade from active deck to inactive deck. Uses pre-cued track if available. */
  crossfade(fadeMs = 2000, curve?: CurveSpec, volume?: number) {
    if (volume !== undefined) {
      this.baseVolume = volume;
      this.duckVolume = volume * 0.25;
    }
    const target = this.ducked ? this.duckVolume : this.baseVolume;
    const c = curve ?? this.defaultCurve;

    if (!this.inactiveDeck.cued) {
      console.warn("Crossfade: nothing cued on inactive deck");
      return;
    }

    const outVol = this.activeDeck.playing ? this.activeDeck.gain.gain.value : 0;
    const steps = Math.max(2, Math.round(fadeMs / 10));
    const [x1, y1, x2, y2] = resolveCurve(c);
    const { fadeIn, fadeOut } = crossfadePair(steps, outVol, target, x1, y1, x2, y2);

    const outgoing = this.activeDeck;
    this.flip();
    const incoming = this.activeDeck;

    if (outgoing.playing) outgoing.applyCurveAndStop(fadeOut, fadeMs);
    const onEnded = !incoming.cueLoop && this.playlist.length > 0 ? () => this.next() : undefined;
    incoming.start(onEnded);
    incoming.applyCurve(fadeIn, fadeMs);
  }

  async play(url: string, volume = 0.3, loop = true, fadeMs = 1000, startAt = 0, curve?: CurveSpec) {
    this.baseVolume = volume;
    this.duckVolume = volume * 0.25;
    const target = this.ducked ? this.duckVolume : this.baseVolume;
    const c = curve ?? this.defaultCurve;

    if (this.activeDeck.playing) {
      // Cue on inactive deck then crossfade
      try {
        await this.inactiveDeck.cue(url, loop, startAt);
      } catch (e) {
        console.error("Music error:", e);
        return;
      }
      this.crossfade(fadeMs, c);
    } else {
      // No crossfade needed — just fade in on active deck
      const deck = this.activeDeck;
      const onEnded = !loop && this.playlist.length > 0 ? () => this.next() : undefined;
      try {
        await deck.load(url, loop, startAt, onEnded);
        deck.curvedFade(0, target, fadeMs, c);
      } catch (e) {
        console.error("Music error:", e);
      }
    }
  }

  enqueue(url: string, volume?: number, loop?: boolean) {
    this.playlist.push({ url, volume, loop });
  }

  async next(crossfadeMs = 2000, curve?: CurveSpec) {
    const item = this.playlist.shift();
    if (!item) return;
    this.baseVolume = item.volume ?? this.baseVolume;
    this.duckVolume = this.baseVolume * 0.25;
    // Cue on inactive deck then crossfade
    try {
      await this.inactiveDeck.cue(item.url, item.loop ?? false, 0);
    } catch (e) {
      console.error("Music next error:", e);
      return;
    }
    this.crossfade(crossfadeMs, curve ?? this.defaultCurve);
  }

  stop(fadeMs = 1000) {
    this.deckA.fadeOutAndStop(fadeMs, this.defaultCurve);
    this.deckB.fadeOutAndStop(fadeMs, this.defaultCurve);
  }

  fadeOut(fadeMs = 2000) {
    this.stop(fadeMs);
  }

  duck() {
    if (this.ducked) return;
    this.ducked = true;
    if (this.activeDeck.playing) this.activeDeck.fadeTo(this.duckVolume, 300);
  }

  unduck() {
    if (!this.ducked) return;
    this.ducked = false;
    if (this.activeDeck.playing) this.activeDeck.fadeTo(this.baseVolume, 500);
  }

  setVolume(volume: number, fadeMs = 1000) {
    this.baseVolume = volume;
    this.duckVolume = volume * 0.25;
    const target = this.ducked ? this.duckVolume : this.baseVolume;
    if (this.activeDeck.playing) this.activeDeck.fadeTo(target, fadeMs);
  }

  async playSfx(url: string, volume = 0.8) {
    try {
      const resp = await fetch(url);
      const buf = await resp.arrayBuffer();
      const audio = await this.ctx.decodeAudioData(buf);
      const src = this.ctx.createBufferSource();
      src.buffer = audio;
      this.sfxGain.gain.value = volume;
      src.connect(this.sfxGain);
      src.start();
    } catch (e) {
      console.error("SFX error:", e);
    }
  }
}

// ---- MorphOrb — original CSS filter approach, paused+seeked via rAF (like Remotion) ----

function colorShades(hex: string) {
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

const LiveMorphOrb: React.FC<{ color: string; x: number; y: number; size?: number }> = ({ color, x, y, size = 2.5 }) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number>(0);
  const startRef = useRef(performance.now());

  const { light, dark, lightA, darkA, lightA2 } = useMemo(() => colorShades(color), [color]);
  const time = 2; // animation cycle seconds

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    startRef.current = performance.now();

    const tick = () => {
      const t = (performance.now() - startRef.current) / 1000;
      el.dataset.t = String(t);
      // Update all animation-delay properties to seek to current time
      const style = el.querySelector(".morb-style") as HTMLStyleElement;
      if (style) {
        style.textContent = buildCSS(t);
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [light, dark, lightA, darkA, lightA2, size]);

  function buildCSS(t: number) {
    return `
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
  }

  return (
    <div ref={containerRef} style={{ position: "absolute", left: x, top: y, transform: "translate(-50%, -50%)" }}>
      <style className="morb-style" dangerouslySetInnerHTML={{ __html: buildCSS(0) }} />
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

// ---- Speaker circle ----

const SpeakerCircle: React.FC<{ name: string; isActive: boolean; x: number; y: number }> = ({ name, isActive, x, y }) => {
  const color = nameToColor(name);
  return (
    <div style={{
      position: "absolute", left: x, top: y, transform: "translate(-50%, -50%)",
      display: "flex", flexDirection: "column", alignItems: "center", gap: 12,
      transition: "transform 0.3s ease",
      ...(isActive ? { transform: `translate(-50%, -50%) scale(${1.04})` } : {}),
    }}>
      <div style={{
        width: 110, height: 110, borderRadius: "50%", border: `3px solid ${color}`,
        opacity: isActive ? 1 : 0.35,
        boxShadow: isActive ? `0 0 30px ${color}80, 0 0 60px ${color}40, inset 0 0 20px ${color}20` : "none",
        display: "flex", alignItems: "center", justifyContent: "center",
        backgroundColor: isActive ? `${color}15` : `${color}08`,
        transition: "all 0.3s ease",
      }}>
        <span style={{ fontSize: 44, fontWeight: 700, color: isActive ? color : `${color}90`, fontFamily: "sans-serif" }}>
          {name[0]}
        </span>
      </div>
      <span style={{
        fontSize: 18, fontWeight: isActive ? 700 : 400,
        color: isActive ? color : "#666", fontFamily: "sans-serif",
        letterSpacing: 1, textTransform: "uppercase", transition: "all 0.3s ease",
      }}>
        {name}
      </span>
    </div>
  );
};

// ---- On Air badge ----

const OnAirBadge: React.FC = () => (
  <div style={{ position: "absolute", top: 40, left: 60, display: "flex", alignItems: "center", gap: 12, animation: "onair-blink 2.6s ease infinite" }}>
    <style dangerouslySetInnerHTML={{ __html: `@keyframes onair-blink { 0%,100% { opacity: 1; } 50% { opacity: 0.6; } }` }} />
    <div style={{ width: 14, height: 14, borderRadius: "50%", backgroundColor: "#ff3333", boxShadow: "0 0 12px #ff3333, 0 0 24px #ff333380" }} />
    <span style={{ fontSize: 22, fontWeight: 700, color: "#ff3333", fontFamily: "sans-serif", letterSpacing: 4 }}>ON AIR</span>
  </div>
);

// ---- Subtitle ----

const Subtitle: React.FC<{ name: string; text: string }> = ({ name, text }) => {
  const color = nameToColor(name);
  return (
    <div style={{
      position: "absolute", bottom: 80, left: 0, right: 0,
      display: "flex", justifyContent: "center", alignItems: "flex-end",
      animation: "subtitle-in 0.15s ease-out",
    }}>
      <style dangerouslySetInnerHTML={{ __html: `@keyframes subtitle-in { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }` }} />
      <div style={{ maxWidth: "80%", textAlign: "center" }}>
        <div style={{ color, fontSize: 26, fontFamily: "sans-serif", fontWeight: 700, marginBottom: 8, textTransform: "uppercase", letterSpacing: 2 }}>
          {name}
        </div>
        <div style={{
          color: "#fff", fontSize: 36, fontFamily: "Georgia, serif", lineHeight: 1.5,
          textShadow: "0 2px 8px rgba(0,0,0,0.8)", padding: "12px 24px",
          backgroundColor: "rgba(0,0,0,0.5)", borderRadius: 12, borderLeft: `3px solid ${color}`,
        }}>
          {text}
        </div>
      </div>
    </div>
  );
};

// ---- Main ----

export const LiveApp: React.FC = () => {
  const [header, setHeader] = useState(emptyHeader);
  const [cast, setCast] = useState<string[]>([]);
  const [activeSpeaker, setActiveSpeaker] = useState("");
  const [subtitle, setSubtitle] = useState<{ name: string; text: string } | null>(null);
  const [mood, setMood] = useState("default");
  const [connected, setConnected] = useState(false);
  const [started, setStarted] = useState(false);
  const startedRef = useRef(false);
  const pendingRef = useRef<WSMessage[]>([]);
  const audioQueue = useRef(new AudioQueue());
  const mixer = useRef<AudioMixer | null>(null);
  const timelineRef = useRef<LiveEntry[]>([]);
  const subtitleGen = useRef(0); // generation counter to prevent stale timeouts clearing subtitles

  // Subtitle synced to audio playback — shows when audio starts, clears when it ends.
  audioQueue.current.onStart = (url) => {
    const entry = timelineRef.current.find((e) => e.audioUrl === url);
    if (entry) {
      subtitleGen.current++;
      setActiveSpeaker(entry.name);
      setSubtitle({ name: entry.name, text: entry.text });
      if (entry.mood) setMood(entry.mood);
    }
  };

  audioQueue.current.onEnd = () => {
    subtitleGen.current++;
    setSubtitle(null);
    setActiveSpeaker("");
  };

  audioQueue.current.onSpeechStart = () => { mixer.current?.duck(); };
  audioQueue.current.onSpeechEnd = () => { mixer.current?.unduck(); };

  const replayingRef = useRef(false);

  function applyMessage(msg: WSMessage) {
    const isReplay = replayingRef.current;

    switch (msg.type) {
      case "init":
        setHeader(msg.header);
        const names = Object.keys(msg.header.cast || {});
        setCast(names);
        timelineRef.current = [];
        setActiveSpeaker("");
        setSubtitle(null);
        break;

      case "cast": {
        const newCast = msg.cast;
        setCast(Object.keys(newCast));
        setHeader((prev) => ({ ...prev, cast: newCast }));
        break;
      }

      case "line":
      case "scene": {
        const entry = msg.entry;
        timelineRef.current.push(entry);
        if (entry.mood) setMood(entry.mood);

        if (isReplay) {
          break;
        }

        if (entry.audioUrl) {
          // Subtitle will appear via onStart when audio actually plays
          audioQueue.current.play(entry.audioUrl);
        } else if (entry.type === "scene") {
          // Non-narrated scenes: show subtitle with timer
          const gen = ++subtitleGen.current;
          setSubtitle({ name: "scene", text: entry.text });
          setTimeout(() => { if (subtitleGen.current === gen) setSubtitle(null); }, (entry.duration || 3) * 1000);
        }
        break;
      }

      case "chapter": {

        const gen = ++subtitleGen.current;
        setSubtitle({ name: "", text: msg.text });
        setTimeout(() => { if (subtitleGen.current === gen) setSubtitle(null); }, 2000);
        break;
      }

      case "music": {
        if (isReplay) break;
        audioQueue.current.ensure();
        if (!mixer.current && audioQueue.current.ctx) {
          mixer.current = new AudioMixer(audioQueue.current.ctx);
        }
        const mx = mixer.current;
        if (!mx) break;
        const curve = msg.curve as CurveSpec | undefined;
        if (msg.action === "curve" && msg.curve) {
          mx.setCurve(msg.curve as CurveSpec);
        } else if (msg.action === "load" && msg.url) {
          mx.cue(msg.url, msg.loop ?? true, msg.start_at ?? 0, msg.volume);
        } else if (msg.action === "crossfade") {
          mx.crossfade(msg.fade_ms ?? 2000, curve, msg.volume);
        } else if (msg.action === "play" && msg.url) {
          mx.play(msg.url, msg.volume ?? 0.3, msg.loop ?? true, msg.fade_ms ?? 1000, msg.start_at ?? 0, curve);
        } else if (msg.action === "queue" && msg.url) {
          mx.enqueue(msg.url, msg.volume, msg.loop);
        } else if (msg.action === "next") {
          mx.next(msg.fade_ms ?? 2000, curve);
        } else if (msg.action === "stop") {
          mx.stop(msg.fade_ms ?? 1000);
        } else if (msg.action === "fade_out") {
          mx.fadeOut(msg.fade_ms ?? 2000);
        } else if (msg.action === "volume") {
          mx.setVolume(msg.volume ?? 0.3, msg.fade_ms ?? 1000);
        }
        break;
      }

      case "sfx": {
        if (isReplay) break;
        audioQueue.current.ensure();
        if (!mixer.current && audioQueue.current.ctx) {
          mixer.current = new AudioMixer(audioQueue.current.ctx);
        }
        mixer.current?.playSfx(msg.url, msg.volume ?? 0.8);
        break;
      }

      case "stop":
        mixer.current?.stop(500);
        break;
    }
  }

  const ensureAudio = useCallback(() => {
    audioQueue.current.ensure();
    if (!mixer.current && audioQueue.current.ctx) {
      mixer.current = new AudioMixer(audioQueue.current.ctx);
    }
  }, []);

  // Auto-unlock audio: play a silent <audio> element to prime browser autoplay.
  // Chrome allows <audio autoplay> on localhost, which then unlocks AudioContext.
  useEffect(() => {
    // Generate a tiny silent WAV (44 bytes header + 0 samples)
    const header = new ArrayBuffer(44);
    const v = new DataView(header);
    const s = (o: number, str: string) => { for (let i = 0; i < str.length; i++) v.setUint8(o + i, str.charCodeAt(i)); };
    s(0, "RIFF"); v.setUint32(4, 36, true); s(8, "WAVE");
    s(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
    v.setUint16(22, 1, true); v.setUint32(24, 8000, true); v.setUint32(28, 8000, true);
    v.setUint16(32, 1, true); v.setUint16(34, 8, true); s(36, "data"); v.setUint32(40, 0, true);
    const blob = new Blob([header], { type: "audio/wav" });
    const url = URL.createObjectURL(blob);
    const a = new Audio(url);
    a.autoplay = true;
    a.volume = 0;
    a.play().then(() => {
      // Audio autoplay succeeded — now AudioContext will be unlocked too
      ensureAudio();
      URL.revokeObjectURL(url);
    }).catch(() => {
      // Autoplay blocked — fall back to gesture-based unlock
      URL.revokeObjectURL(url);
    });
  }, []);

  const handleMessage = useCallback((msg: WSMessage) => {
    if (msg.type === "init") {
      applyMessage(msg);
      if (!startedRef.current) {
        startedRef.current = true;
        setStarted(true);
        ensureAudio();
      }
      return;
    }
    if (!startedRef.current) {
      pendingRef.current.push(msg);
      return;
    }
    ensureAudio();
    applyMessage(msg);
  }, []);

  // Replay pending messages once we're started (from WebSocket replay)
  useEffect(() => {
    if (started && pendingRef.current.length > 0) {
      replayingRef.current = true;
      for (const msg of pendingRef.current) applyMessage(msg);
      replayingRef.current = false;
      pendingRef.current = [];
    }
  }, [started]);

  // Fallback: resume AudioContext on user gesture if autoplay was blocked
  useEffect(() => {
    const resume = () => {
      ensureAudio();
      const ctx = audioQueue.current.ctx;
      if (ctx && ctx.state === "suspended") ctx.resume();
    };
    window.addEventListener("click", resume);
    window.addEventListener("touchstart", resume);
    window.addEventListener("keydown", resume);
    return () => {
      window.removeEventListener("click", resume);
      window.removeEventListener("touchstart", resume);
      window.removeEventListener("keydown", resume);
    };
  }, []);

  useEffect(() => {
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout>;
    let stopped = false;

    function connect() {
      if (stopped) return;
      ws = new WebSocket(`ws://${location.host}/ws`);
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        if (!stopped) reconnectTimer = setTimeout(connect, 2000);
      };
      ws.onerror = () => ws?.close();
      ws.onmessage = (e) => {
        try { handleMessage(JSON.parse(e.data)); }
        catch (err) { console.error("Bad WS message:", err); }
      };
    }
    connect();

    return () => {
      stopped = true;
      clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, [handleMessage]);

  const title = header.title || "Live Radio Studio";
  const moodColor = (moodColors[mood] ?? moodColors.default).accent;
  const panelWidth = 1920 / (cast.length + 1);

  // Compute scale to fit 1920x1080 into viewport
  const [scale, setScale] = useState(1);
  useEffect(() => {
    const update = () => setScale(Math.min(window.innerWidth / 1920, window.innerHeight / 1080));
    update();
    window.addEventListener("resize", update);
    return () => window.removeEventListener("resize", update);
  }, []);

  return (
    <div style={{
      width: "100vw", height: "100vh",
      backgroundColor: "#000", overflow: "hidden",
      position: "relative",
    }}>
      <div style={{
        width: 1920, height: 1080, position: "absolute",
        left: "50%", top: "50%",
        transform: `translate(-50%, -50%) scale(${scale})`,
        overflow: "hidden",
        background: "linear-gradient(180deg, #0a0a0f 0%, #0f1018 40%, #111320 100%)",
      }}>
          {/* Grid */}
          <div style={{
            position: "absolute", inset: 0,
            backgroundImage: "linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)",
            backgroundSize: "60px 60px",
          }} />

          {/* Orb */}
          <LiveMorphOrb color={moodColor} x={960} y={540} />

          <OnAirBadge />

          {/* Show title */}
          <div style={{ position: "absolute", top: 38, right: 60, fontSize: 20, color: "#444", fontFamily: "sans-serif", fontWeight: 300, letterSpacing: 2, textTransform: "uppercase" }}>
            {title}
          </div>

          {/* Speaker panels */}
          {cast.map((name, i) => (
            <SpeakerCircle key={name} name={name} isActive={activeSpeaker === name} x={panelWidth * (i + 1)} y={180} />
          ))}

          {/* Divider */}
          <div style={{ position: "absolute", top: 270, left: 80, right: 80, height: 1, background: "linear-gradient(90deg, transparent, #ffffff10, transparent)" }} />

          {/* Subtitle */}
          {subtitle && <Subtitle name={subtitle.name} text={subtitle.text} />}

          {/* Connecting indicator */}
          {!connected && !started && (
            <div style={{
              position: "absolute", top: 16, left: "50%", transform: "translateX(-50%)",
              padding: "8px 16px", borderRadius: 8,
              backgroundColor: "rgba(255,255,255,0.1)",
              color: "#666", fontSize: 14, fontFamily: "sans-serif", zIndex: 100,
            }}>
              Connecting...
            </div>
          )}

          {started && !connected && (
            <div style={{ position: "absolute", top: 16, right: 16, padding: "8px 16px", borderRadius: 8, backgroundColor: "rgba(255,50,50,0.9)", color: "#fff", fontSize: 14, fontFamily: "sans-serif", zIndex: 100 }}>
              Disconnected
            </div>
          )}
      </div>
    </div>
  );
};
