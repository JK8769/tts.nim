import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import type { ScriptLine, TimelineEntry, ScriptData } from "./types";
import { nameToColor, moodColors } from "./types";

/** Timeline entry with its own audio URL (for live streaming). */
interface LiveEntry extends TimelineEntry {
  audioUrl?: string;
}

const FPS = 30;

// ---- Types ----

type WSMessage =
  | { type: "init"; header: ScriptData["header"] }
  | { type: "line"; entry: LiveEntry; scriptLine?: ScriptLine }
  | { type: "chapter"; text: string; index: number }
  | { type: "scene"; entry: LiveEntry; scriptLine?: ScriptLine }
  | { type: "music"; action: string; url?: string; volume?: number; loop?: boolean; fade_ms?: number; start_at?: number }
  | { type: "sfx"; url: string; volume?: number }
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

class MusicPlayer {
  private ctx: AudioContext;
  private gain: GainNode;
  private source: AudioBufferSourceNode | null = null;
  private buffer: AudioBuffer | null = null;
  private currentUrl = "";
  private baseVolume = 0.3;
  private duckVolume = 0.08;
  private ducked = false;
  private playlist: { url: string; volume?: number; loop?: boolean }[] = [];

  constructor(ctx: AudioContext) {
    this.ctx = ctx;
    this.gain = ctx.createGain();
    this.gain.gain.value = 0;
    this.gain.connect(ctx.destination);
  }

  async play(url: string, volume = 0.3, loop = true, fadeMs = 1000, startAt = 0) {
    this.stop(0);
    this.baseVolume = volume;
    this.duckVolume = volume * 0.25;
    this.currentUrl = url;
    try {
      const resp = await fetch(url);
      const buf = await resp.arrayBuffer();
      const audio = await this.ctx.decodeAudioData(buf);
      this.buffer = audio;
      this.source = this.ctx.createBufferSource();
      this.source.buffer = audio;
      this.source.loop = loop;
      this.source.connect(this.gain);
      // If not looping, auto-advance to next in playlist when done
      if (!loop && this.playlist.length > 0) {
        this.source.onended = () => this.next();
      }
      this.source.start(0, startAt);
      // Fade in
      const target = this.ducked ? this.duckVolume : this.baseVolume;
      this.gain.gain.setValueAtTime(0, this.ctx.currentTime);
      this.gain.gain.linearRampToValueAtTime(target, this.ctx.currentTime + fadeMs / 1000);
    } catch (e) {
      console.error("Music error:", e);
    }
  }

  enqueue(url: string, volume?: number, loop?: boolean) {
    this.playlist.push({ url, volume, loop });
  }

  async next(crossfadeMs = 2000) {
    const item = this.playlist.shift();
    if (!item) return;
    // Crossfade: move old source to a temporary gain for fade-out,
    // then start new source on the main gain for fade-in
    if (this.source) {
      const oldSrc = this.source;
      const oldGain = this.ctx.createGain();
      oldGain.gain.setValueAtTime(this.gain.gain.value, this.ctx.currentTime);
      oldGain.connect(this.ctx.destination);
      oldSrc.disconnect();
      oldSrc.connect(oldGain);
      const now = this.ctx.currentTime;
      oldGain.gain.linearRampToValueAtTime(0, now + crossfadeMs / 1000);
      setTimeout(() => { try { oldSrc.stop(); } catch {} }, crossfadeMs + 100);
      this.source = null;
      this.buffer = null;
    }
    // Reset main gain and start new track
    this.gain.gain.setValueAtTime(0, this.ctx.currentTime);
    this.play(item.url, item.volume ?? this.baseVolume, item.loop ?? false, crossfadeMs);
  }

  stop(fadeMs = 1000) {
    if (!this.source) return;
    const now = this.ctx.currentTime;
    this.gain.gain.cancelScheduledValues(now);
    this.gain.gain.setValueAtTime(this.gain.gain.value, now);
    this.gain.gain.linearRampToValueAtTime(0, now + fadeMs / 1000);
    const src = this.source;
    this.source = null;
    this.buffer = null;
    setTimeout(() => { try { src.stop(); } catch {} }, fadeMs + 100);
  }

  fadeOut(fadeMs = 2000) {
    this.stop(fadeMs);
  }

  duck() {
    if (this.ducked || !this.source) return;
    this.ducked = true;
    const now = this.ctx.currentTime;
    this.gain.gain.cancelScheduledValues(now);
    this.gain.gain.setValueAtTime(this.gain.gain.value, now);
    this.gain.gain.linearRampToValueAtTime(this.duckVolume, now + 0.3);
  }

  unduck() {
    if (!this.ducked || !this.source) return;
    this.ducked = false;
    const now = this.ctx.currentTime;
    this.gain.gain.cancelScheduledValues(now);
    this.gain.gain.setValueAtTime(this.gain.gain.value, now);
    this.gain.gain.linearRampToValueAtTime(this.baseVolume, now + 0.5);
  }

  setVolume(volume: number, fadeMs = 1000) {
    this.baseVolume = volume;
    this.duckVolume = volume * 0.25;
    if (!this.source) return;
    const target = this.ducked ? this.duckVolume : this.baseVolume;
    const now = this.ctx.currentTime;
    this.gain.gain.cancelScheduledValues(now);
    this.gain.gain.setValueAtTime(this.gain.gain.value, now);
    this.gain.gain.linearRampToValueAtTime(target, now + fadeMs / 1000);
  }
}

class SfxPlayer {
  private ctx: AudioContext;
  private gain: GainNode;

  constructor(ctx: AudioContext) {
    this.ctx = ctx;
    this.gain = ctx.createGain();
    this.gain.gain.value = 1.0;
    this.gain.connect(ctx.destination);
  }

  async play(url: string, volume = 0.8) {
    try {
      const resp = await fetch(url);
      const buf = await resp.arrayBuffer();
      const audio = await this.ctx.decodeAudioData(buf);
      const src = this.ctx.createBufferSource();
      src.buffer = audio;
      this.gain.gain.value = volume;
      src.connect(this.gain);
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
  const musicPlayer = useRef<MusicPlayer | null>(null);
  const timelineRef = useRef<LiveEntry[]>([]);
  const subtitleGen = useRef(0); // generation counter to prevent stale timeouts clearing subtitles

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
  };

  audioQueue.current.onSpeechStart = () => {
    musicPlayer.current?.duck();
  };

  audioQueue.current.onSpeechEnd = () => {
    musicPlayer.current?.unduck();
  };

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

      case "line":
      case "scene": {
        const entry = msg.entry;
        timelineRef.current.push(entry);
        if (entry.mood) setMood(entry.mood);
        if (isReplay) {
          // During replay, just track state — don't queue old audio
          break;
        }
        if (entry.audioUrl) {
          audioQueue.current.play(entry.audioUrl);
        } else if (entry.type === "scene") {
          // For non-narrated scenes, show as subtitle briefly
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
        if (isReplay) break; // Don't replay old music commands
        audioQueue.current.ensure();
        if (!musicPlayer.current && audioQueue.current.ctx) {
          musicPlayer.current = new MusicPlayer(audioQueue.current.ctx);
        }
        const mp = musicPlayer.current;
        if (!mp) break;
        if (msg.action === "play" && msg.url) {
          mp.play(msg.url, msg.volume ?? 0.3, msg.loop ?? true, msg.fade_ms ?? 1000, msg.start_at ?? 0);
        } else if (msg.action === "queue" && msg.url) {
          mp.enqueue(msg.url, msg.volume, msg.loop);
        } else if (msg.action === "next") {
          mp.next(msg.fade_ms ?? 2000);
        } else if (msg.action === "stop") {
          mp.stop(msg.fade_ms ?? 1000);
        } else if (msg.action === "fade_out") {
          mp.fadeOut(msg.fade_ms ?? 2000);
        } else if (msg.action === "volume") {
          mp.setVolume(msg.volume ?? 0.3, msg.fade_ms ?? 1000);
        }
        break;
      }

      case "sfx": {
        if (isReplay) break; // Don't replay old SFX
        audioQueue.current.ensure();
        if (audioQueue.current.ctx) {
          const sfx = new SfxPlayer(audioQueue.current.ctx);
          sfx.play(msg.url, msg.volume ?? 0.8);
        }
        break;
      }

      case "stop":
        musicPlayer.current?.stop(500);
        break;
    }
  }

  const handleMessage = useCallback((msg: WSMessage) => {
    // Init messages apply immediately (they set title/cast for the overlay)
    if (msg.type === "init") {
      applyMessage(msg);
      return;
    }
    if (!startedRef.current) {
      pendingRef.current.push(msg);
      return;
    }
    applyMessage(msg);
  }, []);

  const handleStart = useCallback(() => {
    startedRef.current = true;
    setStarted(true);
    audioQueue.current.ensure();
    if (!musicPlayer.current && audioQueue.current.ctx) {
      musicPlayer.current = new MusicPlayer(audioQueue.current.ctx);
    }
    replayingRef.current = true;
    for (const msg of pendingRef.current) applyMessage(msg);
    replayingRef.current = false;
    pendingRef.current = [];
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
  const castNames = Object.keys(header.cast || {});
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

          {/* Click-to-start */}
          {!started && (
            <div onClick={handleStart} style={{
              position: "absolute", inset: 0, backgroundColor: "rgba(0,0,0,0.85)",
              display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
              cursor: "pointer", zIndex: 200,
            }}>
              <div style={{ width: 80, height: 80, borderRadius: "50%", border: "3px solid #fff", display: "flex", alignItems: "center", justifyContent: "center", marginBottom: 24 }}>
                <div style={{ width: 0, height: 0, borderTop: "18px solid transparent", borderBottom: "18px solid transparent", borderLeft: "30px solid #fff", marginLeft: 6 }} />
              </div>
              <div style={{ color: "#fff", fontSize: 28, fontFamily: "sans-serif", fontWeight: 300, letterSpacing: 4, textTransform: "uppercase" }}>{title}</div>
              {castNames.length > 0 && <div style={{ color: "#666", fontSize: 16, fontFamily: "sans-serif", marginTop: 12, letterSpacing: 2 }}>{castNames.join(" \u00B7 ")}</div>}
              <div style={{ color: "#444", fontSize: 14, fontFamily: "sans-serif", marginTop: 32 }}>{connected ? "Click to start" : "Connecting..."}</div>
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
