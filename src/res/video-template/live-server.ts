#!/usr/bin/env bun
/**
 * Live streaming server for the Radio Studio template.
 *
 * Usage: bun live-server.ts [--port 3333] [--audio-dir /tmp/live-audio]
 *
 * HTTP endpoints:
 *   GET  /              — live player page
 *   GET  /live-entry.js — bundled React app
 *   GET  /audio/:name   — serve audio files from audio-dir
 *   POST /api/init      — initialize show (body: {header})
 *   POST /api/line      — add a spoken line (body: {entry, scriptLine?})
 *   POST /api/scene     — add a scene card (body: {entry, scriptLine?})
 *   POST /api/chapter   — add a chapter title (body: {text, index})
 *   POST /api/stop      — signal end of stream
 *
 * WebSocket:
 *   /ws — broadcasts all updates to connected players
 *         replays full state on connect so late-joining clients catch up
 */

import { readFileSync, existsSync } from "fs";
import { resolve, dirname, join } from "path";

const root = dirname(new URL(import.meta.url).pathname);

// Parse args
let port = 3333;
let audioDir = "/tmp/live-audio";

const args = process.argv.slice(2);
for (let i = 0; i < args.length; i++) {
  if (args[i] === "--port" && args[i + 1]) port = parseInt(args[i + 1]);
  if (args[i] === "--audio-dir" && args[i + 1]) audioDir = resolve(args[i + 1]);
}

// Build the React app bundle
console.log("Building live player bundle...");
const buildResult = await Bun.build({
  entrypoints: [resolve(root, "src/live-entry.tsx")],
  outdir: resolve(root, ".live-build"),
  target: "browser",
  format: "esm",
  minify: true,
  define: {
    "process.env.NODE_ENV": '"production"',
  },
});

if (!buildResult.success) {
  console.error("Build failed:");
  for (const log of buildResult.logs) console.error(log);
  process.exit(1);
}
console.log("Bundle ready.");

// Read static files
const html = readFileSync(resolve(root, "live.html"), "utf-8");
const bundlePath = resolve(root, ".live-build/live-entry.js");

// ---- State: replay log for late-joining clients ----
const messageLog: object[] = [];

// WebSocket clients
const clients = new Set<any>();

function broadcast(msg: object) {
  messageLog.push(msg);
  const json = JSON.stringify(msg);
  for (const ws of clients) {
    try {
      ws.send(json);
    } catch {}
  }
}

// Audio MIME types
function audioMime(name: string): string {
  if (name.endsWith(".wav")) return "audio/wav";
  if (name.endsWith(".mp3")) return "audio/mpeg";
  if (name.endsWith(".ogg")) return "audio/ogg";
  return "application/octet-stream";
}

const server = Bun.serve({
  port,
  async fetch(req, server) {
    const url = new URL(req.url);

    // WebSocket upgrade
    if (url.pathname === "/ws") {
      if (server.upgrade(req)) return undefined as any;
      return new Response("WebSocket upgrade failed", { status: 400 });
    }

    // HTML page
    if (url.pathname === "/") {
      return new Response(html, {
        headers: { "Content-Type": "text/html; charset=utf-8" },
      });
    }

    // JS bundle — read fresh on each request so rebuilds take effect
    if (url.pathname === "/live-entry.js") {
      return new Response(readFileSync(bundlePath, "utf-8"), {
        headers: { "Content-Type": "application/javascript", "Cache-Control": "no-cache" },
      });
    }

    // Audio files — with Range request support for Remotion seeking
    if (url.pathname.startsWith("/audio/")) {
      const name = url.pathname.slice(7); // strip /audio/
      const filePath = join(audioDir, name);
      if (existsSync(filePath)) {
        const file = Bun.file(filePath);
        const size = file.size;
        const mime = audioMime(name);
        const rangeHeader = req.headers.get("range");

        if (rangeHeader) {
          const match = rangeHeader.match(/bytes=(\d+)-(\d*)/);
          if (match) {
            const start = parseInt(match[1]);
            const end = match[2] ? parseInt(match[2]) : size - 1;
            const chunk = file.slice(start, end + 1);
            return new Response(chunk, {
              status: 206,
              headers: {
                "Content-Type": mime,
                "Content-Range": `bytes ${start}-${end}/${size}`,
                "Content-Length": String(end - start + 1),
                "Accept-Ranges": "bytes",
                "Access-Control-Allow-Origin": "*",
              },
            });
          }
        }

        return new Response(file, {
          headers: {
            "Content-Type": mime,
            "Content-Length": String(size),
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*",
          },
        });
      }
      return new Response("Not found", { status: 404 });
    }

    // Control API
    if (req.method === "POST" && url.pathname.startsWith("/api/")) {
      const body = await req.json();
      const endpoint = url.pathname.slice(5); // strip /api/

      switch (endpoint) {
        case "init":
          // Clear previous state on re-init
          messageLog.length = 0;
          broadcast({ type: "init", header: body.header });
          return new Response(JSON.stringify({ ok: true }));

        case "line":
          broadcast({
            type: "line",
            entry: body.entry,
            scriptLine: body.scriptLine,
          });
          return new Response(JSON.stringify({ ok: true }));

        case "scene":
          broadcast({
            type: "scene",
            entry: body.entry,
            scriptLine: body.scriptLine,
          });
          return new Response(JSON.stringify({ ok: true }));

        case "chapter":
          broadcast({
            type: "chapter",
            text: body.text,
            index: body.index,
          });
          return new Response(JSON.stringify({ ok: true }));

        case "music":
          broadcast({
            type: "music",
            action: body.action,
            url: body.url,
            volume: body.volume,
            loop: body.loop,
            fade_ms: body.fade_ms,
            start_at: body.start_at,
          });
          return new Response(JSON.stringify({ ok: true }));

        case "sfx":
          broadcast({
            type: "sfx",
            url: body.url,
            volume: body.volume,
          });
          return new Response(JSON.stringify({ ok: true }));

        case "stop":
          broadcast({ type: "stop" });
          return new Response(JSON.stringify({ ok: true }));

        default:
          return new Response("Unknown endpoint", { status: 404 });
      }
    }

    return new Response("Not found", { status: 404 });
  },

  websocket: {
    open(ws) {
      clients.add(ws);
      console.log(`Client connected (${clients.size} total)`);
      // Replay all previous messages so late-joining clients catch up
      for (const msg of messageLog) {
        try {
          ws.send(JSON.stringify(msg));
        } catch {}
      }
    },
    close(ws) {
      clients.delete(ws);
      console.log(`Client disconnected (${clients.size} total)`);
    },
    message(_ws, _data) {
      // Clients don't send messages — control is via HTTP API
    },
  },
});

console.log(`\nLive Radio Studio running at http://localhost:${port}`);
console.log(`Audio directory: ${audioDir}`);
console.log(`\nOpen in browser for OBS capture. Control via POST to /api/*`);
