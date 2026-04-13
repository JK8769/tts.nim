#!/usr/bin/env bun
/**
 * suno-dl — Browse and download trending songs from Suno (no auth required).
 *
 * Uses playwright-cli to scrape suno.com/explore for song metadata,
 * then downloads directly from cdn1.suno.ai (public, no auth).
 *
 * Usage:
 *   bun scripts/suno-dl.ts                    # list trending songs, save catalog.json
 *   bun scripts/suno-dl.ts --download 3       # download top 3
 *   bun scripts/suno-dl.ts --download 1,3,5   # download specific indices
 *   bun scripts/suno-dl.ts --out ~/Music      # output directory (default ~/Music)
 *   bun scripts/suno-dl.ts --json              # only output JSON (no table)
 */

import { execSync } from "child_process";
import { existsSync, mkdirSync, writeFileSync, unlinkSync } from "fs";
import { resolve, join } from "path";
import { homedir } from "os";

const args = process.argv.slice(2);
let outDir = join(homedir(), "Music");
let downloadArg = "";
let jsonOnly = false;

for (let i = 0; i < args.length; i++) {
  if (args[i] === "--out" && args[i + 1]) outDir = resolve(args[i + 1]);
  if (args[i] === "--download" && args[i + 1]) downloadArg = args[i + 1];
  if (args[i] === "--json") jsonOnly = true;
}

if (!existsSync(outDir)) mkdirSync(outDir, { recursive: true });

/** Run playwright-cli eval with a JS file to avoid escaping hell. */
function evalJs(code: string): string {
  const tmp = "/tmp/suno-dl-eval.js";
  writeFileSync(tmp, code);
  try {
    const out = execSync(
      `playwright-cli eval "$(cat ${tmp})"`,
      { encoding: "utf-8", timeout: 20000, shell: "/bin/bash" }
    );
    const match = out.match(/### Result\n([\s\S]*?)(\n###|$)/);
    if (match) return match[1].replace(/^"|"$/g, "").trim();
    return "";
  } catch (e: any) {
    return e.stdout || "";
  } finally {
    try { unlinkSync(tmp); } catch {}
  }
}

function cli(cmd: string): string {
  try {
    return execSync(`playwright-cli ${cmd}`, { encoding: "utf-8", timeout: 15000 });
  } catch (e: any) {
    return e.stdout || "";
  }
}

interface Song {
  id: string;
  title: string;
  artist: string;
  model: string;
  plays: string;
  likes: string;
  saves: string;
  cdn_url: string;
  duration_s?: number;
  duration?: string;
}

/** Get duration in seconds from a CDN URL via ffprobe (reads header only). */
function probeDuration(url: string): number | null {
  try {
    const out = execSync(
      `ffprobe -v quiet -show_entries format=duration -of csv=p=0 "${url}"`,
      { encoding: "utf-8", timeout: 10000 }
    );
    const secs = parseFloat(out.trim());
    return isNaN(secs) ? null : Math.round(secs);
  } catch {
    return null;
  }
}

function formatDuration(secs: number): string {
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

async function main() {
  // Ensure browser is open at explore page
  const openOut = cli("open https://suno.com/explore");
  if (openOut.includes("already open")) {
    cli("goto https://suno.com/explore");
  }

  console.log("Waiting for songs to load...");

  // Poll for song links
  let count = 0;
  for (let i = 0; i < 8; i++) {
    const r = evalJs(`new Promise(r => setTimeout(() => r(String(document.querySelectorAll('a[href*="/song/"]').length)), 3000))`);
    count = parseInt(r) || 0;
    if (count > 0) break;
    console.log(`  attempt ${i + 1}: waiting...`);
  }

  if (count === 0) {
    console.log("No songs found.");
    process.exit(1);
  }

  // Extract songs with full metadata from the page
  // Each song row has: title link, model version badge, artist link, play/like/save counts
  const raw = evalJs(`
    (function() {
      var links = Array.from(document.querySelectorAll('a[href*="/song/"]'))
        .filter(function(a) { return !a.href.includes('show_comments'); });
      var seen = {};
      var results = [];
      links.forEach(function(a) {
        var id = a.href.split('/song/')[1].split('?')[0];
        if (seen[id]) return;
        seen[id] = true;
        var title = (a.textContent || '').trim();

        // Walk up to find the row container with all metadata
        var row = a;
        for (var i = 0; i < 10; i++) {
          if (!row.parentElement) break;
          row = row.parentElement;
          // Stop at the row that contains the stats (plays/likes)
          if (row.querySelectorAll && row.querySelectorAll('svg').length >= 3) break;
        }

        var text = row ? row.innerText : '';
        var lines = text.split('\\n').map(function(s) { return s.trim(); }).filter(Boolean);

        // Find model version (v4, v4.5+, v5, etc)
        var model = '';
        for (var j = 0; j < lines.length; j++) {
          if (/^v[0-9]/.test(lines[j])) { model = lines[j]; break; }
        }

        // Find artist — typically right after model version
        var artist = '';
        var titleIdx = lines.indexOf(title);
        var modelIdx = model ? lines.indexOf(model) : -1;
        if (modelIdx >= 0 && modelIdx + 1 < lines.length) {
          var candidate = lines[modelIdx + 1];
          if (!/^[0-9]/.test(candidate) && candidate !== title) {
            artist = candidate;
          }
        }

        // Find numeric stats (plays, likes, saves) — typically formatted like "194K"
        var stats = [];
        for (var k = 0; k < lines.length; k++) {
          if (/^[0-9][0-9.,]*[KkMm]?$/.test(lines[k])) {
            stats.push(lines[k]);
          }
        }

        results.push([id, title, artist, model, stats[0]||'', stats[1]||'', stats[2]||''].join('\\t'));
      });
      return results.join('\\n');
    })()
  `);

  const songs: Song[] = [];
  for (const line of raw.split("\\n")) {
    const parts = line.split("\\t");
    if (parts.length < 1 || !parts[0]) continue;
    const [id, title, artist, model, plays, likes, saves] = parts;
    songs.push({
      id: id.trim(),
      title: (title || "").trim(),
      artist: (artist || "").trim(),
      model: (model || "").trim(),
      plays: (plays || "").trim(),
      likes: (likes || "").trim(),
      saves: (saves || "").trim(),
      cdn_url: `https://cdn1.suno.ai/${id.trim()}.m4a`,
    });
  }

  // Probe duration for each song
  console.log(`Probing duration for ${songs.length} songs...`);
  let totalDuration = 0;
  for (const song of songs) {
    const secs = probeDuration(song.cdn_url);
    if (secs !== null) {
      song.duration_s = secs;
      song.duration = formatDuration(secs);
      totalDuration += secs;
    }
  }

  // Save catalog JSON
  const catalogPath = join(outDir, "suno-catalog.json");
  writeFileSync(catalogPath, JSON.stringify(songs, null, 2));

  if (jsonOnly) {
    console.log(JSON.stringify(songs, null, 2));
    process.exit(0);
  }

  console.log(`\nFound ${songs.length} songs (total ${formatDuration(totalDuration)}):\n`);
  songs.forEach((s, i) => {
    const num = String(i + 1).padStart(2);
    const dur = s.duration ? ` [${s.duration}]` : "";
    const plays = s.plays ? ` (${s.plays} plays)` : "";
    const artist = s.artist ? ` — ${s.artist}` : "";
    const model = s.model ? ` ${s.model}` : "";
    console.log(`  ${num}. ${s.title || "(untitled)"}${artist}${dur}${model}${plays}`);
  });
  console.log(`\nCatalog saved to ${catalogPath}`);

  if (!downloadArg) {
    console.log("Use --download N to download top N, or --download 1,3,5 for specific ones.");
    process.exit(0);
  }

  // Parse indices
  let indices: number[];
  if (downloadArg.includes(",")) {
    indices = downloadArg.split(",").map(s => parseInt(s) - 1);
  } else {
    const n = parseInt(downloadArg);
    indices = Array.from({ length: n }, (_, i) => i);
  }
  const toDownload = indices.filter(i => i >= 0 && i < songs.length).map(i => songs[i]);

  console.log(`\nDownloading ${toDownload.length} songs to ${outDir}...\n`);

  for (const song of toDownload) {
    const safeName = (song.title || song.id)
      .toLowerCase()
      .replace(/[^a-z0-9\u4e00-\u9fff]+/g, "_")
      .replace(/^_|_$/g, "")
      .slice(0, 60);
    const mp3Path = join(outDir, `${safeName}.mp3`);

    if (existsSync(mp3Path)) {
      console.log(`  Skip (exists): ${safeName}.mp3`);
      continue;
    }

    const tmpPath = `/tmp/suno_${song.id}.m4a`;

    process.stdout.write(`  ${song.title || song.id}... `);
    try {
      execSync(`curl -sfL -o "${tmpPath}" "${song.cdn_url}"`, { timeout: 60000 });
      execSync(`ffmpeg -i "${tmpPath}" -q:a 2 "${mp3Path}" -y 2>/dev/null`, { timeout: 30000 });
      try { unlinkSync(tmpPath); } catch {}
      const size = (Bun.file(mp3Path).size / 1024 / 1024).toFixed(1);
      console.log(`OK (${size}MB)`);
    } catch {
      console.log("FAILED");
    }
  }

  console.log("\nDone!");
}

main();
