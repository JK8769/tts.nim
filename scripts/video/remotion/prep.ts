#!/usr/bin/env bun
/**
 * Prepare script data for Remotion rendering.
 *
 * Usage: bun prep.ts <script.jsonl> <audio.mp3> <render_result.json>
 *
 * render_result.json is the JSON output from the MCP script render action,
 * containing { duration, lines_rendered, timeline: [...] }.
 */

import { readFileSync, writeFileSync, copyFileSync, mkdirSync } from "fs";
import { resolve, dirname } from "path";

const args = process.argv.slice(2);
if (args.length < 3) {
  console.error("Usage: bun prep.ts <script.jsonl> <audio.mp3> <render_result.json>");
  process.exit(1);
}

const scriptPath = resolve(args[0]);
const audioPath = resolve(args[1]);
const renderPath = resolve(args[2]);

const root = dirname(new URL(import.meta.url).pathname);

// Parse JSONL script — build index map (line index → line data)
const rawLines = readFileSync(scriptPath, "utf-8")
  .split("\n")
  .filter((l) => l.trim())
  .map((l) => JSON.parse(l));

const header = rawLines[0]?.type === "header" ? rawLines[0] : null;
if (!header) {
  console.error("Script missing header line");
  process.exit(1);
}

// Lines after header, indexed from 0 (matching render timeline index - 0 based after header)
const scriptLines = rawLines.slice(1);

// Parse render result
const renderResult = JSON.parse(readFileSync(renderPath, "utf-8"));
const audioDuration = renderResult.duration;
const rawTimeline: any[] = renderResult.timeline ?? [];

// Split text into subtitle-sized chunks at sentence/clause boundaries
function splitSubtitles(text: string, maxLen = 80): string[] {
  if (text.length <= maxLen) return [text];
  const chunks: string[] = [];
  // Split on sentence endings first, then commas/semicolons
  const sentences = text.match(/[^.!?]+[.!?]+\s*|[^.!?]+$/g) ?? [text];
  let current = "";
  for (const s of sentences) {
    if (current.length + s.length > maxLen && current.length > 0) {
      chunks.push(current.trim());
      current = s;
    } else {
      current += s;
    }
  }
  if (current.trim()) chunks.push(current.trim());
  // If any chunk is still too long, split on commas
  const result: string[] = [];
  for (const chunk of chunks) {
    if (chunk.length <= maxLen) {
      result.push(chunk);
      continue;
    }
    const parts = chunk.split(/,\s*/);
    let line = "";
    for (const p of parts) {
      const candidate = line ? line + ", " + p : p;
      if (candidate.length > maxLen && line.length > 0) {
        result.push(line.trim());
        line = p;
      } else {
        line = candidate;
      }
    }
    if (line.trim()) result.push(line.trim());
  }
  return result.length > 0 ? result : [text];
}

// Enrich timeline entries and split long dialogue into subtitle chunks
const timeline: any[] = [];
for (const entry of rawTimeline) {
  const scriptLine = scriptLines[entry.index] ?? {};
  const fullText = scriptLine.text ?? entry.text ?? "";

  if (entry.type === "scene") {
    // Scenes keep full text (displayed differently)
    timeline.push({
      ...entry,
      text: fullText,
      setting: scriptLine.setting,
      characters: scriptLine.characters,
      mood: scriptLine.mood,
    });
    continue;
  }

  // Split dialogue into subtitle chunks, timed proportional to char length
  const chunks = splitSubtitles(fullText);
  // Calculate lead time: use the silence gap before this entry
  const entryIdx = rawTimeline.indexOf(entry);
  let leadTime = 0;
  if (entryIdx > 0) {
    const prev = rawTimeline[entryIdx - 1];
    const gapStart = prev.start + prev.duration;
    const gap = entry.start - gapStart;
    // Use up to 80% of the gap, capped at 0.5s
    leadTime = Math.min(gap * 0.8, 0.5);
    if (leadTime < 0.05) leadTime = 0;
  }

  const totalChars = chunks.reduce((sum, c) => sum + c.length, 0);
  let offset = 0;
  for (let i = 0; i < chunks.length; i++) {
    const chunkDuration = (chunks[i].length / totalChars) * entry.duration;
    timeline.push({
      ...entry,
      text: chunks[i],
      start: Math.max(0, entry.start + offset - (i === 0 ? leadTime : 0)),
      duration: chunkDuration + (i === 0 ? leadTime : 0),
      setting: scriptLine.setting,
      characters: scriptLine.characters,
      mood: scriptLine.mood,
    });
    offset += chunkDuration;
  }
}

// Write data.json
const data = {
  header: {
    type: header.type,
    title: header.title,
    format: header.format,
    cast: header.cast,
    defaults: header.defaults,
  },
  lines: scriptLines,
  timeline,
  audioDuration,
};

const dataPath = resolve(root, "src/data.json");
writeFileSync(dataPath, JSON.stringify(data, null, 2));
console.log(`Wrote ${dataPath} (${timeline.length} timeline entries, ${audioDuration.toFixed(1)}s)`);

// Copy audio to public/
const publicDir = resolve(root, "public");
mkdirSync(publicDir, { recursive: true });
const audioDest = resolve(publicDir, "audio.mp3");
copyFileSync(audioPath, audioDest);
console.log(`Copied audio to ${audioDest}`);

console.log("\nReady! Run:");
console.log("  cd video && bun run dev     # preview in browser");
console.log("  cd video && bun run build   # render to out/video.mp4");
