// Pick a recording, set the options, run the pipeline in the back-end and follow its progress.
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";

type Stage = "load" | "decode" | "diarize" | "transcribe" | "assemble";

type ProgressEvent =
  | { kind: "started"; stage: Stage }
  | { kind: "chunk"; done: number; total: number; audioDone: number; audioTotal: number }
  | { kind: "finished"; stage: Stage; seconds: number };

interface Cue {
  start: number;
  end: number;
  speaker: number;
  text: string;
}

interface RunResult {
  audioDurationS: number;
  numSpeakers: number;
  processingS: number;
  cues: Cue[];
}

interface RunError {
  kind: "input" | "modelMissing" | "processing" | "busy";
  message: string;
}

const STAGES: [Stage, string][] = [
  ["load", "Loading models"],
  ["decode", "Decoding audio"],
  ["diarize", "Finding speakers"],
  ["transcribe", "Transcribing"],
  ["assemble", "Assembling the transcript"],
];

const EXTENSIONS = ["mp4", "m4a", "mov", "mkv", "webm", "wav", "mp3", "flac", "ogg", "aac", "aiff", "caf"];

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const form = $<HTMLFormElement>("run-form");
const fileLabel = $("file");
const runButton = $<HTMLButtonElement>("run");
const pickButton = $<HTMLButtonElement>("pick");
const stagesList = $<HTMLOListElement>("stages");
const chunkBar = $<HTMLProgressElement>("chunks");
const chunkText = $("chunk-text");
const errorBox = $("error");
const resultBox = $("result");

let path: string | null = null;
let running = false;

function setRunning(on: boolean) {
  running = on;
  runButton.disabled = on || path === null;
  pickButton.disabled = on;
  for (const el of form.querySelectorAll<HTMLInputElement>("input, select, textarea")) el.disabled = on;
  runButton.textContent = on ? "Transcribing…" : "Transcribe";
}

function clock(s: number): string {
  const t = Math.floor(s);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const sec = String(t % 60).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
}

function showError(e: RunError) {
  errorBox.textContent = e.message;
  errorBox.dataset.kind = e.kind;
  errorBox.hidden = false;
}

function resetProgress() {
  stagesList.replaceChildren(
    ...STAGES.map(([stage, label]) => {
      const li = document.createElement("li");
      li.dataset.stage = stage;
      li.dataset.state = "pending";
      li.textContent = label;
      return li;
    }),
  );
  chunkBar.value = 0;
  chunkText.textContent = "";
  $("progress").hidden = false;
}

function onProgress(p: ProgressEvent) {
  if (p.kind === "chunk") {
    chunkBar.value = p.audioTotal > 0 ? p.audioDone / p.audioTotal : 1;
    chunkText.textContent = `Chunk ${p.done} of ${p.total} · ${clock(p.audioDone)} of ${clock(p.audioTotal)} of speech`;
    return;
  }
  const li = stagesList.querySelector<HTMLLIElement>(`[data-stage="${p.stage}"]`);
  if (!li) return;
  const label = STAGES.find(([s]) => s === p.stage)?.[1] ?? p.stage;
  li.dataset.state = p.kind === "started" ? "active" : "done";
  li.textContent = p.kind === "started" ? `${label}…` : `${label} (${p.seconds.toFixed(1)} s)`;
}

function showResult(r: RunResult) {
  $("summary").textContent =
    `${clock(r.audioDurationS)} of audio, ${r.numSpeakers} speaker${r.numSpeakers === 1 ? "" : "s"}, ` +
    `${r.cues.length} cues, processed in ${r.processingS.toFixed(0)} s.`;
  $("cues").replaceChildren(
    ...r.cues.map((c) => {
      const li = document.createElement("li");
      const meta = document.createElement("span");
      meta.className = "muted";
      meta.textContent = `${clock(c.start)} Speaker ${c.speaker}`;
      li.append(meta, ` ${c.text}`);
      return li;
    }),
  );
  resultBox.hidden = false;
}

pickButton.addEventListener("click", async () => {
  const picked = await open({ multiple: false, directory: false, filters: [{ name: "Recordings", extensions: EXTENSIONS }] });
  if (typeof picked !== "string") return;
  path = picked;
  fileLabel.textContent = picked;
  fileLabel.classList.remove("muted");
  setRunning(running);
});

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  if (running || path === null) return;
  const speakers = $<HTMLInputElement>("speakers").valueAsNumber;
  const options = {
    language: $<HTMLSelectElement>("language").value,
    prompt: $<HTMLTextAreaElement>("prompt").value,
    numSpeakers: Number.isFinite(speakers) ? speakers : null,
  };
  errorBox.hidden = true;
  resultBox.hidden = true;
  resetProgress();
  setRunning(true);
  try {
    await invoke("start_run", { path, options });
  } catch (e) {
    setRunning(false);
    $("progress").hidden = true;
    showError(e as RunError);
  }
});

void listen<ProgressEvent>("run-progress", (e) => onProgress(e.payload));
void listen<RunResult>("run-finished", (e) => {
  setRunning(false);
  showResult(e.payload);
});
void listen<RunError>("run-failed", (e) => {
  setRunning(false);
  showError(e.payload);
});
