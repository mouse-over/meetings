// Pick a recording, set the options, run the pipeline in the back-end and follow its progress, then play the
// recording next to its transcript.
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";

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

interface Exported {
  vtt: string;
  json: string;
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

const AUDIO_EXTENSIONS = ["m4a", "wav", "mp3", "flac", "ogg", "aac", "aiff", "caf"];
const EXTENSIONS = ["mp4", "mov", "mkv", "webm", ...AUDIO_EXTENSIONS];

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
const media = $<HTMLVideoElement>("media");
const mediaError = $("media-error");
const cueList = $<HTMLOListElement>("cues");
const exportButton = $<HTMLButtonElement>("export");
const exportStatus = $("export-status");

let path: string | null = null;
/** The recording the shown transcript belongs to. */
let runPath: string | null = null;
let running = false;
let cues: Cue[] = [];
let activeCue = -1;

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
  cues = r.cues;
  activeCue = -1;
  exportStatus.hidden = true;
  cueList.replaceChildren(
    ...cues.map((c, i) => {
      const li = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      button.className = "cue";
      button.dataset.index = String(i);
      // Colour per speaker: golden-angle hue steps keep neighbouring speaker numbers far apart.
      button.style.setProperty("--speaker-hue", String((c.speaker * 137.5 + 210) % 360));
      const time = document.createElement("span");
      time.className = "time";
      time.textContent = clock(c.start);
      const speaker = document.createElement("span");
      speaker.className = "speaker";
      speaker.textContent = `Speaker ${c.speaker}`;
      const text = document.createElement("span");
      text.className = "text";
      text.textContent = c.text;
      button.append(time, speaker, text);
      li.append(button);
      return li;
    }),
  );
  loadMedia(runPath);
  resultBox.hidden = false;
}

/** Plays the recording from the asset protocol; the dialog allowed the picked file in its scope. */
function loadMedia(p: string | null) {
  mediaError.hidden = true;
  if (p === null) {
    media.removeAttribute("src");
    media.load();
    return;
  }
  const src = convertFileSrc(p);
  if (media.src === src) return;
  // Guess from the extension until the metadata tells.
  const ext = p.split(".").pop()?.toLowerCase() ?? "";
  resultBox.classList.toggle("audio-only", AUDIO_EXTENSIONS.includes(ext));
  media.src = src;
}

/** Index of the cue playing at `t`, or -1 between cues. */
function cueAt(t: number): number {
  let lo = 0;
  let hi = cues.length - 1;
  let found = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (cues[mid].start <= t) {
      found = mid;
      lo = mid + 1;
    } else hi = mid - 1;
  }
  return found >= 0 && t < cues[found].end ? found : -1;
}

function setActiveCue(i: number) {
  if (i === activeCue) return;
  cueList.querySelector(".cue[aria-current]")?.removeAttribute("aria-current");
  activeCue = i;
  if (i < 0) return;
  const button = cueList.querySelector<HTMLButtonElement>(`.cue[data-index="${i}"]`);
  if (!button) return;
  button.setAttribute("aria-current", "true");
  button.scrollIntoView({ block: "center", behavior: "smooth" });
}

media.addEventListener("timeupdate", () => setActiveCue(cueAt(media.currentTime)));
media.addEventListener("seeked", () => setActiveCue(cueAt(media.currentTime)));
// Audio-only recordings (or a video file without a picture) keep just the player bar.
media.addEventListener("loadedmetadata", () => resultBox.classList.toggle("audio-only", media.videoWidth === 0));
media.addEventListener("error", () => {
  if (media.getAttribute("src")) mediaError.hidden = false;
});

cueList.addEventListener("click", (ev) => {
  const button = (ev.target as Element).closest<HTMLButtonElement>(".cue");
  if (!button) return;
  const i = Number(button.dataset.index);
  // A hair past the start, so the seeked position falls inside this cue even when it starts where the previous ends.
  media.currentTime = cues[i].start + 0.001;
  setActiveCue(i);
  void media.play().catch(() => {});
});

/** The recording's path with `.vtt` in place of its extension, like the CLI's default output. */
function defaultVttPath(p: string): string {
  const dot = p.lastIndexOf(".");
  const base = dot > Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\")) ? p.slice(0, dot) : p;
  return `${base}.vtt`;
}

exportButton.addEventListener("click", async () => {
  if (runPath === null) return;
  const target = await save({ defaultPath: defaultVttPath(runPath), filters: [{ name: "WebVTT", extensions: ["vtt"] }] });
  if (target === null) return;
  errorBox.hidden = true;
  try {
    const out = await invoke<Exported>("export_transcript", { path: target });
    exportStatus.textContent = `Saved ${out.vtt} and ${out.json}.`;
    exportStatus.hidden = false;
  } catch (e) {
    exportStatus.hidden = true;
    showError(e as RunError);
  }
});

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
  media.pause();
  runPath = path;
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
