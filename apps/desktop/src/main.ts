// Download the models on first start, then pick a recording, set the options, run the pipeline in the back-end and
// follow its progress, then play the recording next to its transcript.
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open, save } from "@tauri-apps/plugin-dialog";

type Stage = "load" | "decode" | "diarize" | "transcribe" | "assemble";

type ProgressEvent =
  | { kind: "started"; stage: Stage }
  | { kind: "chunk"; done: number; total: number; audioDone: number; audioTotal: number }
  | { kind: "finished"; stage: Stage; seconds: number }
  | { kind: "warning"; message: string };

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
  kind: "input" | "modelMissing" | "processing" | "busy" | "download";
  message: string;
}

interface ModelStatus {
  id: string;
  kind: "asr" | "segmentation" | "embedding";
  size: number;
  license: string;
  attribution: string;
  downloaded: boolean;
}

interface Setup {
  dir: string;
  needed: ModelStatus[];
  notice: string;
}

type DownloadEvent =
  | { kind: "progress"; id: string; index: number; count: number; done: number; total: number }
  | { kind: "retrying"; id: string; attempt: number; message: string };

const ASR_MODELS = ["whisper-large-v3-turbo", "whisper-large-v3"];
const ASR_KEY = "asrModel";

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
const setupBox = $("setup");
const asrSelect = $<HTMLSelectElement>("asr-model");
const neededBody = $<HTMLTableElement>("needed").tBodies[0];
const downloadButton = $<HTMLButtonElement>("download");
const downloadTotal = $("download-total");
const downloadBox = $("download-progress");
const downloadBar = $<HTMLProgressElement>("download-bar");
const downloadText = $("download-text");

let path: string | null = null;
/** The recording the shown transcript belongs to. */
let runPath: string | null = null;
let running = false;
let cues: Cue[] = [];
let activeCue = -1;
let downloading = false;
let asrModel = savedAsrModel();

function savedAsrModel(): string {
  try {
    const m = localStorage.getItem(ASR_KEY);
    if (m && ASR_MODELS.includes(m)) return m;
  } catch {
    // Storage unavailable: use the default.
  }
  return ASR_MODELS[0];
}

function human(bytes: number): string {
  return bytes >= 2 ** 30 ? `${(bytes / 2 ** 30).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 2 ** 20))} MB`;
}

/** Shows the setup instead of the form while a model the chosen whisper model needs is missing. */
async function refreshSetup() {
  let s: Setup;
  try {
    s = await invoke<Setup>("model_setup", { asrModel });
  } catch (e) {
    showError(e as RunError);
    return;
  }
  $("models-dir").textContent = s.dir;
  $("notice").textContent = s.notice;
  asrSelect.value = asrModel;
  for (const radio of setupBox.querySelectorAll<HTMLInputElement>('input[name="asr"]')) {
    radio.checked = radio.value === asrModel;
  }
  neededBody.replaceChildren(
    ...s.needed.map((m) => {
      const tr = document.createElement("tr");
      const name = document.createElement("td");
      name.textContent = m.id;
      const size = document.createElement("td");
      size.textContent = human(m.size);
      const license = document.createElement("td");
      license.textContent = m.license;
      const attribution = document.createElement("span");
      attribution.className = "attribution";
      attribution.textContent = m.attribution;
      license.append(attribution);
      const state = document.createElement("td");
      state.className = "muted";
      state.textContent = m.downloaded ? "Downloaded" : "";
      tr.append(name, size, license, state);
      return tr;
    }),
  );
  const missing = s.needed.filter((m) => !m.downloaded);
  const bytes = missing.reduce((sum, m) => sum + m.size, 0);
  downloadTotal.textContent = missing.length
    ? `${missing.length} model${missing.length === 1 ? "" : "s"}, ${human(bytes)}`
    : "";
  const show = missing.length > 0 || downloading;
  setupBox.hidden = !show;
  form.hidden = show;
}

function chooseAsrModel(m: string) {
  if (downloading || !ASR_MODELS.includes(m)) return;
  asrModel = m;
  try {
    localStorage.setItem(ASR_KEY, m);
  } catch {
    // Not remembered; the choice still applies now.
  }
  void refreshSetup();
}

function setDownloading(on: boolean) {
  downloading = on;
  downloadButton.disabled = on;
  for (const radio of setupBox.querySelectorAll<HTMLInputElement>('input[name="asr"]')) radio.disabled = on;
  downloadButton.textContent = on ? "Downloading…" : "Download";
}

function onDownload(e: DownloadEvent) {
  if (e.kind === "retrying") {
    downloadText.textContent = `Connection problem with ${e.id}, trying again (attempt ${e.attempt + 1})…`;
    return;
  }
  downloadBar.value = e.total > 0 ? e.done / e.total : 0;
  downloadText.textContent = `Downloading ${e.id} (${e.index} of ${e.count}) · ${human(e.done)} of ${human(e.total)}`;
}

function setRunning(on: boolean) {
  running = on;
  runButton.disabled = on || path === null;
  pickButton.disabled = on;
  for (const el of form.querySelectorAll<HTMLInputElement>("input, select, textarea")) el.disabled = on;
  runButton.textContent = on ? "Transcribing…" : "Transcribe";
}

setupBox.addEventListener("change", (ev) => {
  const t = ev.target as HTMLInputElement;
  if (t.name === "asr") chooseAsrModel(t.value);
});
asrSelect.addEventListener("change", () => chooseAsrModel(asrSelect.value));

downloadButton.addEventListener("click", async () => {
  if (downloading) return;
  errorBox.hidden = true;
  downloadBar.value = 0;
  downloadText.textContent = "Starting the download…";
  downloadBox.hidden = false;
  setDownloading(true);
  try {
    await invoke("download_models", { asrModel });
  } catch (e) {
    setDownloading(false);
    downloadBox.hidden = true;
    showError(e as RunError);
  }
});

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
  if (p.kind === "warning") {
    const li = document.createElement("li");
    li.dataset.state = "warning";
    li.textContent = `Warning: ${p.message}`;
    stagesList.append(li);
    return;
  }
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
    asrModel,
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
  if (e.payload.kind === "modelMissing") void refreshSetup();
});
void listen<DownloadEvent>("download-progress", (e) => onDownload(e.payload));
void listen("download-finished", () => {
  setDownloading(false);
  downloadBox.hidden = true;
  void refreshSetup();
});
void listen<RunError>("download-failed", (e) => {
  setDownloading(false);
  downloadButton.textContent = "Retry download";
  showError(e.payload);
});

void refreshSetup();
