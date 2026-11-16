//! Project files: one SQLite file per meeting (document model, e.g. `meeting.mom`) holding its runs and their
//! intermediate data, so work on a meeting can be saved, reopened and resumed.
//!
//! - The recording is referenced, not copied: its path (absolute and relative to the project), size, mtime and
//!   a hash of the first 4 MiB. [`Project::recording`] finds it again or says it moved; [`Project::locate`]
//!   points the project at the new place after checking it is the same file.
//! - A run stores its settings (`config_json`, see [`Config::settings_json`]), the diarization turns with their
//!   speaker embeddings, the chunks, the words of each chunk as soon as the chunk is done, the exclusive turns
//!   and the cues. A run that did not finish (`running`, `cancelled`, `failed`) with the same settings is
//!   resumed by [`Project::transcribe`] from the next chunk.
//! - Schema version in `PRAGMA user_version`, migrations in [`MIGRATIONS`]; `PRAGMA application_id` marks the
//!   file as a project.

use mo_meetings_core::assembly;
use mo_meetings_core::diarize::Diarization;
use mo_meetings_core::diarize::merge::MergeReport;
use mo_meetings_core::pipeline::{self, ChunkResult, Config, Progress, Resume, RunStore, Transcript};
use mo_meetings_core::{Chunk, Turn, Word};
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{SystemTime, UNIX_EPOCH};

/// `PRAGMA application_id` of project files ("mom1").
pub const APPLICATION_ID: i32 = 0x6d6f_6d31;

/// Bytes of the recording hashed to recognize it after it moved.
const HASH_BYTES: u64 = 4 << 20;

/// Schema migrations; migration `i` brings `user_version` from `i` to `i + 1`.
const MIGRATIONS: &[&str] = &[r#"
CREATE TABLE project (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    created_at INTEGER NOT NULL,
    tool TEXT NOT NULL,
    recording_path TEXT NOT NULL,          -- absolute
    recording_rel TEXT,                    -- relative to the project file's directory, when there is one
    recording_size INTEGER NOT NULL,
    recording_mtime INTEGER,               -- unix seconds
    recording_hash TEXT NOT NULL           -- SHA-256 of the first 4 MiB
);
CREATE TABLE run (
    id INTEGER PRIMARY KEY,
    parent_run INTEGER REFERENCES run(id),
    kind TEXT NOT NULL CHECK (kind IN ('full', 'asr', 'segment', 'assembly')),
    status TEXT NOT NULL CHECK (status IN ('running', 'done', 'cancelled', 'failed')),
    config_json TEXT NOT NULL,
    profile TEXT,
    model TEXT NOT NULL,
    language TEXT,
    prompt TEXT,
    range_start REAL,
    range_end REAL,
    tool TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    finished_at INTEGER,
    error TEXT,
    num_speakers INTEGER,                  -- diarization
    num_speakers_estimated INTEGER,
    merge_report_json TEXT,
    meta_json TEXT                         -- pipeline::Metadata of the finished run
);
CREATE TABLE turn (
    run_id INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
    exclusive INTEGER NOT NULL,            -- 0 = diarization (may overlap), 1 = non-overlapping, used for words
    idx INTEGER NOT NULL,
    start REAL NOT NULL,
    "end" REAL NOT NULL,
    speaker INTEGER NOT NULL,
    embedding BLOB,                        -- f32 little-endian, diarization turns only
    PRIMARY KEY (run_id, exclusive, idx)
);
CREATE TABLE chunk (
    run_id INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    start REAL NOT NULL,
    "end" REAL NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'done')),
    language TEXT,                         -- used or detected
    PRIMARY KEY (run_id, idx)
);
CREATE TABLE word (
    run_id INTEGER NOT NULL,
    chunk_idx INTEGER NOT NULL,
    idx INTEGER NOT NULL,
    text TEXT NOT NULL,
    start REAL NOT NULL,
    "end" REAL NOT NULL,
    prob REAL,
    speaker INTEGER,                       -- internal id, set when the run is assembled
    PRIMARY KEY (run_id, chunk_idx, idx),
    FOREIGN KEY (run_id, chunk_idx) REFERENCES chunk(run_id, idx) ON DELETE CASCADE
);
CREATE TABLE cue (
    run_id INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE,
    idx INTEGER NOT NULL,
    start REAL NOT NULL,
    "end" REAL NOT NULL,
    speaker INTEGER NOT NULL,              -- display number (Speaker N)
    text TEXT NOT NULL,
    PRIMARY KEY (run_id, idx)
);
-- Which run's text is used for each chunk of the current transcript.
CREATE TABLE selection (
    chunk_idx INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES run(id) ON DELETE CASCADE
);
"#];

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Core(#[from] mo_meetings_core::Error),
    #[error("project file {path}: {source}")]
    Sqlite { path: PathBuf, source: rusqlite::Error },
    #[error("{0} is not a mo-meetings project")]
    NotAProject(PathBuf),
    #[error("{path} was written by a newer mo-meetings (schema {found}, this version reads up to {supported})")]
    NewerSchema { path: PathBuf, found: i32, supported: i32 },
    #[error("project {0} already exists")]
    Exists(PathBuf),
    #[error(
        "recording not found at {0} (moved? give its new path, e.g. `mo-meetings transcribe <new path> --project <file>`)"
    )]
    RecordingMissing(PathBuf),
    #[error("{found} is not the project's recording ({reason})")]
    RecordingMismatch { found: PathBuf, reason: String },
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

impl Error {
    /// Exit code in the CLI's scheme: 2 bad input, 3 model missing, 4 processing failed.
    pub fn exit_code(&self) -> i32 {
        match self {
            Error::Core(e) => e.exit_code(),
            Error::Sqlite { .. } | Error::Io(_) => 4,
            _ => 2,
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// What identifies a recording: size and a hash of its start (cheap on hour-long files).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recording {
    pub path: PathBuf,
    pub size: u64,
    pub mtime: Option<i64>,
    pub hash: String,
}

impl Recording {
    pub fn identify(path: &Path) -> Result<Recording> {
        let path = std::path::absolute(path)?;
        let md = std::fs::metadata(&path).map_err(|_| Error::RecordingMissing(path.clone()))?;
        if !md.is_file() {
            return Err(Error::RecordingMissing(path));
        }
        let mtime = md.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs() as i64);
        let mut h = Sha256::new();
        let mut f = std::fs::File::open(&path)?.take(HASH_BYTES);
        let mut buf = vec![0u8; 1 << 16];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            h.update(&buf[..n]);
        }
        let hash = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
        Ok(Recording { path, size: md.len(), mtime, hash })
    }

    /// `Ok` when `other` has the same size and hash.
    fn check_same(&self, other: &Recording) -> Result<()> {
        let reason = if self.size != other.size {
            format!("size {} B, expected {} B", other.size, self.size)
        } else if self.hash != other.hash {
            "content differs".to_string()
        } else {
            return Ok(());
        };
        Err(Error::RecordingMismatch { found: other.path.clone(), reason })
    }
}

/// How a [`Project::transcribe`] call went.
pub struct Outcome {
    pub run_id: i64,
    pub transcript: Transcript,
    /// Chunks taken over from an interrupted run (0 for a new run).
    pub resumed_chunks: usize,
    /// Diarization taken over from an interrupted run.
    pub resumed_diarization: bool,
}

/// A run as listed by [`Project::runs`].
#[derive(Clone, Debug, PartialEq)]
pub struct RunInfo {
    pub id: i64,
    pub kind: String,
    pub status: String,
    pub model: String,
    pub language: Option<String>,
    pub profile: Option<String>,
    pub chunks: usize,
    pub chunks_done: usize,
    pub cues: usize,
}

pub struct Project {
    conn: Connection,
    path: PathBuf,
}

fn now() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

fn tool() -> String {
    format!("mo-meetings {}", env!("CARGO_PKG_VERSION"))
}

fn path_str(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// `target` relative to `base` when it is below it (the usual case: recording next to the project).
fn relative(target: &Path, base: &Path) -> Option<String> {
    target.strip_prefix(base).ok().map(path_str)
}

impl Project {
    /// Creates a new project for `recording` at `path` (fails when `path` exists).
    pub fn create(path: &Path, recording: &Path) -> Result<Project> {
        if path.exists() {
            return Err(Error::Exists(path.to_path_buf()));
        }
        let rec = Recording::identify(recording)?;
        let mut p = Project::connect(path)?;
        p.migrate()?;
        let rel = relative(&rec.path, &p.dir());
        p.conn
            .execute(
                "INSERT INTO project (id, created_at, tool, recording_path, recording_rel, recording_size,
                    recording_mtime, recording_hash) VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![now(), tool(), path_str(&rec.path), rel, rec.size as i64, rec.mtime, rec.hash],
            )
            .map_err(|e| p.sql(e))?;
        Ok(p)
    }

    /// Opens an existing project and brings its schema up to date.
    pub fn open(path: &Path) -> Result<Project> {
        if !path.is_file() {
            return Err(Error::Core(mo_meetings_core::Error::InvalidInput(format!(
                "project not found: {}",
                path.display()
            ))));
        }
        let mut p = Project::connect(path)?;
        let app: i32 = p.conn.pragma_query_value(None, "application_id", |r| r.get(0)).map_err(|e| p.sql(e))?;
        if app != APPLICATION_ID {
            return Err(Error::NotAProject(path.to_path_buf()));
        }
        p.migrate()?;
        Ok(p)
    }

    fn connect(path: &Path) -> Result<Project> {
        let path = std::path::absolute(path)?;
        let conn = Connection::open(&path).map_err(|e| Error::Sqlite { path: path.clone(), source: e })?;
        let p = Project { conn, path };
        p.conn.pragma_update(None, "foreign_keys", true).map_err(|e| p.sql(e))?;
        Ok(p)
    }

    fn sql(&self, source: rusqlite::Error) -> Error {
        Error::Sqlite { path: self.path.clone(), source }
    }

    fn dir(&self) -> PathBuf {
        self.path.parent().map(Path::to_path_buf).unwrap_or_default()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn migrate(&mut self) -> Result<()> {
        let path = self.path.clone();
        let sql = |source| Error::Sqlite { path: path.clone(), source };
        let version: i32 = self.conn.pragma_query_value(None, "user_version", |r| r.get(0)).map_err(sql)?;
        let supported = MIGRATIONS.len() as i32;
        if version > supported {
            return Err(Error::NewerSchema { path: self.path.clone(), found: version, supported });
        }
        for (i, m) in MIGRATIONS.iter().enumerate().skip(version.max(0) as usize) {
            let tx = self.conn.transaction().map_err(sql)?;
            tx.execute_batch(m).map_err(sql)?;
            tx.pragma_update(None, "user_version", i as i32 + 1).map_err(sql)?;
            tx.pragma_update(None, "application_id", APPLICATION_ID).map_err(sql)?;
            tx.commit().map_err(sql)?;
        }
        Ok(())
    }

    fn stored_recording(&self) -> Result<(Recording, Option<String>)> {
        self.conn
            .query_row(
                "SELECT recording_path, recording_rel, recording_size, recording_mtime, recording_hash FROM project",
                [],
                |r| {
                    Ok((
                        Recording {
                            path: PathBuf::from(r.get::<_, String>(0)?),
                            size: r.get::<_, i64>(2)? as u64,
                            mtime: r.get(3)?,
                            hash: r.get(4)?,
                        },
                        r.get(1)?,
                    ))
                },
            )
            .map_err(|e| self.sql(e))
    }

    /// The recording's current path: the stored absolute path, else the stored relative one (project and
    /// recording moved together). Checks size and hash.
    pub fn recording(&self) -> Result<PathBuf> {
        let (stored, rel) = self.stored_recording()?;
        let mut candidates = vec![stored.path.clone()];
        candidates.extend(rel.map(|r| self.dir().join(r)));
        for c in candidates {
            if c.is_file() {
                let found = Recording::identify(&c)?;
                stored.check_same(&found)?;
                return Ok(found.path);
            }
        }
        Err(Error::RecordingMissing(stored.path))
    }

    /// Points the project at the recording's new place, after checking it is the same file.
    pub fn locate(&mut self, recording: &Path) -> Result<PathBuf> {
        let (stored, _) = self.stored_recording()?;
        let found = Recording::identify(recording)?;
        stored.check_same(&found)?;
        let rel = relative(&found.path, &self.dir());
        self.conn
            .execute(
                "UPDATE project SET recording_path = ?1, recording_rel = ?2, recording_mtime = ?3",
                params![path_str(&found.path), rel, found.mtime],
            )
            .map_err(|e| self.sql(e))?;
        Ok(found.path)
    }

    /// The runs, oldest first.
    pub fn runs(&self) -> Result<Vec<RunInfo>> {
        let mut st = self
            .conn
            .prepare(
                "SELECT id, kind, status, model, language, profile,
                    (SELECT count(*) FROM chunk c WHERE c.run_id = run.id),
                    (SELECT count(*) FROM chunk c WHERE c.run_id = run.id AND c.status = 'done'),
                    (SELECT count(*) FROM cue q WHERE q.run_id = run.id)
                 FROM run ORDER BY id",
            )
            .map_err(|e| self.sql(e))?;
        let rows = st
            .query_map([], |r| {
                Ok(RunInfo {
                    id: r.get(0)?,
                    kind: r.get(1)?,
                    status: r.get(2)?,
                    model: r.get(3)?,
                    language: r.get(4)?,
                    profile: r.get(5)?,
                    chunks: r.get::<_, i64>(6)? as usize,
                    chunks_done: r.get::<_, i64>(7)? as usize,
                    cues: r.get::<_, i64>(8)? as usize,
                })
            })
            .and_then(|rows| rows.collect::<rusqlite::Result<Vec<_>>>())
            .map_err(|e| self.sql(e))?;
        Ok(rows)
    }

    /// Transcribes the project's recording with `cfg` and stores the run. An unfinished full run with the same
    /// settings is resumed: its diarization and finished chunks are reused, the remaining chunks transcribed.
    pub fn transcribe(
        &mut self,
        cfg: &Config,
        progress: &mut dyn FnMut(Progress),
        cancel: &AtomicBool,
    ) -> Result<Outcome> {
        self.transcribe_using(cfg, &mut |recording, resume, store| {
            pipeline::run_with(recording, cfg, resume, store, progress, cancel)
        })
    }

    /// [`Project::transcribe`] with the pipeline given by the caller (other backends, tests).
    pub fn transcribe_using(
        &mut self,
        cfg: &Config,
        run: &mut dyn FnMut(&Path, Resume, &mut dyn RunStore) -> mo_meetings_core::Result<Transcript>,
    ) -> Result<Outcome> {
        let recording = self.recording()?;
        let settings = cfg.settings_json().to_string();
        let unfinished: Option<i64> = self
            .conn
            .query_row(
                "SELECT id FROM run WHERE kind = 'full' AND status != 'done' AND config_json = ?1
                 ORDER BY id DESC LIMIT 1",
                [&settings],
                |r| r.get(0),
            )
            .optional()
            .map_err(|e| self.sql(e))?;
        let (run_id, resume) = match unfinished {
            Some(id) => {
                self.conn
                    .execute(
                        "UPDATE run SET status = 'running', error = NULL, profile = ?2, tool = ?3 WHERE id = ?1",
                        params![id, cfg.profile, tool()],
                    )
                    .map_err(|e| self.sql(e))?;
                (id, self.load_resume(id)?)
            }
            None => {
                self.conn
                    .execute(
                        "INSERT INTO run (kind, status, config_json, profile, model, language, prompt, tool, started_at)
                         VALUES ('full', 'running', ?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                        params![settings, cfg.profile, cfg.asr_model, cfg.asr.language, cfg.asr.prompt, tool(), now()],
                    )
                    .map_err(|e| self.sql(e))?;
                (self.conn.last_insert_rowid(), Resume::default())
            }
        };
        let resumed_chunks = resume.done.len();
        let resumed_diarization = resume.prepared.is_some();

        let mut store = DbStore { project: self, run_id };
        let result = run(&recording, resume, &mut store);
        match result {
            Ok(transcript) => {
                self.finish(run_id, &transcript)?;
                Ok(Outcome { run_id, transcript, resumed_chunks, resumed_diarization })
            }
            Err(e) => {
                let status = if matches!(e, mo_meetings_core::Error::Cancelled) { "cancelled" } else { "failed" };
                self.conn
                    .execute(
                        "UPDATE run SET status = ?2, error = ?3 WHERE id = ?1",
                        params![run_id, status, e.to_string()],
                    )
                    .map_err(|e| self.sql(e))?;
                Err(e.into())
            }
        }
    }

    fn load_resume(&self, run_id: i64) -> Result<Resume> {
        let sql = |e| self.sql(e);
        let diar: Option<(Option<i64>, Option<bool>, Option<String>)> = self
            .conn
            .query_row(
                "SELECT num_speakers, num_speakers_estimated, merge_report_json FROM run WHERE id = ?1",
                [run_id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()
            .map_err(sql)?;
        let Some((Some(num_speakers), Some(estimated), report)) = diar else {
            return Ok(Resume::default()); // interrupted before the diarization was stored
        };
        let report: MergeReport = report.and_then(|r| serde_json::from_str(&r).ok()).unwrap_or_default();

        let mut st = self
            .conn
            .prepare(
                "SELECT start, \"end\", speaker, embedding FROM turn WHERE run_id = ?1 AND exclusive = 0 ORDER BY idx",
            )
            .map_err(sql)?;
        let mut turns = Vec::new();
        let mut embeddings = Vec::new();
        let mut rows = st.query([run_id]).map_err(sql)?;
        while let Some(r) = rows.next().map_err(sql)? {
            turns.push(Turn::new(
                r.get(0).map_err(sql)?,
                r.get(1).map_err(sql)?,
                r.get::<_, i64>(2).map_err(sql)? as usize,
            ));
            embeddings.push(r.get::<_, Option<Vec<u8>>>(3).map_err(sql)?.map(|b| from_blob(&b)));
        }
        drop(rows);
        let diarization =
            Diarization { turns, num_speakers: num_speakers as usize, estimated, report, timings: vec![], embeddings };

        let mut st = self
            .conn
            .prepare("SELECT idx, start, \"end\", status, language FROM chunk WHERE run_id = ?1 ORDER BY idx")
            .map_err(sql)?;
        let chunk_rows: Vec<(usize, Chunk, bool, Option<String>)> = st
            .query_map([run_id], |r| {
                Ok((
                    r.get::<_, i64>(0)? as usize,
                    Chunk { start: r.get(1)?, end: r.get(2)? },
                    r.get::<_, String>(3)? == "done",
                    r.get(4)?,
                ))
            })
            .and_then(|rows| rows.collect::<rusqlite::Result<Vec<_>>>())
            .map_err(sql)?;
        let chunks: Vec<Chunk> = chunk_rows.iter().map(|c| c.1).collect();

        let mut st = self
            .conn
            .prepare("SELECT text, start, \"end\", prob FROM word WHERE run_id = ?1 AND chunk_idx = ?2 ORDER BY idx")
            .map_err(sql)?;
        let mut done = Vec::new();
        for (index, chunk, is_done, language) in chunk_rows {
            if !is_done {
                continue;
            }
            let words = st
                .query_map(params![run_id, index as i64], |r| {
                    Ok(Word { prob: r.get(3)?, ..Word::new(r.get::<_, String>(0)?, r.get(1)?, r.get(2)?) })
                })
                .and_then(|rows| rows.collect::<rusqlite::Result<Vec<_>>>())
                .map_err(sql)?;
            done.push(ChunkResult { index, chunk, words, language });
        }
        Ok(Resume { prepared: Some((diarization, chunks)), done })
    }

    /// Stores the assembled result of a finished run and selects its chunks.
    fn finish(&mut self, run_id: i64, tr: &Transcript) -> Result<()> {
        let path = self.path.clone();
        let sql = |source| Error::Sqlite { path: path.clone(), source };
        let tx = self.conn.transaction().map_err(sql)?;
        // Speakers of the stored words: assignment is per word, so per chunk gives the same result.
        let mut words: Vec<(i64, i64, Word)> = Vec::new();
        {
            let mut st =
                tx.prepare("SELECT chunk_idx, idx, text, start, \"end\" FROM word WHERE run_id = ?1").map_err(sql)?;
            let mut rows = st.query([run_id]).map_err(sql)?;
            while let Some(r) = rows.next().map_err(sql)? {
                let w = Word::new(r.get::<_, String>(2).map_err(sql)?, r.get(3).map_err(sql)?, r.get(4).map_err(sql)?);
                words.push((r.get(0).map_err(sql)?, r.get(1).map_err(sql)?, w));
            }
        }
        let mut only: Vec<Word> = words.iter().map(|w| w.2.clone()).collect();
        assembly::assign_speakers(&mut only, &tr.turns);
        {
            let mut st = tx
                .prepare("UPDATE word SET speaker = ?4 WHERE run_id = ?1 AND chunk_idx = ?2 AND idx = ?3")
                .map_err(sql)?;
            for ((c, i, _), w) in words.iter().zip(&only) {
                st.execute(params![run_id, c, i, w.speaker.map(|s| s as i64)]).map_err(sql)?;
            }
            insert_turns(&tx, run_id, true, &tr.turns, &[]).map_err(sql)?;
            let mut st = tx
                .prepare("INSERT INTO cue (run_id, idx, start, \"end\", speaker, text) VALUES (?1, ?2, ?3, ?4, ?5, ?6)")
                .map_err(sql)?;
            for (i, q) in tr.cues.iter().enumerate() {
                st.execute(params![run_id, i as i64, q.start, q.end, q.speaker as i64, q.text]).map_err(sql)?;
            }
        }
        tx.execute("DELETE FROM selection", []).map_err(sql)?;
        tx.execute(
            "INSERT INTO selection (chunk_idx, run_id) SELECT idx, run_id FROM chunk WHERE run_id = ?1",
            [run_id],
        )
        .map_err(sql)?;
        let meta = serde_json::to_string(&tr.meta).unwrap_or_default();
        tx.execute(
            "UPDATE run SET status = 'done', finished_at = ?2, meta_json = ?3 WHERE id = ?1",
            params![run_id, now(), meta],
        )
        .map_err(sql)?;
        tx.commit().map_err(sql)
    }
}

fn to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn from_blob(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
}

fn insert_turns(
    tx: &Transaction,
    run_id: i64,
    exclusive: bool,
    turns: &[Turn],
    embeddings: &[Option<Vec<f32>>],
) -> rusqlite::Result<()> {
    let mut st = tx.prepare(
        "INSERT INTO turn (run_id, exclusive, idx, start, \"end\", speaker, embedding) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
    )?;
    for (i, t) in turns.iter().enumerate() {
        let emb = embeddings.get(i).and_then(|e| e.as_deref()).map(to_blob);
        st.execute(params![run_id, exclusive, i as i64, t.start, t.end, t.speaker as i64, emb])?;
    }
    Ok(())
}

/// Writes a run's diarization, chunks and per-chunk words as they are produced, one transaction each.
struct DbStore<'a> {
    project: &'a mut Project,
    run_id: i64,
}

impl DbStore<'_> {
    fn err(&self, e: rusqlite::Error) -> mo_meetings_core::Error {
        mo_meetings_core::Error::Io(std::io::Error::other(format!("project file {}: {e}", self.project.path.display())))
    }

    fn write_prepared(&mut self, d: &Diarization, chunks: &[Chunk]) -> rusqlite::Result<()> {
        let run_id = self.run_id;
        let tx = self.project.conn.transaction()?;
        tx.execute("DELETE FROM chunk WHERE run_id = ?1", [run_id])?;
        tx.execute("DELETE FROM turn WHERE run_id = ?1", [run_id])?;
        tx.execute(
            "UPDATE run SET num_speakers = ?2, num_speakers_estimated = ?3, merge_report_json = ?4 WHERE id = ?1",
            params![run_id, d.num_speakers as i64, d.estimated, serde_json::to_string(&d.report).unwrap_or_default()],
        )?;
        insert_turns(&tx, run_id, false, &d.turns, &d.embeddings)?;
        {
            let mut st = tx.prepare(
                "INSERT INTO chunk (run_id, idx, start, \"end\", status) VALUES (?1, ?2, ?3, ?4, 'pending')",
            )?;
            for (i, c) in chunks.iter().enumerate() {
                st.execute(params![run_id, i as i64, c.start, c.end])?;
            }
        }
        tx.commit()
    }

    fn write_chunk(&mut self, r: &ChunkResult) -> rusqlite::Result<()> {
        let run_id = self.run_id;
        let tx = self.project.conn.transaction()?;
        let idx = r.index as i64;
        tx.execute("DELETE FROM word WHERE run_id = ?1 AND chunk_idx = ?2", params![run_id, idx])?;
        {
            let mut st = tx.prepare(
                "INSERT INTO word (run_id, chunk_idx, idx, text, start, \"end\", prob) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )?;
            for (i, w) in r.words.iter().enumerate() {
                st.execute(params![run_id, idx, i as i64, w.text, w.start, w.end, w.prob])?;
            }
        }
        tx.execute(
            "UPDATE chunk SET status = 'done', language = ?3 WHERE run_id = ?1 AND idx = ?2",
            params![run_id, idx, r.language],
        )?;
        tx.commit()
    }
}

impl RunStore for DbStore<'_> {
    fn prepared(&mut self, diarization: &Diarization, chunks: &[Chunk]) -> mo_meetings_core::Result<()> {
        self.write_prepared(diarization, chunks).map_err(|e| self.err(e))
    }

    fn chunk_done(&mut self, result: &ChunkResult) -> mo_meetings_core::Result<()> {
        self.write_chunk(result).map_err(|e| self.err(e))
    }
}

#[cfg(test)]
mod tests;
