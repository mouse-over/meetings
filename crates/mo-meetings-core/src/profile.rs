//! Settings profiles: named presets (TOML) that set up a run for a type of meeting in one step.
//!
//! A profile is a `<name>.toml` file in one of the profile directories:
//!
//! ```toml
//! description = "Weekly team sync"
//! extends = "team"                  # inherit from another profile
//! match = ["*sync*", "*standup*"]   # suggest this profile for recordings with matching file names
//! language = "cs"                   # 2-letter code or "auto"
//! prompt = "Weekly sync of the backend team."   # or prompt_file = "sync.txt" (relative to this file)
//! glossary = ["Kubernetes", "pull request"]
//! model = "whisper-large-v3-turbo"  # or "turbo" / "large"
//! num_speakers = 6                  # at most this many speakers
//!
//! [chunks]                          # advanced, see ChunkRules
//! max_gap = 2.0
//!
//! [cues]                            # advanced, see CueRules
//! max_duration = 6.0
//! ```
//!
//! With `extends`, values set in the profile override the parent's, glossaries are merged (parent's terms
//! first). The whisper prompt is the prompt followed by the glossary terms.
//!
//! Directories, in order: `<config>/profiles`, then the `profile_dirs` listed in `<config>/config.toml`
//! (relative to the config directory, `~/` allowed). A later directory wins when two have a profile with the
//! same name. `<config>` is `$MO_MEETINGS_CONFIG_DIR`, else `<user config dir>/mo-meetings`
//! (`~/Library/Application Support/mo-meetings` on macOS, `~/.config/mo-meetings` on Linux).

use crate::assembly::CueRules;
use crate::chunk::ChunkRules;
use crate::pipeline::Config;
use crate::{Error, Result, asr, models};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Config directory: `$MO_MEETINGS_CONFIG_DIR`, else `<user config dir>/mo-meetings`.
pub fn config_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("MO_MEETINGS_CONFIG_DIR") {
        return PathBuf::from(d);
    }
    dirs::config_dir().unwrap_or_else(std::env::temp_dir).join("mo-meetings")
}

/// `<config>/config.toml`.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfigFile {
    #[serde(default)]
    profile_dirs: Vec<PathBuf>,
}

/// The profile directories in order of precedence (later wins), read from `config_dir`'s `config.toml`.
pub fn profile_dirs(config_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut dirs = vec![config_dir.join("profiles")];
    let file = config_dir.join("config.toml");
    let text = match std::fs::read_to_string(&file) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(dirs),
        Err(e) => return Err(invalid(format!("cannot read {}: {e}", file.display()))),
    };
    let cf: ConfigFile = toml::from_str(&text).map_err(|e| invalid(format!("{}: {e}", file.display())))?;
    dirs.extend(cf.profile_dirs.iter().map(|d| expand(d, config_dir)));
    Ok(dirs)
}

/// `~/x` → home, relative → against `base`; `..` and `.` are removed so the paths read well in messages.
fn expand(p: &Path, base: &Path) -> PathBuf {
    let joined = match (p.strip_prefix("~"), dirs::home_dir()) {
        (Ok(rest), Some(home)) => home.join(rest),
        _ => base.join(p),
    };
    let mut out = PathBuf::new();
    for c in joined.components() {
        match c {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir if out.file_name().is_some() => {
                out.pop();
            }
            c => out.push(c),
        }
    }
    out
}

fn invalid(msg: impl Into<String>) -> Error {
    Error::InvalidInput(msg.into())
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ChunkOverrides {
    pub max_len: Option<f64>,
    pub max_gap: Option<f64>,
    pub pad: Option<f64>,
    pub search: Option<f64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CueOverrides {
    pub max_pause: Option<f64>,
    pub max_duration: Option<f64>,
    pub sentence_min_duration: Option<f64>,
}

/// One profile file as written.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileFile {
    description: Option<String>,
    extends: Option<String>,
    #[serde(rename = "match", default)]
    patterns: Vec<String>,
    language: Option<String>,
    prompt: Option<String>,
    prompt_file: Option<PathBuf>,
    #[serde(default)]
    glossary: Vec<String>,
    model: Option<String>,
    num_speakers: Option<u32>,
    #[serde(default)]
    chunks: ChunkOverrides,
    #[serde(default)]
    cues: CueOverrides,
}

/// A profile found in a directory.
#[derive(Clone, Debug)]
pub struct ProfileInfo {
    pub name: String,
    pub path: PathBuf,
    pub description: Option<String>,
    pub extends: Option<String>,
    /// File name patterns (`*` and `?`, case-insensitive) that suggest this profile.
    pub patterns: Vec<String>,
    /// Files with the same name in earlier directories, hidden by this one.
    pub shadowed: Vec<PathBuf>,
    file: ProfileFile,
}

/// All profiles of the profile directories.
#[derive(Clone, Debug, Default)]
pub struct Profiles {
    pub dirs: Vec<PathBuf>,
    by_name: BTreeMap<String, ProfileInfo>,
    /// Problems that did not stop loading: missing extra directories, files that do not parse.
    pub warnings: Vec<String>,
}

impl Profiles {
    /// Profiles from the directories configured in [`config_dir`].
    pub fn load() -> Result<Self> {
        let dirs = profile_dirs(&config_dir())?;
        Ok(Self::load_from(&dirs))
    }

    /// Profiles from `dirs` (later wins on the same name). The first directory may be missing; a missing later
    /// one (configured explicitly) and unreadable files become [`Profiles::warnings`].
    pub fn load_from(dirs: &[PathBuf]) -> Self {
        let mut out = Self { dirs: dirs.to_vec(), ..Default::default() };
        for (i, dir) in dirs.iter().enumerate() {
            let entries = match std::fs::read_dir(dir) {
                Ok(e) => e,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound && i == 0 => continue,
                Err(e) => {
                    out.warnings.push(format!("profile directory {}: {e}", dir.display()));
                    continue;
                }
            };
            let mut paths: Vec<PathBuf> = entries
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "toml") && p.is_file())
                .collect();
            paths.sort();
            for path in paths {
                let Some(name) = path.file_stem().and_then(|s| s.to_str()).map(str::to_string) else { continue };
                let file = match std::fs::read_to_string(&path)
                    .map_err(|e| e.to_string())
                    .and_then(|t| toml::from_str::<ProfileFile>(&t).map_err(|e| e.to_string()))
                {
                    Ok(f) => f,
                    Err(e) => {
                        out.warnings.push(format!("profile {}: {e}", path.display()));
                        continue;
                    }
                };
                let mut shadowed = Vec::new();
                if let Some(prev) = out.by_name.remove(&name) {
                    shadowed = prev.shadowed;
                    shadowed.push(prev.path);
                }
                let info = ProfileInfo {
                    name: name.clone(),
                    path,
                    description: file.description.clone(),
                    extends: file.extends.clone(),
                    patterns: file.patterns.clone(),
                    shadowed,
                    file,
                };
                out.by_name.insert(name, info);
            }
        }
        out
    }

    /// Profiles sorted by name.
    pub fn list(&self) -> impl Iterator<Item = &ProfileInfo> {
        self.by_name.values()
    }

    pub fn get(&self, name: &str) -> Option<&ProfileInfo> {
        self.by_name.get(name)
    }

    /// Names of the profiles whose `match` patterns match the file name of `recording`.
    pub fn suggest(&self, recording: &Path) -> Vec<&str> {
        let Some(file) = recording.file_name().and_then(|f| f.to_str()) else { return Vec::new() };
        self.list().filter(|p| p.patterns.iter().any(|pat| glob_match(pat, file))).map(|p| p.name.as_str()).collect()
    }

    /// The profile `name` with everything it extends merged in, prompt files read and values checked.
    pub fn resolve(&self, name: &str) -> Result<Resolved> {
        let mut chain: Vec<&ProfileInfo> = Vec::new();
        let mut next = Some(name);
        while let Some(n) = next {
            if chain.iter().any(|p| p.name == n) {
                let names: Vec<&str> = chain.iter().map(|p| p.name.as_str()).chain([n]).collect();
                return Err(invalid(format!("profile `extends` cycle: {}", names.join(" → "))));
            }
            let p = self.get(n).ok_or_else(|| match chain.last() {
                None => invalid(format!("unknown profile `{n}` ({})", self.known())),
                Some(child) => invalid(format!("profile `{}` extends unknown profile `{n}`", child.name)),
            })?;
            chain.push(p);
            next = p.file.extends.as_deref();
        }
        chain.reverse(); // base first
        let mut r = Resolved { name: name.to_string(), ..Default::default() };
        let mut prompt = None;
        for p in chain {
            r.merge(p, &mut prompt)?;
        }
        let glossary = (!r.glossary.is_empty()).then(|| r.glossary.join(", "));
        r.prompt = match (prompt, glossary) {
            (Some(p), Some(g)) => Some(format!("{p} {g}")),
            (p, g) => p.or(g),
        };
        Ok(r)
    }

    fn known(&self) -> String {
        if self.by_name.is_empty() {
            let dirs: Vec<String> = self.dirs.iter().map(|d| d.display().to_string()).collect();
            format!("no profiles in {}", dirs.join(", "))
        } else {
            format!("known: {}", self.by_name.keys().cloned().collect::<Vec<_>>().join(", "))
        }
    }
}

/// A profile with its parents merged in. `None` fields leave the run's default.
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct Resolved {
    pub name: String,
    /// The profile files, base first.
    pub sources: Vec<PathBuf>,
    /// Language code, or `auto`.
    pub language: Option<String>,
    /// The whisper prompt: the prompt followed by the glossary terms.
    pub prompt: Option<String>,
    /// The merged glossary (already part of `prompt`).
    pub glossary: Vec<String>,
    /// Registry id of the whisper model.
    pub model: Option<String>,
    pub num_speakers: Option<u32>,
    pub chunks: ChunkOverrides,
    pub cues: CueOverrides,
}

impl Resolved {
    fn merge(&mut self, p: &ProfileInfo, prompt: &mut Option<String>) -> Result<()> {
        let f = &p.file;
        let at = || p.path.display().to_string();
        self.sources.push(p.path.clone());
        if let Some(l) = &f.language {
            asr::parse_language(l).map_err(|e| invalid(format!("{}: {e}", at())))?;
            self.language = Some(l.trim().to_string());
        }
        match (&f.prompt, &f.prompt_file) {
            (Some(_), Some(_)) => return Err(invalid(format!("{}: set either `prompt` or `prompt_file`", at()))),
            (Some(t), None) => *prompt = Some(t.trim().to_string()),
            (None, Some(file)) => {
                let path = p.path.parent().map_or_else(|| file.clone(), |d| d.join(file));
                let t = std::fs::read_to_string(&path)
                    .map_err(|e| invalid(format!("{}: cannot read prompt file {}: {e}", at(), path.display())))?;
                *prompt = Some(t.trim().to_string());
            }
            (None, None) => {}
        }
        if prompt.as_deref() == Some("") {
            *prompt = None;
        }
        for term in &f.glossary {
            let term = term.trim();
            if !term.is_empty() && !self.glossary.iter().any(|g| g == term) {
                self.glossary.push(term.to_string());
            }
        }
        if let Some(m) = &f.model {
            let id = models::asr_model_id(m).ok_or_else(|| {
                invalid(format!("{}: unknown whisper model `{m}` (use turbo, large or a registry id)", at()))
            })?;
            self.model = Some(id.to_string());
        }
        if let Some(n) = f.num_speakers {
            if !(1..=64).contains(&n) {
                return Err(invalid(format!("{}: num_speakers must be 1-64", at())));
            }
            self.num_speakers = Some(n);
        }
        let positive = |v: Option<f64>, key: &str| match v {
            Some(x) if !(x.is_finite() && x >= 0.0) => Err(invalid(format!("{}: {key} must be a number >= 0", at()))),
            _ => Ok(()),
        };
        let c = &f.chunks;
        for (v, k) in [(c.max_len, "chunks.max_len"), (c.max_gap, "chunks.max_gap"), (c.pad, "chunks.pad")] {
            positive(v, k)?;
        }
        positive(c.search, "chunks.search")?;
        if c.max_len.is_some_and(|x| x < 1.0) {
            return Err(invalid(format!("{}: chunks.max_len must be at least 1 s", at())));
        }
        let q = &f.cues;
        for (v, k) in [
            (q.max_pause, "cues.max_pause"),
            (q.max_duration, "cues.max_duration"),
            (q.sentence_min_duration, "cues.sentence_min_duration"),
        ] {
            positive(v, k)?;
        }
        let (dc, dq) = (&mut self.chunks, &mut self.cues);
        dc.max_len = c.max_len.or(dc.max_len);
        dc.max_gap = c.max_gap.or(dc.max_gap);
        dc.pad = c.pad.or(dc.pad);
        dc.search = c.search.or(dc.search);
        dq.max_pause = q.max_pause.or(dq.max_pause);
        dq.max_duration = q.max_duration.or(dq.max_duration);
        dq.sentence_min_duration = q.sentence_min_duration.or(dq.sentence_min_duration);
        Ok(())
    }

    /// Writes the profile's values into `cfg` (callers apply explicit options afterwards).
    pub fn apply(&self, cfg: &mut Config) -> Result<()> {
        cfg.profile = Some(self.name.clone());
        if let Some(l) = &self.language {
            cfg.asr.language = asr::parse_language(l)?;
        }
        if let Some(p) = &self.prompt {
            cfg.asr.prompt = Some(p.clone());
        }
        if let Some(m) = &self.model {
            cfg.asr_model = m.clone();
        }
        if let Some(n) = self.num_speakers {
            cfg.num_speakers = Some(n as usize);
        }
        apply_chunks(&self.chunks, &mut cfg.chunk_rules);
        apply_cues(&self.cues, &mut cfg.cue_rules);
        Ok(())
    }
}

fn apply_chunks(c: &ChunkOverrides, r: &mut ChunkRules) {
    r.max_len = c.max_len.unwrap_or(r.max_len);
    r.max_gap = c.max_gap.unwrap_or(r.max_gap);
    r.pad = c.pad.unwrap_or(r.pad);
    r.search = c.search.unwrap_or(r.search);
}

fn apply_cues(q: &CueOverrides, r: &mut CueRules) {
    r.max_pause = q.max_pause.unwrap_or(r.max_pause);
    r.max_duration = q.max_duration.unwrap_or(r.max_duration);
    r.sentence_min_duration = q.sentence_min_duration.unwrap_or(r.sentence_min_duration);
}

/// Case-insensitive glob with `*` (any run) and `?` (one character).
pub fn glob_match(pattern: &str, text: &str) -> bool {
    let p: Vec<char> = pattern.to_lowercase().chars().collect();
    let t: Vec<char> = text.to_lowercase().chars().collect();
    let (mut i, mut j) = (0, 0);
    let mut star: Option<(usize, usize)> = None;
    while j < t.len() {
        if i < p.len() && (p[i] == '?' || p[i] == t[j]) {
            i += 1;
            j += 1;
        } else if i < p.len() && p[i] == '*' {
            star = Some((i, j));
            i += 1;
        } else if let Some((si, sj)) = star {
            i = si + 1;
            j = sj + 1;
            star = Some((si, sj + 1));
        } else {
            return false;
        }
    }
    p[i..].iter().all(|&c| c == '*')
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Tmp(PathBuf);
    impl Tmp {
        fn new(tag: &str) -> Self {
            let d = std::env::temp_dir().join(format!("mo-meetings-profile-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(&d).unwrap();
            Self(d)
        }
        fn write(&self, rel: &str, text: &str) -> PathBuf {
            let p = self.0.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, text).unwrap();
            p
        }
    }
    impl Drop for Tmp {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn extends_overrides_scalars_and_merges_glossaries() {
        let t = Tmp::new("extends");
        t.write("a/base.toml", "language = \"cs\"\nprompt = \"Company meeting.\"\nglossary = [\"Kubernetes\", \"sprint\"]\nnum_speakers = 8\n[cues]\nmax_duration = 6.0\n");
        t.write("a/team.toml", "extends = \"base\"\nlanguage = \"auto\"\nglossary = [\"sprint\", \"backlog\"]\nmodel = \"large\"\n[chunks]\nmax_gap = 2.0\n");
        let ps = Profiles::load_from(&[t.0.join("a")]);
        assert!(ps.warnings.is_empty(), "{:?}", ps.warnings);
        let r = ps.resolve("team").unwrap();
        assert_eq!(r.language.as_deref(), Some("auto"));
        assert_eq!(r.glossary, ["Kubernetes", "sprint", "backlog"]);
        assert_eq!(r.prompt.as_deref(), Some("Company meeting. Kubernetes, sprint, backlog"));
        assert_eq!(r.model.as_deref(), Some("whisper-large-v3"));
        assert_eq!(r.num_speakers, Some(8));
        assert_eq!(r.sources.len(), 2);

        let mut cfg = Config::default();
        r.apply(&mut cfg).unwrap();
        assert_eq!(cfg.profile.as_deref(), Some("team"));
        assert_eq!(cfg.asr.language, None);
        assert_eq!(cfg.asr_model, "whisper-large-v3");
        assert_eq!(cfg.num_speakers, Some(8));
        assert_eq!(cfg.chunk_rules.max_gap, 2.0);
        assert_eq!(cfg.chunk_rules.max_len, ChunkRules::default().max_len);
        assert_eq!(cfg.cue_rules.max_duration, 6.0);
    }

    #[test]
    fn prompt_file_is_relative_to_the_profile() {
        let t = Tmp::new("prompt-file");
        t.write("p/glossaries/sync.txt", "  Weekly sync.\n");
        t.write("p/sync.toml", "prompt_file = \"glossaries/sync.txt\"\n");
        t.write("p/both.toml", "prompt = \"x\"\nprompt_file = \"glossaries/sync.txt\"\n");
        let ps = Profiles::load_from(&[t.0.join("p")]);
        assert_eq!(ps.resolve("sync").unwrap().prompt.as_deref(), Some("Weekly sync."));
        assert!(ps.resolve("both").is_err());
    }

    #[test]
    fn later_directory_wins_and_remembers_the_shadowed_file() {
        let t = Tmp::new("dirs");
        let user = t.write("user/x.toml", "language = \"cs\"\n");
        let team = t.write("team/x.toml", "language = \"en\"\n");
        let ps = Profiles::load_from(&[t.0.join("user"), t.0.join("team")]);
        let x = ps.get("x").unwrap();
        assert_eq!(x.path, team);
        assert_eq!(x.shadowed, [user]);
        assert_eq!(ps.resolve("x").unwrap().language.as_deref(), Some("en"));
    }

    #[test]
    fn cycles_unknown_names_and_bad_values_are_errors() {
        let t = Tmp::new("errors");
        t.write("p/a.toml", "extends = \"b\"\n");
        t.write("p/b.toml", "extends = \"a\"\n");
        t.write("p/orphan.toml", "extends = \"nope\"\n");
        t.write("p/lang.toml", "language = \"czech\"\n");
        t.write("p/model.toml", "model = \"tiny\"\n");
        t.write("p/typo.toml", "langauge = \"cs\"\n");
        let ps = Profiles::load_from(&[t.0.join("p")]);
        assert!(ps.resolve("a").unwrap_err().to_string().contains("cycle"));
        assert!(ps.resolve("orphan").unwrap_err().to_string().contains("`nope`"));
        assert!(ps.resolve("lang").is_err());
        assert!(ps.resolve("model").is_err());
        assert!(ps.resolve("missing").unwrap_err().to_string().contains("known:"));
        // An unknown key makes the file unusable and is reported, not silently ignored.
        assert!(ps.get("typo").is_none());
        assert!(ps.warnings.iter().any(|w| w.contains("typo.toml")), "{:?}", ps.warnings);
    }

    #[test]
    fn missing_dirs() {
        let t = Tmp::new("missing");
        let ps = Profiles::load_from(&[t.0.join("none"), t.0.join("extra")]);
        assert_eq!(ps.warnings.len(), 1, "only the explicitly configured directory is reported");
        assert!(ps.resolve("x").unwrap_err().to_string().contains("no profiles in"));
    }

    #[test]
    fn config_file_lists_extra_dirs() {
        let t = Tmp::new("config");
        assert_eq!(profile_dirs(&t.0).unwrap(), [t.0.join("profiles")]);
        t.write("config.toml", "profile_dirs = [\"team\", \"/abs/dir\", \"../shared/./p\"]\n");
        let parent = t.0.parent().unwrap();
        assert_eq!(
            profile_dirs(&t.0).unwrap(),
            [t.0.join("profiles"), t.0.join("team"), PathBuf::from("/abs/dir"), parent.join("shared/p")]
        );
        t.write("config.toml", "profile_dir = [\"x\"]\n");
        assert!(profile_dirs(&t.0).is_err());
    }

    #[test]
    fn suggests_by_file_name() {
        let t = Tmp::new("suggest");
        t.write("p/sync.toml", "match = [\"*weekly*sync*\"]\n");
        t.write("p/other.toml", "match = [\"interview-??.mp4\"]\n");
        let ps = Profiles::load_from(&[t.0.join("p")]);
        assert_eq!(ps.suggest(Path::new("/rec/2026-11-16 Weekly Team Sync.mp4")), ["sync"]);
        assert_eq!(ps.suggest(Path::new("interview-07.mp4")), ["other"]);
        assert!(ps.suggest(Path::new("interview-107.mp4")).is_empty());
    }

    #[test]
    fn glob() {
        assert!(glob_match("*", ""));
        assert!(glob_match("a*b*c", "aXXbYc"));
        assert!(glob_match("A?C", "abc"));
        assert!(!glob_match("a*b", "aXc"));
        assert!(glob_match("*.mp4", "x.MP4"));
        assert!(!glob_match("a", "ab"));
    }
}
