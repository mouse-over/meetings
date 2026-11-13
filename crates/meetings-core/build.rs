//! macOS/Apple Silicon: fetch Microsoft's onnxruntime release (pinned, SHA-256 verified) into
//! `<target>/<profile>/onnxruntime/`. Binaries put that directory first in their rpath (see meetings-cli's
//! build.rs), so it is used instead of the onnxruntime shipped with the sherpa-onnx release. Both are 1.28.2,
//! but the sherpa-onnx build ran diarization ~2.3x slower (measured on an M5; the PyPI wheel is fast too).
//!
//! Set `MEETINGS_SKIP_ORT_DOWNLOAD=1` to skip (offline builds; diarization is then slower).

use sha2::{Digest, Sha256};
use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const ORT_VERSION: &str = "1.28.2";
const ORT_SHA256: &str = "c4fceacfc53765d0869dc9180c31ec91054d149017a99d1e80ffe28dc79596de";

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=MEETINGS_SKIP_ORT_DOWNLOAD");
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if os != "macos" {
        return;
    }
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let profile_dir = profile_dir(&out_dir);
    // Unit and integration test binaries are not next to the shared libraries: give them absolute rpaths.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", profile_dir.join("onnxruntime").display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", profile_dir.display());
    if arch != "aarch64" || env::var_os("MEETINGS_SKIP_ORT_DOWNLOAD").is_some() {
        return;
    }
    if let Err(e) = fetch_ort(&out_dir, &profile_dir.join("onnxruntime")) {
        println!("cargo:warning=could not fetch Microsoft onnxruntime ({e}); diarization will be slower");
    }
}

/// `<target>/<profile>` from `<target>/<profile>/build/<pkg>[-hash]/[hash/]out`.
fn profile_dir(out_dir: &Path) -> PathBuf {
    out_dir
        .ancestors()
        .find(|p| p.file_name().is_some_and(|n| n == "build"))
        .and_then(Path::parent)
        .unwrap_or(out_dir)
        .to_path_buf()
}

fn fetch_ort(out_dir: &Path, dest_dir: &Path) -> Result<(), String> {
    let dest = dest_dir.join("libonnxruntime.dylib");
    if dest.is_file() {
        return Ok(());
    }
    let name = format!("onnxruntime-osx-arm64-{ORT_VERSION}");
    let archive = out_dir.join(format!("{name}.tgz"));
    if !archive.is_file() || sha256(&archive)? != ORT_SHA256 {
        let url = format!("https://github.com/microsoft/onnxruntime/releases/download/v{ORT_VERSION}/{name}.tgz");
        run(Command::new("curl").args(["-fsSL", "--retry", "3", "-o"]).arg(&archive).arg(&url))?;
        let got = sha256(&archive)?;
        if got != ORT_SHA256 {
            return Err(format!("SHA-256 mismatch for {url}: {got}"));
        }
    }
    run(Command::new("tar").arg("xzf").arg(&archive).arg("-C").arg(out_dir))?;
    std::fs::create_dir_all(dest_dir).map_err(|e| e.to_string())?;
    let lib = out_dir.join(&name).join("lib").join(format!("libonnxruntime.{ORT_VERSION}.dylib"));
    let tmp = dest_dir.join("libonnxruntime.dylib.tmp");
    std::fs::copy(&lib, &tmp).map_err(|e| format!("copy {}: {e}", lib.display()))?;
    // Same install name as the library sherpa-onnx links against, so dyld loads only this copy.
    run(Command::new("install_name_tool").args(["-id", "@rpath/libonnxruntime.dylib"]).arg(&tmp))?;
    run(Command::new("codesign").args(["-f", "-s", "-"]).arg(&tmp))?;
    std::fs::rename(&tmp, &dest).map_err(|e| e.to_string())?;
    Ok(())
}

fn sha256(path: &Path) -> Result<String, String> {
    let data = std::fs::read(path).map_err(|e| e.to_string())?;
    Ok(Sha256::digest(&data).iter().map(|b| format!("{b:02x}")).collect())
}

fn run(cmd: &mut Command) -> Result<(), String> {
    let status = cmd.status().map_err(|e| format!("{cmd:?}: {e}"))?;
    if status.success() { Ok(()) } else { Err(format!("{cmd:?} failed: {status}")) }
}
