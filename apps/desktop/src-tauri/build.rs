fn main() {
    // sherpa-onnx is linked as shared libraries next to the binary (its build script copies them into
    // target/<profile>/); let the binary find them there and in an app bundle's Frameworks directory.
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match os.as_str() {
        "macos" => {
            // Microsoft's onnxruntime (fetched by mo-meetings-core's build.rs) must win over the bundled one.
            println!("cargo:rustc-link-arg-bins=-Wl,-rpath,@executable_path/onnxruntime");
            println!("cargo:rustc-link-arg-bins=-Wl,-rpath,@executable_path");
            println!("cargo:rustc-link-arg-bins=-Wl,-rpath,@executable_path/../Frameworks");
            link_clang_runtime();
        }
        "linux" => println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN"),
        _ => {}
    }
    tauri_build::build();
}

/// `tauri build` sets MACOSX_DEPLOYMENT_TARGET to bundle.macOS.minimumSystemVersion, so whisper.cpp's ggml-metal
/// guards newer Metal APIs with `@available`, which calls `__isPlatformVersionAtLeast` from clang's compiler-rt.
/// rustc links with `-nodefaultlibs` and leaves compiler-rt out; add it explicitly.
fn link_clang_runtime() {
    let out = std::process::Command::new("clang").arg("--print-runtime-dir").output();
    let dir = match out {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => {
            println!("cargo:warning=`clang --print-runtime-dir` failed; not linking libclang_rt.osx.a");
            return;
        }
    };
    let lib = std::path::Path::new(&dir).join("libclang_rt.osx.a");
    if lib.is_file() {
        println!("cargo:rustc-link-arg-bins={}", lib.display());
    } else {
        println!("cargo:warning={} not found", lib.display());
    }
}
