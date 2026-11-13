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
        }
        "linux" => println!("cargo:rustc-link-arg-bins=-Wl,-rpath,$ORIGIN"),
        _ => {}
    }
    tauri_build::build();
}
