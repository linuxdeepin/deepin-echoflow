// Re-emit the runtime rpath for THIS crate's own test and bin binaries.
//
// `crispasr-sys/build.rs` links libcrispasr and emits `cargo:rustc-link-arg`
// rpath entries — but those apply only to the crate that emitted them. A
// dependent's test binaries get none, so `cargo test -p crispasr` linked with
// zero LC_RPATH entries and failed at load time with
//
//     dyld: Library not loaded: @rpath/libcrispasr.0.8.28.dylib
//
// which is why running the integration suite needed DYLD_LIBRARY_PATH set by
// hand. `-tests` / `-bins` / `-benches` are the per-target forms that do apply
// to this crate's own artifacts.
//
// The directories arrive via cargo's `links` metadata channel: crispasr-sys
// declares `links = "crispasr"` and prints `cargo:libdir=` / `cargo:ggmldir=`,
// which cargo hands to a direct dependent as DEP_CRISPASR_LIBDIR / _GGMLDIR.
// If they are absent (a pre-installed system library via CRISPASR_LIB_DIR, or
// a future change to the sys crate) this is a no-op rather than an error.

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "windows" {
        return; // DLLs are found next to the .exe; no rpath concept.
    }

    let mut dirs: Vec<String> = Vec::new();
    for key in ["DEP_CRISPASR_LIBDIR", "DEP_CRISPASR_GGMLDIR"] {
        if let Ok(d) = env::var(key) {
            if !d.is_empty() {
                dirs.push(d);
            }
        }
    }
    // Relative entries so a bundled consumer (Tauri app, packaged binary) still
    // resolves after the library has been copied next to or below the binary —
    // the same pair crispasr-sys adds for its own artifacts.
    match target_os.as_str() {
        "macos" => {
            dirs.push("@executable_path/../Frameworks".to_string());
            dirs.push("@loader_path/../Frameworks".to_string());
        }
        _ => {
            dirs.push("$ORIGIN/../lib".to_string());
            dirs.push("$ORIGIN".to_string());
        }
    }

    // Plain `rustc-link-arg` rather than the per-kind forms. `-tests` covers
    // integration test binaries but NOT the lib-test target (the `#[cfg(test)]`
    // unit tests compiled from src/lib.rs), which is a separate kind — that one
    // still died with "Library not loaded: @rpath/libcrispasr.1.dylib". The
    // plain form applies to every binary artifact this crate produces
    // (benchmarks, binaries, cdylibs, examples and all tests), and unlike
    // `-bins` it does not hard-error when the package has no bin target.
    for d in dirs {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{d}");
    }
}
