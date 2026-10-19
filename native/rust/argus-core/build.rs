use std::env;
use std::path::PathBuf;

fn main() {
    let core = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../..");
    println!("cargo:rerun-if-changed={}", core.join("src/argus.cpp").display());
    println!("cargo:rerun-if-changed={}", core.join("include/argus.h").display());
    println!("cargo:rerun-if-changed={}", core.join("CMakeLists.txt").display());

    // Build the C++ library with CMake and link it statically.
    let dst = cmake::Config::new(&core)
        .define("ARGUS_BUILD_TESTS", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release")
        .build_target("argus_static")
        .build();
    println!("cargo:rustc-link-search=native={}/build", dst.display());
    println!("cargo:rustc-link-lib=static=argus_static");
    println!("cargo:rustc-link-lib=sqlite3");
    // llama.cpp (Homebrew by default; override with ARGUS_LLAMA_PREFIX). It is a dylib, so record an
    // rpath for the binaries Cargo links.
    println!("cargo:rerun-if-env-changed=ARGUS_LLAMA_PREFIX");
    let llama = env::var("ARGUS_LLAMA_PREFIX").unwrap_or_else(|_| "/opt/homebrew".into());
    println!("cargo:rustc-link-search=native={llama}/lib");
    println!("cargo:rustc-link-lib=llama");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{llama}/lib");
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // Raw bindings come from the real header, so the Rust side cannot drift from the C ABI.
    let bindings = bindgen::Builder::default()
        .header(core.join("include/argus.h").to_string_lossy())
        .allowlist_function("argus_.*")
        .allowlist_type("argus_.*")
        .rustified_enum("argus_status")
        .generate()
        .expect("bindgen failed on include/argus.h");
    bindings
        .write_to_file(PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs"))
        .expect("could not write bindings");
}
