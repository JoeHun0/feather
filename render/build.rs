//! Compiles every GLSL shader in `shaders/` to SPIR-V at build time.
//! Output lands in OUT_DIR as `<name>.spv`; embed with include_bytes! (see below).
//! A shader syntax error becomes a BUILD error, and there's no runtime compiler.
//!
//! Placement: the crate that owns your pipelines (the `render` crate per the
//! architecture doc), at the crate root next to Cargo.toml, with a `shaders/`
//! dir beside `src/`. Add to that crate's Cargo.toml:
//!     [build-dependencies]
//!     shaderc = "0.8"

use std::{env, fs, path::PathBuf};

fn main() {
    let shader_dir = PathBuf::from("shaders");
    println!("cargo:rerun-if-changed=shaders");
    if !shader_dir.exists() {
        return; // nothing to compile yet
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let compiler = shaderc::Compiler::new().expect("failed to init shaderc");
    let mut options = shaderc::CompileOptions::new().expect("failed to init shaderc options");
    options.set_target_env(
        shaderc::TargetEnv::Vulkan,
        shaderc::EnvVersion::Vulkan1_3 as u32,
    );

    let release = env::var("PROFILE").unwrap_or_default() == "release";
    options.set_optimization_level(if release {
        shaderc::OptimizationLevel::Performance
    } else {
        shaderc::OptimizationLevel::Zero
    });
    if !release {
        // Source-level shader debugging in RenderDoc. Comment out if captures bloat.
        options.set_generate_debug_info();
    }

    for entry in fs::read_dir(&shader_dir).expect("read shaders/") {
        let path = entry.expect("dir entry").path();
        let kind = match path.extension().and_then(|e| e.to_str()) {
            Some("vert") => shaderc::ShaderKind::Vertex,
            Some("frag") => shaderc::ShaderKind::Fragment,
            Some("comp") => shaderc::ShaderKind::Compute,
            Some("geom") => shaderc::ShaderKind::Geometry,
            _ => continue, // skip .glsl includes, README, etc.
        };
        println!("cargo:rerun-if-changed={}", path.display());

        let src = fs::read_to_string(&path).expect("read shader source");
        let name = path.file_name().unwrap().to_str().unwrap();
        let artifact = compiler
            .compile_into_spirv(&src, kind, name, "main", Some(&options))
            .unwrap_or_else(|e| panic!("shader {name}:\n{e}"));

        fs::write(out_dir.join(format!("{name}.spv")), artifact.as_binary_u8())
            .expect("write .spv");
    }
}
