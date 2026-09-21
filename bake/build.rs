//! intel_tex_2's prebuilt ISPC kernels reference the C++ runtime
//! (`__gxx_personality_v0`) without linking it, so on Linux the link fails
//! until libstdc++ is added. Scoped to this crate: the engine binary never
//! links the encoder.
fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-lib=stdc++");
    }
}
