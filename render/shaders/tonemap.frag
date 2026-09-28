#version 450

// Resolve the linear HDR scene into an _SRGB target: exposure -> tonemap.
// Output stays linear; the _SRGB target applies the OETF on store, so we do NOT
// gamma-encode here (no double correction). The target is the swapchain, or the
// LDR intermediate when FXAA is enabled — both _SRGB, so one pipeline serves.

layout(set = 0, binding = 0) uniform sampler2D u_hdr;
// Level 0 of the bloom chain (§13): the sum of every level's filtered copy
// of the HDR image, at half resolution.
layout(set = 0, binding = 1) uniform sampler2D u_bloom;

// Must match render::tonemap's push layout (a test checks its size).
layout(push_constant) uniform Push {
    float exposure;
    float bloom;        // mix strength; 0 skips bloom (its image isn't read)
    float bloom_levels; // how many levels were summed into u_bloom
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

// ACES filmic approximation (Narkowicz 2015). Compact starting point; an AgX
// curve can drop in here later without touching the pipeline.
vec3 aces(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(u_hdr, v_uv).rgb;
    // Physically based bloom: a small mix towards the blurred image, averaged
    // over its levels, so a flat image stays itself (energy-conserving).
    if (pc.bloom > 0.0) {
        vec3 bloom = texture(u_bloom, v_uv).rgb / pc.bloom_levels;
        hdr = mix(hdr, bloom, pc.bloom);
    }
    vec3 mapped = aces(hdr * pc.exposure);
    o_color = vec4(mapped, 1.0);
}
