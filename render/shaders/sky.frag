#version 450

// Fullscreen background sky. Reconstructs a world-space view ray per pixel from
// the inverse view-projection and evaluates the same procedural sky the mesh
// shader reflects for IBL — here with a sharp sun disk so the sun is visible.

layout(push_constant) uniform Push {
    mat4 inv_view_proj;
    vec4 camera_pos;  // xyz
    vec4 light_dir;   // xyz = direction the light travels
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color; // linear HDR

// The level's atmosphere (§13): the same specialization constants as
// mesh.frag, which render::Environment fills. Defaults = Environment::default()
// (a test checks). sky() and the fog functions are copies of mesh.frag's, text
// for text (a test checks); the sharp sun disk is added in main() alone.
layout(constant_id = 3) const float SKY_ZENITH_R = 0.10;
layout(constant_id = 4) const float SKY_ZENITH_G = 0.22;
layout(constant_id = 5) const float SKY_ZENITH_B = 0.55;
layout(constant_id = 6) const float SKY_HORIZON_R = 0.55;
layout(constant_id = 7) const float SKY_HORIZON_G = 0.65;
layout(constant_id = 8) const float SKY_HORIZON_B = 0.85;
layout(constant_id = 9) const float SKY_GROUND_R = 0.17;
layout(constant_id = 10) const float SKY_GROUND_G = 0.18;
layout(constant_id = 11) const float SKY_GROUND_B = 0.19;
layout(constant_id = 12) const float SKY_SUN_R = 1.0;
layout(constant_id = 13) const float SKY_SUN_G = 0.95;
layout(constant_id = 14) const float SKY_SUN_B = 0.85;
layout(constant_id = 15) const float SKY_INTENSITY = 1.0;
layout(constant_id = 16) const float SUN_GLOW = 0.6;
layout(constant_id = 17) const float SUN_DISK = 60.0;
layout(constant_id = 18) const float FOG_DENSITY = 0.010; // per metre, at FOG_HEIGHT
layout(constant_id = 19) const float FOG_HEIGHT = 0.0;
layout(constant_id = 20) const float FOG_FALLOFF = 0.0;
layout(constant_id = 21) const float FOG_R = 0.5;
layout(constant_id = 22) const float FOG_G = 0.5;
layout(constant_id = 23) const float FOG_B = 0.5;
layout(constant_id = 24) const float FOG_SKY = 1.0; // 1: the fog takes the sky's colour
layout(constant_id = 25) const float FOG_SUN = 0.0;
const vec3 FOG_COLOR = vec3(FOG_R, FOG_G, FOG_B);

const vec3 SKY_ZENITH = vec3(SKY_ZENITH_R, SKY_ZENITH_G, SKY_ZENITH_B);
const vec3 SKY_HORIZON = vec3(SKY_HORIZON_R, SKY_HORIZON_G, SKY_HORIZON_B);
const vec3 SKY_GROUND = vec3(SKY_GROUND_R, SKY_GROUND_G, SKY_GROUND_B);
const vec3 SUN_COLOR = vec3(SKY_SUN_R, SKY_SUN_G, SKY_SUN_B);

vec3 sky(vec3 d) {
    vec3 sundir = normalize(-pc.light_dir.xyz);
    float up = clamp(d.y, 0.0, 1.0);
    float down = clamp(-d.y, 0.0, 1.0);
    vec3 col = mix(SKY_HORIZON, SKY_ZENITH, pow(up, 0.5));
    col = mix(col, SKY_GROUND, down);
    float s = max(dot(d, sundir), 0.0);
    // Soft glow only. The sharp disk belongs to sky.frag's visible background
    // alone: in reflections it would double-count the analytic sun on smooth
    // metals, and in fog it would show through the haze.
    col += SUN_COLOR * pow(s, 16.0) * SUN_GLOW;
    return col * SKY_INTENSITY;
}
// Height fog (§13): density FOG_DENSITY at FOG_HEIGHT, falling by e every
// 1/FOG_FALLOFF metres up. The optical depth along a ray from `eye` in unit
// direction `dir` over `dist` metres, integrated exactly; dist < 0 means to
// infinity, which stays finite only while the ray climbs out of the layer.
// FOG_FALLOFF = 0 gives uniform fog, FOG_DENSITY * dist.
// NOTE: must match the other shader's copy, character for character (a test
// checks), as must sky().
float fog_optical_depth(vec3 eye, vec3 dir, float dist) {
    // Specialization constants fold, `x * 0.0` can't (x may be NaN): this is
    // what makes uniform fog cost what it always did.
    if (FOG_FALLOFF == 0.0) {
        return dist < 0.0 ? 1e9 : FOG_DENSITY * dist;
    }
    float base = FOG_DENSITY * exp(-FOG_FALLOFF * (eye.y - FOG_HEIGHT));
    float kv = FOG_FALLOFF * dir.y;
    if (dist < 0.0) {
        return kv > 1e-6 ? base / kv : 1e9;
    }
    float x = kv * dist;
    if (abs(x) < 0.01) {
        // (1 - e^-x) / x by its series: the exact form cancels badly in f32.
        return base * dist * (1.0 - x * (0.5 - x / 6.0));
    }
    return base * (1.0 - exp(-max(x, -80.0))) / kv;
}
// The fog's colour seen along `dir`: its own, or the sky's, plus a glow
// towards the sun.
vec3 fog_color(vec3 dir) {
    vec3 col = FOG_SKY > 0.5 ? sky(dir) : FOG_COLOR;
    if (FOG_SUN > 0.0) {
        float s = max(dot(dir, normalize(-pc.light_dir.xyz)), 0.0);
        col += SUN_COLOR * FOG_SUN * pow(s, 8.0);
    }
    return col;
}

void main() {
    // v_uv in [0,1] -> NDC (matches the fullscreen triangle's clip position).
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec4 world = pc.inv_view_proj * vec4(ndc, 1.0, 1.0); // far plane
    vec3 dir = normalize(world.xyz / world.w - pc.camera_pos.xyz);
    float s = max(dot(dir, normalize(-pc.light_dir.xyz)), 0.0);
    vec3 col = sky(dir) + SUN_COLOR * pow(s, 4000.0) * SUN_DISK * SKY_INTENSITY;
    // The background is infinitely far, so it fogs only through a height
    // layer (FOG_FALLOFF > 0): looking up, the fog above is finite; below the
    // horizon it's total, which hides the void past the ground's edge.
    // Uniform fog leaves the sky alone, as it always did.
    if (FOG_FALLOFF > 0.0) {
        float fog = 1.0 - exp(-fog_optical_depth(pc.camera_pos.xyz, dir, -1.0));
        col = mix(col, fog_color(dir), fog);
    }
    o_color = vec4(col, 1.0);
}
