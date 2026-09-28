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
// (a test checks). sky() must stay in step with mesh.frag's (the IBL reflection
// path) by hand; the extra sharp sun disk is unique to the background.
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
    col += SUN_COLOR * (pow(s, 4000.0) * SUN_DISK + pow(s, 16.0) * SUN_GLOW); // disk + glow
    return col * SKY_INTENSITY;
}

void main() {
    // v_uv in [0,1] -> NDC (matches the fullscreen triangle's clip position).
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec4 world = pc.inv_view_proj * vec4(ndc, 1.0, 1.0); // far plane
    vec3 dir = normalize(world.xyz / world.w - pc.camera_pos.xyz);
    o_color = vec4(sky(dir), 1.0);
}
