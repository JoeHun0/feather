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

// Palette + soft-glow term must stay in sync with mesh.frag's sky() (the IBL
// reflection path). The extra sharp sun disk below is unique to the background.
const vec3 SKY_ZENITH = vec3(0.10, 0.22, 0.55);
const vec3 SKY_HORIZON = vec3(0.55, 0.65, 0.85);
const vec3 SKY_GROUND = vec3(0.17, 0.18, 0.19);
const vec3 SUN_COLOR = vec3(1.0, 0.95, 0.85);
const float SKY_INTENSITY = 1.0;

vec3 sky(vec3 d) {
    vec3 sundir = normalize(-pc.light_dir.xyz);
    float up = clamp(d.y, 0.0, 1.0);
    float down = clamp(-d.y, 0.0, 1.0);
    vec3 col = mix(SKY_HORIZON, SKY_ZENITH, pow(up, 0.5));
    col = mix(col, SKY_GROUND, down);
    float s = max(dot(d, sundir), 0.0);
    col += SUN_COLOR * (pow(s, 4000.0) * 60.0 + pow(s, 16.0) * 0.6); // disk + glow
    return col * SKY_INTENSITY;
}

void main() {
    // v_uv in [0,1] -> NDC (matches the fullscreen triangle's clip position).
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec4 world = pc.inv_view_proj * vec4(ndc, 1.0, 1.0); // far plane
    vec3 dir = normalize(world.xyz / world.w - pc.camera_pos.xyz);
    o_color = vec4(sky(dir), 1.0);
}
