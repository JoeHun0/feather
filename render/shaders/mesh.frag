#version 450

layout(push_constant) uniform Push {
    mat4 view_proj;
    vec4 light_dir;
} pc;

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_color; // linear albedo

layout(location = 0) out vec4 o_color; // linear HDR (RGBA16F target)

// Provisional linear-space Lambert. Not PBR yet, but correct color management:
// linear albedo * (ambient + HDR sun * NdotL), written unclamped to the HDR
// target for the tonemap pass to resolve. Sun intensity > 1 exercises the HDR
// range so tonemapping has something to compress.
const vec3 AMBIENT = vec3(0.03);
const vec3 SUN_RADIANCE = vec3(3.0);

void main() {
    vec3 L = normalize(pc.light_dir.xyz);
    float ndl = max(dot(normalize(v_normal), -L), 0.0);
    vec3 lit = v_color * (AMBIENT + SUN_RADIANCE * ndl);
    o_color = vec4(lit, 1.0);
}
