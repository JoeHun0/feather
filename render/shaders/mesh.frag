#version 450

layout(push_constant) uniform Push {
    mat4 view_proj;
    vec4 light_dir;
} pc;

// std430, 64 bytes (see render::GpuMaterial). Base color is linear.
struct Material {
    vec4 base_color_factor;
    vec4 emissive; // rgb = emissive, a = metallic
    vec4 params;   // x = roughness, y = normal_scale, z = occlusion, w = alpha_cutoff
    uvec4 tex;     // bindless indices (unused until textures)
};
layout(set = 0, binding = 1) readonly buffer Materials {
    Material materials[];
};

layout(location = 0) in vec3 v_normal;
layout(location = 1) in flat uint v_material;

layout(location = 0) out vec4 o_color; // linear HDR (RGBA16F target)

// Provisional linear-space Lambert on the material's base color. Not PBR yet
// (metallic/roughness stored but unused); written unclamped to the HDR target.
const vec3 AMBIENT = vec3(0.03);
const vec3 SUN_RADIANCE = vec3(3.0);

void main() {
    Material m = materials[v_material];
    vec3 albedo = m.base_color_factor.rgb;
    vec3 L = normalize(pc.light_dir.xyz);
    float ndl = max(dot(normalize(v_normal), -L), 0.0);
    vec3 lit = albedo * (AMBIENT + SUN_RADIANCE * ndl) + m.emissive.rgb;
    o_color = vec4(lit, 1.0);
}
