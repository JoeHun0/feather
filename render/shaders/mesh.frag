#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(push_constant) uniform Push {
    mat4 view_proj;
    vec4 light_dir;
} pc;

// std430, 64 bytes (see render::GpuMaterial). Base color factor is linear.
struct Material {
    vec4 base_color_factor;
    vec4 emissive; // rgb = emissive, a = metallic
    vec4 params;   // x = roughness, y = normal_scale, z = occlusion, w = alpha_cutoff
    uvec4 tex;     // x = base color slot into textures[]; y/z/w reserved
};
layout(set = 0, binding = 1) readonly buffer Materials {
    Material materials[];
};
// Bindless-lite: fixed-size array; slot 0 is a white default. Base-color images
// are _SRGB, so texture() returns linear. Must match render::MAX_TEXTURES.
layout(set = 0, binding = 2) uniform sampler2D textures[64];

layout(location = 0) in vec3 v_normal;
layout(location = 1) in flat uint v_material;
layout(location = 2) in vec2 v_uv;

layout(location = 0) out vec4 o_color; // linear HDR (RGBA16F target)

const vec3 AMBIENT = vec3(0.03);
const vec3 SUN_RADIANCE = vec3(3.0);

void main() {
    Material m = materials[v_material];
    // material_id varies across instances in a draw -> index is non-uniform.
    vec4 tex = texture(textures[nonuniformEXT(m.tex.x)], v_uv);
    vec3 albedo = tex.rgb * m.base_color_factor.rgb;
    vec3 L = normalize(pc.light_dir.xyz);
    float ndl = max(dot(normalize(v_normal), -L), 0.0);
    vec3 lit = albedo * (AMBIENT + SUN_RADIANCE * ndl) + m.emissive.rgb;
    o_color = vec4(lit, 1.0);
}
