#version 450
#extension GL_EXT_nonuniform_qualifier : require

// Depth-only passes for masked materials (§5): the depth prepass and the sun
// shadow map. Cuts out fragments below the material's alpha cutoff; writes no
// colour. Paired with mesh.vert, whose outputs it reads.

// std430, 64 bytes (see render::GpuMaterial); the same as mesh.frag's.
struct Material {
    vec4 base_color_factor; // linear
    vec4 emissive;          // rgb = emissive, a = metallic factor
    vec4 params;            // x = roughness factor, y = normal_scale, z = occlusion, w = alpha_cutoff
    uvec4 tex;              // x = base color, y = normal, z = metallic-roughness, w = flags
};
layout(set = 0, binding = 1) readonly buffer Materials {
    Material materials[];
};
layout(set = 0, binding = 2) uniform sampler2D textures[];

layout(location = 1) in flat uint v_material;
layout(location = 2) in vec2 v_uv;

// Cutout (§5): the base colour's alpha for material `m` at `uv`, raised
// with the mip level, since box-filtered mips average alpha down and a cutout
// would otherwise thin away with distance.
// NOTE: must match the other shader's copy, character for character (a test
// checks).
float cutout_alpha(Material m, vec2 uv) {
    float a = texture(textures[nonuniformEXT(m.tex.x)], uv).a * m.base_color_factor.a;
    float lod = textureQueryLod(textures[nonuniformEXT(m.tex.x)], uv).x;
    return a * (1.0 + max(lod, 0.0) * 0.25);
}

void main() {
    Material m = materials[v_material];
    // Nothing after this needs derivatives, so a plain discard is enough.
    if (cutout_alpha(m, v_uv) < m.params.w) {
        discard;
    }
}
