#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(push_constant) uniform Push {
    mat4 view_proj;
    vec4 light_dir;   // xyz = direction the light travels
    vec4 camera_pos;  // xyz = world-space eye
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
// Bindless-lite: slot 0 is white. Base-color images are _SRGB -> linear on read.
layout(set = 0, binding = 2) uniform sampler2D textures[64];

layout(location = 0) in vec3 v_normal;
layout(location = 1) in flat uint v_material;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in vec3 v_world_pos;

layout(location = 0) out vec4 o_color; // linear HDR (RGBA16F target)

const float PI = 3.14159265359;
// Single directional light (HDR radiance) + flat ambient stand-in for IBL.
const vec3 SUN_RADIANCE = vec3(8.0);
const vec3 AMBIENT = vec3(0.03);

// Cook-Torrance, metallic-roughness workflow.
vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}
float distribution_ggx(float ndh, float rough) {
    float a = rough * rough;
    float a2 = a * a;
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}
float geometry_smith(float ndv, float ndl, float rough) {
    float r = rough + 1.0;
    float k = (r * r) / 8.0;
    float gv = ndv / (ndv * (1.0 - k) + k);
    float gl = ndl / (ndl * (1.0 - k) + k);
    return gv * gl;
}

void main() {
    Material m = materials[v_material];
    vec3 albedo = texture(textures[nonuniformEXT(m.tex.x)], v_uv).rgb * m.base_color_factor.rgb;
    float metallic = clamp(m.emissive.a, 0.0, 1.0);
    float roughness = clamp(m.params.x, 0.04, 1.0);

    vec3 N = normalize(v_normal);
    vec3 V = normalize(pc.camera_pos.xyz - v_world_pos);
    vec3 L = normalize(-pc.light_dir.xyz); // toward the light
    vec3 H = normalize(V + L);

    float ndl = max(dot(N, L), 0.0);
    float ndv = max(dot(N, V), 0.0);
    float ndh = max(dot(N, H), 0.0);
    float hdv = max(dot(H, V), 0.0);

    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    float ndf = distribution_ggx(ndh, roughness);
    float g = geometry_smith(ndv, ndl, roughness);
    vec3 f = fresnel_schlick(hdv, f0);

    vec3 specular = (ndf * g * f) / (4.0 * ndv * ndl + 0.0001);
    vec3 kd = (vec3(1.0) - f) * (1.0 - metallic);

    vec3 lo = (kd * albedo / PI + specular) * SUN_RADIANCE * ndl;
    vec3 ambient = AMBIENT * albedo; // IBL later
    vec3 color = ambient + lo + m.emissive.rgb;
    o_color = vec4(color, 1.0);
}
