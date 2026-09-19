#version 450
#extension GL_EXT_nonuniform_qualifier : require

layout(push_constant) uniform Push {
    mat4 view_proj;
    vec4 light_dir;   // xyz = direction the light travels
    vec4 camera_pos;  // xyz = world-space eye
} pc;

// std430, 64 bytes (see render::GpuMaterial).
struct Material {
    vec4 base_color_factor; // linear
    vec4 emissive;          // rgb = emissive, a = metallic factor
    vec4 params;            // x = roughness factor, y = normal_scale, z = occlusion, w = alpha_cutoff
    uvec4 tex;              // x = base color, y = normal, z = metallic-roughness
};
layout(set = 0, binding = 1) readonly buffer Materials {
    Material materials[];
};
// Bindless-lite: slot 0 = white, slot 1 = flat normal. Base color _SRGB;
// normal + MR are _UNORM. Must match render::MAX_TEXTURES.
layout(set = 0, binding = 2) uniform sampler2D textures[64];
// Sun shadow map (§11): comparison-sampled, 3x3 PCF below.
layout(set = 0, binding = 3) uniform sampler2DShadow u_shadow;
// Per-frame globals: the sun's light-space matrix + shadow params.
layout(set = 0, binding = 4) uniform Globals {
    mat4 light_view_proj;
    vec4 shadow_params; // x = texel size (1/dim), y = depth bias
} g;

layout(location = 0) in vec3 v_normal;
layout(location = 1) in flat uint v_material;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in vec3 v_world_pos;

layout(location = 0) out vec4 o_color; // linear HDR (RGBA16F target)

const float PI = 3.14159265359;
const vec3 SUN_RADIANCE = vec3(8.0);
const float FOG_DENSITY = 0.010; // exp distance fog; tuned for the 200-unit far plane

// Analytic procedural sky (stand-in for a precomputed IBL cubemap). Linear HDR.
// NOTE: these constants + the soft-glow term must stay in sync with sky.frag.
// The only intended difference is that sky.frag (the visible background) also
// adds a sharp sun disk, which this reflection path deliberately omits.
// SKY_GROUND tracks the level ground material tone so downward reflections match.
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
    // Soft glow only (the direct sun is a separate analytic light; a sharp disk
    // here would double-count on smooth metals). Coefficient matches sky.frag's
    // glow so a reflection shows the same haze the background sky does.
    col += SUN_COLOR * pow(s, 16.0) * 0.6;
    return col * SKY_INTENSITY;
}
// Cheap hemisphere-averaged irradiance (diffuse IBL).
vec3 sky_irradiance(vec3 n) {
    vec3 sky_avg = mix(SKY_HORIZON, SKY_ZENITH, 0.5);
    return mix(SKY_GROUND, sky_avg, n.y * 0.5 + 0.5) * SKY_INTENSITY;
}
// Karis' analytic environment BRDF (avoids a precomputed LUT).
vec2 env_brdf_approx(float rough, float ndv) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = rough * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * ndv)) * r.x + r.y;
    return vec2(-1.04, 1.04) * a004 + r.zw;
}

// Tangent frame from screen-space derivatives (no per-vertex tangent needed).
mat3 cotangent_frame(vec3 n, vec3 p, vec2 uv) {
    vec3 dp1 = dFdx(p);
    vec3 dp2 = dFdy(p);
    vec2 duv1 = dFdx(uv);
    vec2 duv2 = dFdy(uv);
    vec3 dp2perp = cross(dp2, n);
    vec3 dp1perp = cross(n, dp1);
    vec3 t = dp2perp * duv1.x + dp1perp * duv2.x;
    vec3 b = dp2perp * duv1.y + dp1perp * duv2.y;
    // Guard against zero UV derivatives (degenerate/zero UVs), which would make
    // invmax +inf and NaN the frame. The flat-normal path below avoids this case
    // entirely; this is defense-in-depth for textured meshes with bad UVs.
    float invmax = inversesqrt(max(max(dot(t, t), dot(b, b)), 1e-8));
    return mat3(t * invmax, b * invmax, n);
}

vec3 fresnel_schlick(float cos_theta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}
vec3 fresnel_schlick_roughness(float cos_theta, vec3 f0, float rough) {
    return f0 + (max(vec3(1.0 - rough), f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
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

// Sun visibility in [0,1] at a world position (3x3 PCF over the shadow map).
// The light matrix uses orthographic_rh (depth already [0,1]); clip xy -> [0,1]
// UV with no Y flip (shadow map rendered and sampled in the same convention).
float sun_shadow(vec3 world_pos) {
    vec4 lc = g.light_view_proj * vec4(world_pos, 1.0);
    vec3 proj = lc.xyz / lc.w;
    vec2 uv = proj.xy * 0.5 + 0.5;
    // Outside the frustum or past the far plane: treat as fully lit.
    if (proj.z > 1.0 || any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) {
        return 1.0;
    }
    float ref = proj.z - g.shadow_params.y; // constant depth bias
    float texel = g.shadow_params.x;
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            sum += texture(u_shadow, vec3(uv + vec2(x, y) * texel, ref));
        }
    }
    return sum / 9.0;
}

void main() {
    Material m = materials[v_material];

    vec3 albedo = texture(textures[nonuniformEXT(m.tex.x)], v_uv).rgb * m.base_color_factor.rgb;
    vec2 mr = texture(textures[nonuniformEXT(m.tex.z)], v_uv).gb; // green=rough, blue=metal
    float roughness = clamp(m.params.x * mr.x, 0.04, 1.0);
    float metallic = clamp(m.emissive.a * mr.y, 0.0, 1.0);

    vec3 ng = normalize(v_normal);
    vec3 N;
    if (m.tex.y == 1u) {
        // Slot 1 is the flat-normal default: this material has no normal map, so
        // use the geometric normal and skip the derivative TBN. Besides saving the
        // dFdx/dFdy + matrix build, this dodges the div-by-zero in cotangent_frame
        // when UV derivatives are zero (untextured meshes carry uv = 0). v_material
        // is `flat`, so this branch is uniform across the primitive.
        N = ng;
    } else {
        vec3 tn = texture(textures[nonuniformEXT(m.tex.y)], v_uv).xyz * 2.0 - 1.0;
        tn.xy *= m.params.y; // normal_scale
        N = normalize(cotangent_frame(ng, v_world_pos, v_uv) * tn);
    }

    vec3 V = normalize(pc.camera_pos.xyz - v_world_pos);
    vec3 L = normalize(-pc.light_dir.xyz); // toward the light
    vec3 H = normalize(V + L);

    float ndl = max(dot(N, L), 0.0);
    float ndv = max(dot(N, V), 0.0);
    float ndh = max(dot(N, H), 0.0);
    float hdv = max(dot(H, V), 0.0);

    vec3 f0 = mix(vec3(0.04), albedo, metallic);

    // --- Direct light (Cook-Torrance) ---
    float ndf = distribution_ggx(ndh, roughness);
    float g = geometry_smith(ndv, ndl, roughness);
    vec3 f = fresnel_schlick(hdv, f0);
    vec3 specular = (ndf * g * f) / (4.0 * ndv * ndl + 0.0001);
    vec3 kd = (vec3(1.0) - f) * (1.0 - metallic);
    // Sun shadow occludes the direct term only; ambient/IBL stays unshadowed.
    float shadow = sun_shadow(v_world_pos);
    vec3 lo = (kd * albedo / PI + specular) * SUN_RADIANCE * ndl * shadow;

    // --- Ambient (analytic IBL: split-sum against the procedural sky) ---
    vec3 fr = fresnel_schlick_roughness(ndv, f0, roughness);
    vec3 kd_amb = (vec3(1.0) - fr) * (1.0 - metallic);
    vec3 diffuse_ibl = sky_irradiance(N) * albedo;
    vec3 r = reflect(-V, N);
    vec3 prefiltered = mix(sky(r), sky_irradiance(r), roughness); // crude roughness blur
    vec2 ab = env_brdf_approx(roughness, ndv);
    vec3 specular_ibl = prefiltered * (f0 * ab.x + ab.y);
    vec3 ambient = kd_amb * diffuse_ibl + specular_ibl;

    vec3 color = ambient + lo + m.emissive.rgb;

    // Distance fog: blend toward the sky along the view ray. Hides the finite
    // ground edge and reads as depth. sky() here has no sun disk, so no searing
    // dot bleeds into the haze.
    float dist = length(pc.camera_pos.xyz - v_world_pos);
    float fog = 1.0 - exp(-dist * FOG_DENSITY);
    color = mix(color, sky(normalize(v_world_pos - pc.camera_pos.xyz)), fog);

    o_color = vec4(color, 1.0);
}
