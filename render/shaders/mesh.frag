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
    uvec4 tex;              // x = base color, y = normal, z = metallic-roughness, w = flags
};
layout(set = 0, binding = 1) readonly buffer Materials {
    Material materials[];
};
// Bindless: slot 0 = white, slot 1 = flat normal, then one slot per unique image
// the level uses. Runtime-sized: the renderer sizes the binding per level,
// limited only by the device. Base color _SRGB; normal + MR are _UNORM.
layout(set = 0, binding = 2) uniform sampler2D textures[];

// Material flags (tex.w). Must match render::mesh's MATERIAL_DOUBLE_SIDED.
const uint MATERIAL_DOUBLE_SIDED = 1u;
// tex.w's bits above the flags: the occlusion texture's slot (render::mesh's
// MATERIAL_OCCLUSION_SHIFT; a test checks).
const uint MATERIAL_OCCLUSION_SHIFT = 8u;
// Masked (cutout) materials run this same shader: the depth prepass cut
// them (mask.frag), and their pipeline tests depth for EQUAL, so a cut texel
// never gets here (§5).
// Cascades in the sun shadow map. MUST match feather_gfx::SHADOW_CASCADES.
const int SHADOW_CASCADES = 4;
// How far to push the sample along the surface normal, in shadow texels (§11's
// normal-offset bias). Scaled per cascade by its world-texel size, so it stays
// one texel wide whatever that cascade covers.
const float NORMAL_OFFSET_TEXELS = 1.5;
// Where the fade into the next cascade begins, as a fraction toward the edge of
// the current cascade's box.
const float BLEND_START = 0.85;

// Cascaded sun shadow map (§11): one array layer per cascade, comparison-
// sampled, 3x3 PCF below.
layout(set = 0, binding = 3) uniform sampler2DArrayShadow u_shadow;
// Per-frame globals: a light-space matrix per cascade + shadow params.
layout(set = 0, binding = 4) uniform Globals {
    mat4 light_view_proj[SHADOW_CASCADES];
    vec4 texel_world;   // world units per shadow texel, per cascade
    vec4 light_params;  // x = live light count
    vec4 shadow_params; // x = texel size (1/dim), y = depth bias
    mat4 view;           // world -> view, for the cluster lookup
    vec4 cluster_params; // x = near, y = far, z = CLUSTER_Z / ln(far/near)
    vec4 cluster_proj;   // x = tan(fov_x/2), y = tan(fov_y/2)
    vec4 sky_origin;     // sky volume: xyz = minimum corner, w = 1 / cell
    vec4 sky_dims;       // xyz = cells per axis, w = 1 if there is a volume
    vec4 ao_params;      // x = 1 if GTAO ran this frame, yz = the full-resolution size
    // The atmosphere (§13), per frame so the weather can change it:
    // render::Atmosphere. mesh.frag and sky.frag declare this block text for
    // text (a test checks).
    vec4 sun_radiance;   // rgb = the directional light's colour × intensity
    vec4 sun_dir;        // xyz = towards the sun, whose glow the sky shows
    vec4 sky_zenith;     // rgb, w = the sky's intensity
    vec4 sky_horizon;    // rgb, w = the sun's glow
    vec4 sky_ground;     // rgb, w = the sun's disk
    vec4 sky_sun;        // rgb = the glow's and disk's tint, w = the fog's sun glow
    vec4 fog_color;      // rgb, w = 1 to take the sky's colour instead
    vec4 fog;            // x = density at y = height, z = falloff
} g;

// The atmosphere's parts, by the names sky() and the fog functions use; both
// shaders define the same (a test checks).
#define SUN_RADIANCE g.sun_radiance.rgb
#define SUN_DIR g.sun_dir.xyz
#define SKY_ZENITH g.sky_zenith.rgb
#define SKY_INTENSITY g.sky_zenith.w
#define SKY_HORIZON g.sky_horizon.rgb
#define SUN_GLOW g.sky_horizon.w
#define SKY_GROUND g.sky_ground.rgb
#define SUN_DISK g.sky_ground.w
#define SUN_COLOR g.sky_sun.rgb
#define FOG_SUN g.sky_sun.w
#define FOG_COLOR g.fog_color.rgb
#define FOG_SKY g.fog_color.w
#define FOG_DENSITY g.fog.x
#define FOG_HEIGHT g.fog.y
#define FOG_FALLOFF g.fog.z

// Punctual lights (§12), shaded only if this fragment's cluster lists them.
struct Light {
    vec4 pos_radius;      // xyz = world position, w = radius
    vec4 radiance_source; // rgb = colour × intensity (linear), a = source radius
};
layout(set = 0, binding = 5) readonly buffer Lights {
    Light lights[];
};

// Light-cluster grid, written by cluster.comp. MUST match it and render::mesh.
const uint CLUSTER_X = 16;
const uint CLUSTER_Y = 9;
const uint CLUSTER_Z = 24;
const uint MAX_LIGHTS = 128;
const uint CLUSTER_WORDS = MAX_LIGHTS / 32;
// One bitmask per cluster over this frame's light list.
layout(set = 0, binding = 6) readonly buffer Clusters {
    uint cluster_masks[];
};

// The level's sky visibility (§13), baked: per cell an RG32_UINT texel, x the
// SkyVis moments as RGBA8 and y the SkyFree nibbles. Read with texelFetch and
// blended by hand in sky_visibility().
layout(set = 0, binding = 7) uniform usampler3D u_sky;
// GTAO's result (§13): the ambient's small-scale visibility at half
// resolution, and its depth levels, whose level 1 (mip 0 here) holds each
// half-resolution texel's depth. gtao_upsample() brings it to this pixel.
layout(set = 0, binding = 8) uniform sampler2D u_ao;
layout(set = 0, binding = 9) uniform sampler2D u_ao_depth;

layout(location = 0) in vec3 v_normal;
layout(location = 1) in flat uint v_material;
layout(location = 2) in vec2 v_uv;
layout(location = 3) in vec3 v_world_pos;

layout(location = 0) out vec4 o_color; // linear HDR (RGBA16F target)

const float PI = 3.14159265359;
// Analytic procedural sky (stand-in for a precomputed IBL cubemap). Linear HDR.
// NOTE: sky() and the fog functions below are copies of sky.frag's, text for
// text (a test checks). Only sky.frag's background adds the sharp sun disk.
vec3 sky(vec3 d) {
    // The sun's glow, even with the moon as the light (§13's weather): it
    // lingers on the horizon after sunset.
    vec3 sundir = SUN_DIR;
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
    // Per frame, not per pixel: the branch is the same across a draw, and
    // it's what keeps uniform fog as cheap as it was.
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
        float s = max(dot(dir, SUN_DIR), 0.0);
        col += SUN_COLOR * FOG_SUN * pow(s, 8.0);
    }
    return col;
}
// Cheap hemisphere-averaged irradiance (diffuse IBL).
vec3 sky_irradiance(vec3 n) {
    vec3 sky_avg = mix(SKY_HORIZON, SKY_ZENITH, 0.5);
    return mix(SKY_GROUND, sky_avg, n.y * 0.5 + 0.5) * SKY_INTENSITY;
}
// --- Sky visibility (§13) ---
// feather_assets::bake's constants (a test checks): the SkyVis encoding, the
// fade past a SkyFree distance (cells) and SkyVis::specular's fade (in r.y).
const float SKY_W0_SCALE = 510.0;
const float SKY_W_BIAS = 127.0;
const float SKY_W_SCALE = 508.0;
const float SKY_FREE_SOFT = 0.1;
const float SKY_SPECULAR_FADE = 0.25;
// (w0, w.xyz) of open sky.
const vec4 SKY_OPEN = vec4(0.5, 0.0, 0.5, 0.0);

// How much sky world point `p` sees, as (w0, w): SkyVolume::sample, step for
// step. A trilinear blend of the eight cells round `p`, leaving out any whose
// view towards `p` is blocked (so no cell above a thin roof lights the room
// under it), renormalised; if all are blocked, the plain blend. Outside the
// grid it fades to open sky over one cell.
vec4 sky_visibility(vec3 p) {
    if (g.sky_dims.w == 0.0) {
        return SKY_OPEN;
    }
    vec3 dims = g.sky_dims.xyz;
    vec3 u = (p - g.sky_origin.xyz) * g.sky_origin.w;
    vec3 out3 = max(-u, u - dims);
    float outside = clamp(max(max(out3.x, out3.y), out3.z), 0.0, 1.0);
    vec3 t = clamp(u - 0.5, vec3(0.0), dims - 1.0);
    vec3 i = floor(t);
    vec3 f = t - i;
    vec4 seen = vec4(0.0);
    float seen_w = 0.0;
    vec4 plain = vec4(0.0);
    for (int c = 0; c < 8; ++c) {
        vec3 o = vec3(float(c & 1), float((c >> 1) & 1), float((c >> 2) & 1));
        uvec2 texel = texelFetch(u_sky, ivec3(min(i + o, dims - 1.0)), 0).xy;
        vec3 tri3 = mix(1.0 - f, f, o);
        float tri = tri3.x * tri3.y * tri3.z;
        vec4 b = unpackUnorm4x8(texel.x) * 255.0;
        vec4 v = vec4(b.x / SKY_W0_SCALE, (b.y - SKY_W_BIAS) / SKY_W_SCALE,
                      b.z / SKY_W0_SCALE, (b.w - SKY_W_BIAS) / SKY_W_SCALE);
        // SkyFree::visible: how far this cell sees towards p, per axis.
        uint fr = texel.y;
        vec3 plus = vec3(fr & 15u, (fr >> 8) & 15u, (fr >> 16) & 15u) / 15.0;
        vec3 minus = vec3((fr >> 4) & 15u, (fr >> 12) & 15u, (fr >> 20) & 15u) / 15.0;
        vec3 delta = f - o;
        vec3 over3 = abs(delta) - mix(minus, plus, greaterThanEqual(delta, vec3(0.0)));
        float over = max(max(over3.x, over3.y), over3.z);
        float w = tri * clamp(1.0 - over / SKY_FREE_SOFT, 0.0, 1.0);
        seen += w * v;
        seen_w += w;
        plain += tri * v;
    }
    vec4 vis = seen_w > 1e-3 ? seen / seen_w : plain;
    return mix(vis, SKY_OPEN, outside);
}
// The sky's weight for a surface facing `n`: SkyVis::weight.
float sky_weight(vec4 sv, vec3 n) {
    return max(sv.x + dot(sv.yzw, n), 0.0);
}
// sky_irradiance(n) with the sky occluded by `sv`: the sky part by its
// directional weight, the ground's bounce by the fraction of sky seen
// (SkyVis::seen). Open sky gives sky_irradiance(n) exactly.
vec3 sky_irradiance_occluded(vec3 n, vec4 sv) {
    vec3 sky_avg = mix(SKY_HORIZON, SKY_ZENITH, 0.5);
    return (SKY_GROUND * (0.5 - 0.5 * n.y) * (2.0 * sv.x) + sky_avg * sky_weight(sv, n))
        * SKY_INTENSITY;
}
// How much of the sky's reflection along `r` survives: SkyVis::specular.
float sky_specular(vec4 sv, vec3 r) {
    float ratio = clamp(sky_weight(sv, r) / max(0.5 + 0.5 * r.y, 0.5), 0.0, 1.0);
    return mix(2.0 * sv.x, ratio, smoothstep(0.0, SKY_SPECULAR_FADE, r.y));
}

// --- GTAO (§13) ---
// Jimenez et al. 2016: visibility `v` with the light that bounces between
// occluders of this albedo put back, so bright corners don't go grey. Its
// coefficients are render::ao's AO_MULTIBOUNCE (a test checks).
vec3 ao_multibounce(float v, vec3 albedo) {
    vec3 a = 2.0404 * albedo - 0.3324;
    vec3 b = -4.7951 * albedo + 0.6417;
    vec3 c = 2.7552 * albedo + 0.6903;
    return max(vec3(v), ((v * a + b) * v + c) * v);
}
// glTF's occlusionTexture: `strength` 0 ignores the sample, 1 applies it.
// render::ao's reference (a test checks the text).
float material_occlusion(float sampled, float strength) {
    return 1.0 + strength * (sampled - 1.0);
}
// The material's occlusion and GTAO's, as one visibility: the lower.
float combine_occlusion(float material, float gtao) {
    return min(material, gtao);
}
// Lagarde and de Rousiers 2014: the specular ambient that survives
// visibility `ao` at this view angle and roughness.
float ao_specular(float ndv, float ao, float roughness) {
    return clamp(pow(ndv + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao, 0.0, 1.0);
}

// GTAO's result for this fragment (§13): gtao_denoise.comp's half-resolution
// visibility, upsampled. Texel q stands for pixel 2q. An even pixel copies
// its texel; any other blends the 2 or 4 round it bilinearly, each weighted
// by its distance off this fragment's tangent plane (its own position and
// normal) and not at all past AO_DENOISE_DEPTH of its distance, so occlusion
// doesn't cross a silhouette. If none lies on the plane (a feature too thin
// to have a texel), the one nearest in distance. Positions are in the GTAO
// passes' space: x right, y down, z the distance ahead.
// render::ao's reference (`upsample`) is this, step for step.
const float AO_DENOISE_DEPTH = 0.1;
float gtao_upsample(vec3 world_pos, vec3 n_world) {
    vec3 v = (g.view * vec4(world_pos, 1.0)).xyz;
    vec3 P = vec3(v.x, -v.y, -v.z);
    vec3 nv = mat3(g.view) * n_world;
    vec3 N = vec3(nv.x, -nv.y, -nv.z);
    float tolerance = AO_DENOISE_DEPTH * P.z;
    // A depth's distance, as the passes find it: near·r / (depth + r), with
    // r = far / (near - far).
    float r = g.cluster_params.y / (g.cluster_params.x - g.cluster_params.y);
    ivec2 half_size = textureSize(u_ao, 0);
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 q0 = p >> 1;
    // How far p lies towards the next texel: 0 or 1/2 on each axis.
    vec2 f = vec2(p & 1) * 0.5;
    float sum = 0.0;
    float total = 0.0;
    float nearest = 1.0;
    float nearest_dz = 1e30;
    for (int y = 0; y <= 1; ++y) {
        for (int x = 0; x <= 1; ++x) {
            float bilinear = (x == 0 ? 1.0 - f.x : f.x) * (y == 0 ? 1.0 - f.y : f.y);
            if (bilinear == 0.0) {
                continue;
            }
            ivec2 q = min(q0 + ivec2(x, y), half_size - 1);
            float dist = g.cluster_params.x * r / (texelFetch(u_ao_depth, q, 0).r + r);
            vec2 ndc = (vec2(2 * q) + 0.5) / g.ao_params.yz * 2.0 - 1.0;
            vec3 Q = vec3(ndc * g.cluster_proj.xy * dist, dist);
            float ao = texelFetch(u_ao, q, 0).r;
            float w = bilinear * max(1.0 - abs(dot(Q - P, N)) / tolerance, 0.0);
            sum += w * ao;
            total += w;
            float dz = abs(Q.z - P.z);
            if (dz < nearest_dz) {
                nearest_dz = dz;
                nearest = ao;
            }
        }
    }
    return total > 1e-4 ? sum / total : nearest;
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
// GGX D in terms of alpha (= roughness²), so a caller can widen alpha itself.
float distribution_ggx_alpha(float ndh, float a) {
    float a2 = a * a;
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}
float distribution_ggx(float ndh, float rough) {
    return distribution_ggx_alpha(ndh, rough * rough);
}
float geometry_smith(float ndv, float ndl, float rough) {
    float r = rough + 1.0;
    float k = (r * r) / 8.0;
    float gv = ndv / (ndv * (1.0 - k) + k);
    float gl = ndl / (ndl * (1.0 - k) + k);
    return gv * gl;
}

// 3x3 PCF in one cascade's layer. `uvz` is (uv, biased reference depth).
float pcf(vec3 uvz, int cascade) {
    float texel = g.shadow_params.x;
    float sum = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            sum += texture(u_shadow, vec4(uvz.xy + vec2(x, y) * texel, float(cascade), uvz.z));
        }
    }
    return sum / 9.0;
}

// Project a world position into cascade `i`, applying the normal-offset bias.
// False when it falls outside that cascade's box.
//
// The light matrix uses orthographic_rh (depth already [0,1]); clip xy -> [0,1]
// UV with no Y flip (shadow map rendered and sampled in the same convention).
bool project_cascade(vec3 world_pos, vec3 ng, int i, out vec3 uvz) {
    // Normal-offset bias: stepping along the surface, rather than only along
    // depth, is what lets the constant bias stay small enough that contact
    // shadows do not detach (§11).
    vec3 p = world_pos + ng * (g.texel_world[i] * NORMAL_OFFSET_TEXELS);
    vec4 lc = g.light_view_proj[i] * vec4(p, 1.0);
    vec3 proj = lc.xyz / lc.w;
    vec2 uv = proj.xy * 0.5 + 0.5;
    uvz = vec3(uv, proj.z - g.shadow_params.y);
    return proj.z <= 1.0
        && all(greaterThanEqual(uv, vec2(0.0)))
        && all(lessThanEqual(uv, vec2(1.0)));
}

// Sun visibility in [0,1], from the tightest cascade that contains this point.
//
// Selection is by **projection containment**, not by view-space depth against
// split distances: the fragment shader has no view matrix, and the push constant
// is already 96 B (adding one would pass the 128 B floor Vulkan guarantees).
// Testing containment needs neither, and is correct by construction — cascade 0
// is the tightest, so the first box that contains the point is the best one.
float sun_shadow(vec3 world_pos, vec3 ng) {
    for (int i = 0; i < SHADOW_CASCADES; ++i) {
        vec3 uvz;
        if (!project_cascade(world_pos, ng, i, uvz)) {
            continue;
        }
        float s = pcf(uvz, i);
        // Near this cascade's edge, fade into the next one so the change in
        // resolution reads as a gradient instead of a line across the ground.
        float edge = max(abs(uvz.x * 2.0 - 1.0), abs(uvz.y * 2.0 - 1.0));
        if (i + 1 < SHADOW_CASCADES && edge > BLEND_START) {
            vec3 next;
            if (project_cascade(world_pos, ng, i + 1, next)) {
                s = mix(s, pcf(next, i + 1), smoothstep(BLEND_START, 1.0, edge));
            }
        }
        return s;
    }
    // Past the last cascade: unshadowed. Distance fog covers the transition.
    return 1.0;
}

// Windowed inverse-square falloff. The window drives the light to *exactly*
// zero at its radius; without it the cutoff lands mid-gradient and reads as a
// visible sphere edge across the ground. Mirrors `light_attenuation` in the app,
// which unit-tests the properties.
float attenuation(float dist, float radius) {
    if (radius <= 0.0 || dist >= radius) {
        return 0.0;
    }
    float t = dist / radius;
    float window = clamp(1.0 - t * t * t * t, 0.0, 1.0);
    // +1 keeps it finite at d = 0 rather than exploding.
    return window * window / (dist * dist + 1.0);
}

// Which cluster a world-space point falls in. Tiles are defined by view-space
// slope (x/d, y/d) rather than gl_FragCoord, which is exactly how cluster.comp
// bounds them — the two share one definition, and neither needs the
// framebuffer size.
uint cluster_index(vec3 world_pos) {
    vec3 p = (g.view * vec4(world_pos, 1.0)).xyz;
    float d = max(-p.z, g.cluster_params.x);
    vec2 ndc = vec2(p.x, p.y) / (d * g.cluster_proj.xy); // [-1, 1] on screen
    uint tx = uint(clamp(floor((ndc.x * 0.5 + 0.5) * float(CLUSTER_X)), 0.0, float(CLUSTER_X - 1)));
    // Row 0 is the top of the screen, i.e. +y in view space.
    uint ty = uint(clamp(floor((0.5 - ndc.y * 0.5) * float(CLUSTER_Y)), 0.0, float(CLUSTER_Y - 1)));
    float slice = floor(log(d / g.cluster_params.x) * g.cluster_params.z);
    uint tz = uint(clamp(slice, 0.0, float(CLUSTER_Z - 1)));
    return tx + ty * CLUSTER_X + tz * CLUSTER_X * CLUSTER_Y;
}

// Cook-Torrance for one light, reusing the same BRDF terms as the sun rather
// than a second lighting path.
//
// The light is a **sphere** of `radiance_source.a` (Karis 2013, representative
// point). An infinitesimal point makes a near-singular highlight on smooth
// metal: at roughness 0.04 the GGX lobe is so narrow that the reflection
// becomes a tiny, sun-bright dot. So specular is lit from the point on the
// sphere closest to the reflection ray. That makes D flat-topped across a disc
// the size of the source, so D is scaled by (α/α')², the widened lobe's
// normalisation over the original's, with α' = α + src / 2d (the source's
// angular radius, halved for half-vector space). It is applied to D at the
// *original* α; also evaluating D at α' widens twice and loses ~99% of the
// energy. Measured on the CPU reference: within ±12% of the point light's
// energy at roughness 0.3 and +15–49% on mirror-smooth metal, the known
// looseness of the approximation. Diffuse keeps the centre direction.
//
// Falloff stays on the *centre* distance. Measured from the representative
// point, a light could reach just past its radius, and §12's clusters rely on
// it being exactly zero there. With src = 0 every step reduces to the plain
// point light (the same maths; only the float operation order differs).
//
// `R` is the view reflected about N: light-independent, so the caller computes
// it once per fragment rather than once per light.
vec3 punctual(Light l, vec3 N, vec3 V, vec3 R, vec3 world_pos, vec3 albedo, vec3 f0,
              float roughness, float metallic) {
    vec3 delta = l.pos_radius.xyz - world_pos;
    float dist = length(delta);
    float att = attenuation(dist, l.pos_radius.w);
    if (att <= 0.0) {
        return vec3(0.0);
    }
    float src = l.radiance_source.a;
    // The whole sphere is below this surface's horizon: nothing to light, and
    // this is cheap enough to test before the representative point. With
    // src = 0 it is the old point-light N·L <= 0 test.
    if (dot(N, delta) <= -src) {
        return vec3(0.0);
    }
    vec3 L = delta / max(dist, 1e-4);
    float ndl = max(dot(N, L), 0.0);

    // Representative point: the closest point on the sphere to the reflection ray.
    vec3 center_to_ray = dot(delta, R) * R - delta;
    vec3 closest = delta + center_to_ray * clamp(src / max(length(center_to_ray), 1e-6), 0.0, 1.0);
    vec3 Ls = normalize(closest);
    float ndl_s = max(dot(N, Ls), 0.0);

    vec3 H = normalize(V + Ls);
    float ndv = max(dot(N, V), 1e-4);
    float ndh = max(dot(N, H), 0.0);
    float hdv = max(dot(H, V), 0.0);

    float a = roughness * roughness;
    float a_wide = clamp(a + src / (2.0 * max(dist, 1e-4)), 0.0, 1.0);
    float norm = (a / a_wide) * (a / a_wide);
    float ndf = distribution_ggx_alpha(ndh, a) * norm;
    float gs = geometry_smith(ndv, ndl_s, roughness);
    vec3 f = fresnel_schlick(hdv, f0);
    vec3 spec = (ndf * gs * f) / (4.0 * ndv * ndl_s + 0.0001);
    vec3 kd = (vec3(1.0) - f) * (1.0 - metallic);
    vec3 radiance = l.radiance_source.rgb * att;
    return (kd * albedo / PI * ndl + spec * ndl_s) * radiance;
}

#ifdef WATER
// ---- Water: the transparent pass (§10) ----
//
// Drawn over the opaque scene, which it refracts itself rather than blending
// over, so the pipeline has no blending. Set 1 holds that scene's colour and
// depth, single-sample, and only the transparent pass binds it: the
// standard variant of this file never names it.
layout(set = 1, binding = 0) uniform sampler2D u_scene;
layout(set = 1, binding = 1) uniform sampler2D u_scene_depth;

// Ripples: small waves crossing a sheltered pond, as a direction and
// (wavelength m, amplitude m, speed m/s). Their slopes add up to ~7°.
const vec2 WAVE_DIR[4] = vec2[](
    vec2(0.80, 0.60), vec2(-0.60, 0.80), vec2(0.20, -0.98), vec2(-0.90, -0.43));
const vec3 WAVE[4] = vec3[](
    vec3(1.30, 0.0080, 0.55), vec3(0.75, 0.0040, 0.40),
    vec3(0.45, 0.0022, 0.30), vec3(0.27, 0.0012, 0.22));
// How far the ripples shift the view of the bed, in metres per unit of slope,
// at a metre of water or more; less in the shallows, none at the shore.
const float REFRACTION = 0.6;
// Water's reflectance head-on (an index of 1.33).
const float WATER_F0 = 0.02;

// View-space distance of a depth-buffer value (the projection's near and
// far, [0, 1] depth).
float linear_depth(float d) {
    float n = g.cluster_params.x;
    float f = g.cluster_params.y;
    return n * f / (f - d * (f - n));
}

// The rippled surface's normal at world `p` (xz) and time `t`: the gradient
// of the sum of the waves' heights.
vec3 water_normal(vec2 p, float t) {
    vec2 grad = vec2(0.0);
    for (int i = 0; i < 4; ++i) {
        float k = 2.0 * PI / WAVE[i].x;
        float phase = k * (dot(WAVE_DIR[i], p) - WAVE[i].z * t);
        grad += WAVE_DIR[i] * (WAVE[i].y * k * cos(phase));
    }
    return normalize(vec3(-grad.x, 1.0, -grad.y));
}

void main() {
    Material m = materials[v_material];
    // What the water itself looks like where it's deep, and how far light
    // gets through it before 1/e is left (`params.w`, render::GpuMaterial).
    vec3 deep = m.base_color_factor.rgb;
    float clarity = max(m.params.w, 1e-3);
    float roughness = clamp(m.params.x, 0.02, 1.0);
    const vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 N = water_normal(v_world_pos.xz, g.light_params.y);

    vec3 to_eye = pc.camera_pos.xyz - v_world_pos;
    float dist = length(to_eye);
    vec3 V = to_eye / max(dist, 1e-6);

    // How deep the water is behind this pixel: the opaque scene's depth less
    // the surface's, in view space.
    vec2 size = vec2(textureSize(u_scene, 0));
    vec2 uv = gl_FragCoord.xy / size;
    float z_surface = linear_depth(gl_FragCoord.z);
    float z_bed = linear_depth(texelFetch(u_scene_depth, ivec2(gl_FragCoord.xy), 0).r);
    float depth = max(z_bed - z_surface, 0.0);
    // The bed seen through the ripples: shifted on screen by the slope, as
    // much as the water is deep. Where the shifted look lands on something
    // in front of the water (a bank, a post), the shift is dropped.
    vec2 shift = N.xz * REFRACTION * min(depth, 1.0) / (2.0 * z_surface * g.cluster_proj.xy);
    vec2 ruv = uv + shift;
    float z_r = linear_depth(textureLod(u_scene_depth, ruv, 0.0).r);
    if (z_r <= z_surface) {
        ruv = uv;
        z_r = z_bed;
    }
    vec3 bed = textureLod(u_scene, ruv, 0.0).rgb;
    // The light's path through the water to the bed and back up this view
    // ray: view-space depth scales to distance along it.
    float path = max(z_r - z_surface, 0.0) * dist / max(z_surface, 1e-4);
    float transmit = exp(-path / clarity);

    // Light in the water itself: the sky it sees and the sun, on its colour.
    vec3 L = normalize(-pc.light_dir.xyz);
    vec4 sv = sky_visibility(v_world_pos + up / g.sky_origin.w);
    float shadow = sun_shadow(v_world_pos, up);
    vec3 lit = sky_irradiance_occluded(up, sv) + SUN_RADIANCE * shadow * max(L.y, 0.0);
    vec3 below = mix(deep * lit / PI, bed, transmit);

    // The surface: Fresnel between what's below and the sky it mirrors, then
    // the sun's and the lamps' highlights.
    float ndv = max(dot(N, V), 1e-4);
    vec3 R = reflect(-V, N);
    float fresnel = WATER_F0 + (1.0 - WATER_F0) * pow(1.0 - ndv, 5.0);
    vec3 color = mix(below, sky(R) * sky_specular(sv, R), fresnel);
    vec3 f0 = vec3(WATER_F0);
    vec3 H = normalize(V + L);
    float ndl = max(dot(N, L), 0.0);
    float ndh = max(dot(N, H), 0.0);
    float hdv = max(dot(H, V), 0.0);
    vec3 spec = distribution_ggx(ndh, roughness) * geometry_smith(ndv, ndl, roughness)
              * fresnel_schlick(hdv, f0) / (4.0 * ndv * ndl + 0.0001);
    color += spec * SUN_RADIANCE * ndl * shadow;
    uint cluster = cluster_index(v_world_pos);
    for (uint w = 0; w < CLUSTER_WORDS; ++w) {
        uint bits = cluster_masks[cluster * CLUSTER_WORDS + w];
        while (bits != 0u) {
            uint b = uint(findLSB(bits));
            bits &= bits - 1u;
            color += punctual(lights[w * 32u + b], N, V, R, v_world_pos, vec3(0.0), f0, roughness, 0.0);
        }
    }

    // Height fog along the view ray, as on everything else.
    vec3 dir = -V;
    float fog = 1.0 - exp(-fog_optical_depth(pc.camera_pos.xyz, dir, dist));
    color = mix(color, fog_color(dir), fog);
    o_color = vec4(color, 1.0);
}
#else
void main() {
    Material m = materials[v_material];
    vec3 albedo = texture(textures[nonuniformEXT(m.tex.x)], v_uv).rgb * m.base_color_factor.rgb;
    // green = roughness, blue = metallic; red = occlusion if it's an ARM map.
    vec3 arm = texture(textures[nonuniformEXT(m.tex.z)], v_uv).rgb;
    vec2 mr = arm.gb;
    float roughness = clamp(m.params.x * mr.x, 0.04, 1.0);
    float metallic = clamp(m.emissive.a * mr.y, 0.0, 1.0);

    vec3 ng = normalize(v_normal);
    // A double-sided card seen from behind is the same surface facing us.
    if ((m.tex.w & MATERIAL_DOUBLE_SIDED) != 0u && !gl_FrontFacing) {
        ng = -ng;
    }
    // The level's sky visibility (§13), sampled one cell off the surface so
    // the cells behind it don't count. Here, before the lights, rather than
    // where the ambient uses it: its eight fetches' latency then overlaps the
    // texture samples above instead of stalling after the light loop
    // (lights120 geo 0.96 -> 0.90 ms, §13).
    vec4 sv = sky_visibility(v_world_pos + ng / g.sky_origin.w);
    // The material's own occlusion (glTF occlusionTexture, red), for the
    // ambient: from the MR sample when they're one image, else its own (none
    // is slot 0, white). The branch is per material, so uniform in a quad.
    uint occ_slot = m.tex.w >> MATERIAL_OCCLUSION_SHIFT;
    float occ_sample = occ_slot == m.tex.z ? arm.r
                     : occ_slot == 0u ? 1.0
                     : texture(textures[nonuniformEXT(occ_slot)], v_uv).r;
    float material_ao = material_occlusion(occ_sample, m.params.z);
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
    float gs = geometry_smith(ndv, ndl, roughness);
    vec3 f = fresnel_schlick(hdv, f0);
    vec3 specular = (ndf * gs * f) / (4.0 * ndv * ndl + 0.0001);
    vec3 kd = (vec3(1.0) - f) * (1.0 - metallic);
    // Sun shadow occludes the direct term only; ambient/IBL stays unshadowed.
    float shadow = sun_shadow(v_world_pos, ng);
    vec3 lo = (kd * albedo / PI + specular) * SUN_RADIANCE * ndl * shadow;

    // Punctual lights (§12), added to the same accumulator — only those this
    // fragment's cluster lists, visited in ascending index order. A light
    // outside the cluster contributes exactly zero (the falloff window reaches
    // 0 at the radius), so with conservative clusters this sums precisely what
    // the brute-force loop over every light did. Unshadowed: point shadows need
    // cube maps, which is a feature of its own.
    uint cluster = cluster_index(v_world_pos);
    vec3 R = reflect(-V, N);
    for (uint w = 0; w < CLUSTER_WORDS; ++w) {
        uint bits = cluster_masks[cluster * CLUSTER_WORDS + w];
        while (bits != 0u) {
            uint b = uint(findLSB(bits));
            bits &= bits - 1u;
            lo += punctual(lights[w * 32u + b], N, V, R, v_world_pos, albedo, f0, roughness, metallic);
        }
    }

    // --- Ambient (analytic IBL: split-sum against the procedural sky) ---
    // Occluded by the level's sky visibility (§13), `sv` above.
    vec3 fr = fresnel_schlick_roughness(ndv, f0, roughness);
    vec3 kd_amb = (vec3(1.0) - fr) * (1.0 - metallic);
    vec3 diffuse_ibl = sky_irradiance_occluded(N, sv) * albedo;
    vec3 r = reflect(-V, N);
    vec3 prefiltered = mix(sky(r), sky_irradiance(r), roughness); // crude roughness blur
    vec2 ab = env_brdf_approx(roughness, ndv);
    vec3 specular_ibl = prefiltered * (f0 * ab.x + ab.y) * sky_specular(sv, r);
    // Occlusion on the ambient only: the material's, and GTAO's contact
    // shadowing within a metre (§13). They see many of the same crevices,
    // so the darker of the two rather than their product.
    float ao = material_ao;
    if (g.ao_params.x > 0.5) {
        ao = combine_occlusion(ao, gtao_upsample(v_world_pos, ng));
    }
    diffuse_ibl *= ao_multibounce(ao, albedo);
    specular_ibl *= ao_specular(ndv, ao, roughness);
    vec3 ambient = kd_amb * diffuse_ibl + specular_ibl;

    vec3 color = ambient + lo + m.emissive.rgb;

    // Height fog along the view ray (§13). Reads as depth and hides the
    // finite ground's edge. sky() has no sun disk, so no searing dot bleeds
    // into the haze.
    vec3 to_frag = v_world_pos - pc.camera_pos.xyz;
    float dist = length(to_frag);
    vec3 dir = to_frag / max(dist, 1e-6);
    float fog = 1.0 - exp(-fog_optical_depth(pc.camera_pos.xyz, dir, dist));
    color = mix(color, fog_color(dir), fog);

    o_color = vec4(color, 1.0);
}
#endif
