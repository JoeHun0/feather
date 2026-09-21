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
// Bindless: slot 0 = white, slot 1 = flat normal, then one slot per unique image
// the level uses. Runtime-sized: the renderer sizes the binding per level,
// limited only by the device. Base color _SRGB; normal + MR are _UNORM.
layout(set = 0, binding = 2) uniform sampler2D textures[];
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
} g;

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
