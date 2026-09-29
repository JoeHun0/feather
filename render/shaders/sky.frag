#version 450

// Fullscreen background sky. Reconstructs a world-space view ray per pixel from
// the inverse view-projection and evaluates the same procedural sky the mesh
// shader reflects for IBL — here with a sharp sun disk so the sun is visible.

layout(push_constant) uniform Push {
    mat4 inv_view_proj;
    vec4 camera_pos;  // xyz
    vec4 light_dir;   // xyz = direction the light travels
} pc;

// The level's atmosphere (§13) comes from mesh.frag's per-frame globals: its
// set 0, bound here too, whose block this is, text for text (a test checks),
// as are the #defines. sky() and the fog functions are copies of mesh.frag's,
// text for text (a test checks); the sharp sun disk is added in main() alone.
const int SHADOW_CASCADES = 4;
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

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color; // linear HDR

vec3 sky(vec3 d) {
    vec3 sundir = normalize(-pc.light_dir.xyz);
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
        float s = max(dot(dir, normalize(-pc.light_dir.xyz)), 0.0);
        col += SUN_COLOR * FOG_SUN * pow(s, 8.0);
    }
    return col;
}

void main() {
    // v_uv in [0,1] -> NDC (matches the fullscreen triangle's clip position).
    vec2 ndc = v_uv * 2.0 - 1.0;
    vec4 world = pc.inv_view_proj * vec4(ndc, 1.0, 1.0); // far plane
    vec3 dir = normalize(world.xyz / world.w - pc.camera_pos.xyz);
    float s = max(dot(dir, normalize(-pc.light_dir.xyz)), 0.0);
    vec3 col = sky(dir) + SUN_COLOR * pow(s, 4000.0) * SUN_DISK * SKY_INTENSITY;
    // The background is infinitely far, so it fogs only through a height
    // layer (FOG_FALLOFF > 0): looking up, the fog above is finite; below the
    // horizon it's total, which hides the void past the ground's edge.
    // Uniform fog leaves the sky alone, as it always did.
    if (FOG_FALLOFF > 0.0) {
        float fog = 1.0 - exp(-fog_optical_depth(pc.camera_pos.xyz, dir, -1.0));
        col = mix(col, fog_color(dir), fog);
    }
    o_color = vec4(col, 1.0);
}
