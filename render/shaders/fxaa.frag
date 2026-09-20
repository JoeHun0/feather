#version 450

// FXAA (Lottes' compact form) over the tonemapped LDR image, writing the
// swapchain. Post-AA (§13): one fullscreen pass whose cost tracks resolution
// only, unlike MSAA which scales with geometry.
//
// Colour space: the LDR source is an _SRGB image, so sampling hands back LINEAR
// values and the _SRGB swapchain re-encodes on store — no manual conversion, and
// no double correction. Blending therefore happens in linear, which is the
// physically correct place to average. Only FXAA's *edge thresholds* want
// perceptual luma, and sqrt() approximates the sRGB curve closely enough for a
// heuristic at one instruction per tap (a true OETF would cost a pow each).

layout(set = 0, binding = 0) uniform sampler2D u_ldr;

layout(push_constant) uniform Push {
    vec2 inv_screen; // 1 / framebuffer size
} pc;

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 o_color;

const float EDGE_MIN = 1.0 / 16.0; // absolute contrast floor
const float EDGE_MUL = 1.0 / 8.0;  // contrast relative to the local maximum
const float SPAN_MAX = 8.0;        // longest blend, in pixels

// Perceptual-ish luma from a linear sample: gamma 2.0 in place of sRGB's ~2.2.
float luma(vec3 c) {
    return sqrt(dot(c, vec3(0.299, 0.587, 0.114)));
}

void main() {
    vec3 rgb_m = texture(u_ldr, v_uv).rgb;
    float l_m = luma(rgb_m);
    float l_nw = luma(textureOffset(u_ldr, v_uv, ivec2(-1, -1)).rgb);
    float l_ne = luma(textureOffset(u_ldr, v_uv, ivec2(1, -1)).rgb);
    float l_sw = luma(textureOffset(u_ldr, v_uv, ivec2(-1, 1)).rgb);
    float l_se = luma(textureOffset(u_ldr, v_uv, ivec2(1, 1)).rgb);

    float l_min = min(l_m, min(min(l_nw, l_ne), min(l_sw, l_se)));
    float l_max = max(l_m, max(max(l_nw, l_ne), max(l_sw, l_se)));
    float range = l_max - l_min;

    // Flat enough to leave alone. The early-out is most of why FXAA is cheap.
    if (range < max(EDGE_MIN, l_max * EDGE_MUL)) {
        o_color = vec4(rgb_m, 1.0);
        return;
    }

    // Blend along the edge, i.e. perpendicular to the luma gradient.
    vec2 dir = vec2(
        -((l_nw + l_ne) - (l_sw + l_se)),
        ((l_nw + l_sw) - (l_ne + l_se))
    );
    // Keep near-axis-aligned edges from producing an enormous direction vector.
    float reduce = max((l_nw + l_ne + l_sw + l_se) * 0.25 * EDGE_MUL, 1.0 / 128.0);
    float rcp_dir = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * rcp_dir, vec2(-SPAN_MAX), vec2(SPAN_MAX)) * pc.inv_screen;

    // Inner pair is always safe; the wider pair is sharper but can overshoot, so
    // it is only kept when its luma stays inside the neighbourhood's range.
    vec3 rgb_a = 0.5
        * (texture(u_ldr, v_uv + dir * (1.0 / 3.0 - 0.5)).rgb
            + texture(u_ldr, v_uv + dir * (2.0 / 3.0 - 0.5)).rgb);
    vec3 rgb_b = rgb_a * 0.5
        + 0.25
            * (texture(u_ldr, v_uv + dir * -0.5).rgb
                + texture(u_ldr, v_uv + dir * 0.5).rgb);

    float l_b = luma(rgb_b);
    vec3 result = (l_b < l_min || l_b > l_max) ? rgb_a : rgb_b;

    o_color = vec4(result, 1.0);
}
