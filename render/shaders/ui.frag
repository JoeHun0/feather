#version 450

// Overlay fragment (§19): the atlas holds coverage (a 5x7 pixel font plus one
// solid texel for filled rectangles), so it only modulates alpha.
//
// Colours are emitted LINEAR: this pass draws into the _SRGB swapchain, which
// encodes on store, and Vulkan blends _SRGB attachments in linear space.

layout(set = 0, binding = 0) uniform sampler2D u_atlas;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 0) out vec4 o_color;

void main() {
    float coverage = texture(u_atlas, v_uv).r;
    o_color = vec4(v_color.rgb, v_color.a * coverage);
}
