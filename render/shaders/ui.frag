#version 450

// Overlay fragment (§19): the atlas holds coverage in **alpha** (a rasterized
// mono font plus one solid cell for filled rectangles — see render/font.rs),
// so a sample only modulates alpha.
//
// Colours are emitted LINEAR: this pass draws into the _SRGB swapchain, which
// encodes on store, and Vulkan blends _SRGB attachments in linear space.

layout(set = 0, binding = 0) uniform sampler2D u_atlas;

layout(location = 0) in vec2 v_uv;
layout(location = 1) in vec4 v_color;
layout(location = 0) out vec4 o_color;

void main() {
    float coverage = texture(u_atlas, v_uv).a;
    o_color = vec4(v_color.rgb, v_color.a * coverage);
}
