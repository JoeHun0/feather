#version 450

// Classic attributeless fullscreen triangle. Three vertices covering the screen;
// v_uv maps [0,1] across the framebuffer (0,0 = top-left, matching the HDR
// target's texel layout, so the tonemap samples it 1:1 with no Y flip).
layout(location = 0) out vec2 v_uv;

void main() {
    v_uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(v_uv * 2.0 - 1.0, 0.0, 1.0);
}
