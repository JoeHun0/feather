#version 450

// Classic attributeless fullscreen triangle. Three vertices covering the screen;
// v_uv maps [0,1] across the framebuffer (0,0 = top-left, matching the HDR
// target's texel layout, so the tonemap samples it 1:1 with no Y flip).
//
// z = 1.0 puts it on the far plane. The sky pass needs that: it draws after the
// opaque geometry with a LESS_OR_EQUAL depth test, so it survives only where the
// depth buffer is still at its cleared 1.0 (i.e. background), and is rejected
// wherever geometry already wrote nearer depth. The tonemap pass also uses this
// shader but renders with no depth attachment, so the value is irrelevant there.
layout(location = 0) out vec2 v_uv;

void main() {
    v_uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(v_uv * 2.0 - 1.0, 1.0, 1.0);
}
