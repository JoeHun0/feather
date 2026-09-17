#version 450

layout(push_constant) uniform Push {
    mat4 mvp;
} pc;

layout(location = 0) out vec3 v_color;

// 36 vertices (6 faces x 2 tris), wound CCW when viewed from outside.
const vec3 positions[36] = vec3[](
    // +X (red)
    vec3( 0.5,-0.5,-0.5), vec3( 0.5, 0.5,-0.5), vec3( 0.5, 0.5, 0.5),
    vec3( 0.5,-0.5,-0.5), vec3( 0.5, 0.5, 0.5), vec3( 0.5,-0.5, 0.5),
    // -X (green)
    vec3(-0.5,-0.5,-0.5), vec3(-0.5,-0.5, 0.5), vec3(-0.5, 0.5, 0.5),
    vec3(-0.5,-0.5,-0.5), vec3(-0.5, 0.5, 0.5), vec3(-0.5, 0.5,-0.5),
    // +Y (blue)
    vec3(-0.5, 0.5,-0.5), vec3(-0.5, 0.5, 0.5), vec3( 0.5, 0.5, 0.5),
    vec3(-0.5, 0.5,-0.5), vec3( 0.5, 0.5, 0.5), vec3( 0.5, 0.5,-0.5),
    // -Y (yellow)
    vec3(-0.5,-0.5,-0.5), vec3( 0.5,-0.5,-0.5), vec3( 0.5,-0.5, 0.5),
    vec3(-0.5,-0.5,-0.5), vec3( 0.5,-0.5, 0.5), vec3(-0.5,-0.5, 0.5),
    // +Z (cyan)
    vec3(-0.5,-0.5, 0.5), vec3( 0.5,-0.5, 0.5), vec3( 0.5, 0.5, 0.5),
    vec3(-0.5,-0.5, 0.5), vec3( 0.5, 0.5, 0.5), vec3(-0.5, 0.5, 0.5),
    // -Z (magenta)
    vec3(-0.5,-0.5,-0.5), vec3(-0.5, 0.5,-0.5), vec3( 0.5, 0.5,-0.5),
    vec3(-0.5,-0.5,-0.5), vec3( 0.5, 0.5,-0.5), vec3( 0.5,-0.5,-0.5)
);

const vec3 face_colors[6] = vec3[](
    vec3(1.0, 0.0, 0.0), // +X
    vec3(0.0, 1.0, 0.0), // -X
    vec3(0.0, 0.0, 1.0), // +Y
    vec3(1.0, 1.0, 0.0), // -Y
    vec3(0.0, 1.0, 1.0), // +Z
    vec3(1.0, 0.0, 1.0)  // -Z
);

void main() {
    gl_Position = pc.mvp * vec4(positions[gl_VertexIndex], 1.0);
    v_color = face_colors[gl_VertexIndex / 6];
}
