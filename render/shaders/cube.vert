#version 450

layout(push_constant) uniform Push {
    mat4 mvp;
} pc;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_color;

layout(location = 0) out vec3 v_color;

void main() {
    gl_Position = pc.mvp * vec4(in_pos, 1.0);
    v_color = in_color;
}
