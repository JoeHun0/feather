#version 450

layout(push_constant) uniform Push {
    mat4 view_proj;
} pc;

layout(set = 0, binding = 0) readonly buffer Instances {
    mat4 models[];
};

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_color;

layout(location = 0) out vec3 v_color;

void main() {
    mat4 model = models[gl_InstanceIndex];
    gl_Position = pc.view_proj * model * vec4(in_pos, 1.0);
    v_color = in_color;
}
