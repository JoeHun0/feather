#version 450

layout(push_constant) uniform Push {
    mat4 view_proj;
    vec4 light_dir; // xyz = direction the light travels
} pc;

struct Instance {
    mat4 model;
    vec4 color;
};
layout(set = 0, binding = 0) readonly buffer Instances {
    Instance insts[];
};

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec3 v_color;

void main() {
    Instance it = insts[gl_InstanceIndex];
    gl_Position = pc.view_proj * it.model * vec4(in_pos, 1.0);
    // Uniform scale, so the model's 3x3 transforms normals correctly.
    v_normal = normalize(mat3(it.model) * in_normal);
    v_color = it.color.rgb;
}
