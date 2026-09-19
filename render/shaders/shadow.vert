#version 450

// Depth-only sun shadow pass (§11). Transforms each instance's vertices into the
// light's clip space; no fragment shader — only depth is written. Reuses the mesh
// instance SSBO (binding 0); the light-space matrix comes in as a push constant.

layout(push_constant) uniform Push {
    mat4 light_view_proj;
} pc;

struct Instance {
    mat4 model;
    uint material_id;
    uint _p0;
    uint _p1;
    uint _p2;
};
layout(set = 0, binding = 0) readonly buffer Instances {
    Instance insts[];
};

layout(location = 0) in vec3 in_pos;

void main() {
    Instance it = insts[gl_InstanceIndex];
    gl_Position = pc.light_view_proj * it.model * vec4(in_pos, 1.0);
}
