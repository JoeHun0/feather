#version 450

layout(push_constant) uniform Push {
    mat4 view_proj;
    vec4 light_dir;   // xyz = direction the light travels
    vec4 camera_pos;  // xyz = world-space eye
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
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

layout(location = 0) out vec3 v_normal;
layout(location = 1) out flat uint v_material;
layout(location = 2) out vec2 v_uv;
layout(location = 3) out vec3 v_world_pos;

void main() {
    Instance it = insts[gl_InstanceIndex];
    vec4 world = it.model * vec4(in_pos, 1.0);
    gl_Position = pc.view_proj * world;
    v_world_pos = world.xyz;
    // Normals transform by the inverse-transpose of the model's 3x3 (mat3(model)
    // skews them under any non-uniform scale). The cofactor matrix is
    // det * inverse-transpose: three crosses, no division, and the magnitude
    // drops out in normalize(). Only det's sign matters: negative (mirrored)
    // would point normals inward. Not `sign(det)`: a flattened (det = 0)
    // transform still has a valid cofactor that collapses normals onto the
    // flattened axis.
    mat3 m = mat3(it.model);
    mat3 cof = mat3(cross(m[1], m[2]), cross(m[2], m[0]), cross(m[0], m[1]));
    float flip = dot(m[0], cof[0]) < 0.0 ? -1.0 : 1.0;
    v_normal = flip * normalize(cof * in_normal);
    v_material = it.material_id;
    v_uv = in_uv;
}
