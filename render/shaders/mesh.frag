#version 450

layout(push_constant) uniform Push {
    mat4 view_proj;
    vec4 light_dir;
} pc;

layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_color;

layout(location = 0) out vec4 o_color;

void main() {
    vec3 L = normalize(pc.light_dir.xyz);
    float ndl = max(dot(normalize(v_normal), -L), 0.0);
    float ambient = 0.15;
    vec3 lit = v_color * (ambient + (1.0 - ambient) * ndl);
    o_color = vec4(lit, 1.0);
}
