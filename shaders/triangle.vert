#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

layout(location = 0) out vec3 fragment_color;

layout(set = 0, binding = 0) uniform SceneData {
	mat4 view;
	mat4 projection;
} scene_ubo;

layout(set = 1, binding = 0) uniform ObjectData {
	mat4 model;
	vec4 tint;
} object_ubo;

void main() {
	gl_Position = scene_ubo.projection * scene_ubo.view *
	              object_ubo.model * vec4(position, 1.0);
	fragment_color = color * object_ubo.tint.rgb;
}
