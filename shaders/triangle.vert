#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

layout(location = 0) out vec3 fragment_color;

layout(push_constant) uniform PushConstants {
	mat4 mvp;
} push_constants;

void main() {
	gl_Position = push_constants.mvp * vec4(position, 1.0);
	fragment_color = color;
}
