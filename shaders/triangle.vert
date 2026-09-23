#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;

layout(location = 0) out vec3 fragment_color;

layout(set = 0, binding = 0) uniform UniformData {
	mat4 mvp;
} uniform_data;

void main() {
	gl_Position = uniform_data.mvp * vec4(position, 1.0);
	fragment_color = color;
}
