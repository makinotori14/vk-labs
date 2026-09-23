#version 450

layout(location = 0) out vec3 fragment_color;

layout(push_constant) uniform PushConstants {
	mat4 mvp;
} push_constants;

const vec3 positions[18] = vec3[](
	// Front face
	vec3( 0.0,  0.8,  0.0),
	vec3(-0.7, -0.6, -0.7),
	vec3( 0.7, -0.6, -0.7),

	// Right face
	vec3( 0.0,  0.8,  0.0),
	vec3( 0.7, -0.6, -0.7),
	vec3( 0.7, -0.6,  0.7),

	// Back face
	vec3( 0.0,  0.8,  0.0),
	vec3( 0.7, -0.6,  0.7),
	vec3(-0.7, -0.6,  0.7),

	// Left face
	vec3( 0.0,  0.8,  0.0),
	vec3(-0.7, -0.6,  0.7),
	vec3(-0.7, -0.6, -0.7),

	// Base (two triangles)
	vec3(-0.7, -0.6, -0.7),
	vec3(-0.7, -0.6,  0.7),
	vec3( 0.7, -0.6,  0.7),

	vec3(-0.7, -0.6, -0.7),
	vec3( 0.7, -0.6,  0.7),
	vec3( 0.7, -0.6, -0.7)
);

const vec3 face_colors[6] = vec3[](
	vec3(1.0, 0.0, 0.0),
	vec3(0.0, 1.0, 0.0),
	vec3(0.0, 0.0, 1.0),
	vec3(1.0, 0.8, 0.0),
	vec3(0.7, 0.2, 1.0),
	vec3(0.7, 0.2, 1.0)
);

void main() {
	gl_Position = push_constants.mvp * vec4(positions[gl_VertexIndex], 1.0);
	fragment_color = face_colors[gl_VertexIndex / 3];
}
