#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 0) out vec3 worldPosition;
layout(location = 1) out vec3 worldNormal;

layout(set = 0, binding = 0, std140) uniform Scene {
    mat4 model;
    mat4 viewProjection;
    mat4 normalMatrix;
    vec4 cameraPosition;
    vec4 lightDirectionIntensity;
    vec4 lightColor;
    vec4 ambientColorIntensity;
    vec4 albedo;
    vec4 specularShininess;
} scene;

void main() {
    vec4 position = scene.model * vec4(inPosition, 1.0);
    worldPosition = position.xyz;
    worldNormal = mat3(scene.normalMatrix) * inNormal;
    gl_Position = scene.viewProjection * position;
}
