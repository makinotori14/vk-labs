#version 450

layout(location = 0) in vec3 worldPosition;
layout(location = 1) in vec3 worldNormal;
layout(location = 2) in vec2 uv;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0, std140) uniform Scene {
    mat4 model;
    mat4 viewProjection;
    mat4 normalMatrix;
    vec4 cameraPosition;
    vec4 lightDirectionIntensity;
    vec4 lightColor;
    vec4 ambientColorIntensity;
    vec4 specularShininess;
} scene;

layout(set = 0, binding = 1) uniform sampler2D albedoTexture;

vec3 safeNormalize(vec3 value) {
    return value * inversesqrt(max(dot(value, value), 1e-12));
}

void main() {
    vec3 albedo = texture(albedoTexture, uv).rgb;
    vec3 normal = normalize(worldNormal);
    // Направление лучей противоположно направлению от поверхности к свету.
    vec3 light = -normalize(scene.lightDirectionIntensity.xyz);
    vec3 view = safeNormalize(scene.cameraPosition.xyz - worldPosition);
    vec3 halfway = safeNormalize(light + view);

    float diffuse = max(dot(normal, light), 0.0);
    float specular = 0.0;
    if (diffuse > 0.0) {
        specular = pow(max(dot(normal, halfway), 0.0), scene.specularShininess.w);
    }
    vec3 ambient = scene.ambientColorIntensity.rgb * scene.ambientColorIntensity.w * albedo;
    vec3 direct = scene.lightColor.rgb * scene.lightDirectionIntensity.w
        * (albedo * diffuse + scene.specularShininess.rgb * specular);
    outColor = vec4(ambient + direct, 1.0);
}
