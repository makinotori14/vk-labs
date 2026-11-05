#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace application {

inline glm::mat4 calculateModel(const glm::vec3& scale, const glm::vec3& translation,
                                float rotation_degrees) {
	// M = T * R * S: сначала масштабирование, затем поворот и перемещение.
	glm::mat4 model = glm::translate(glm::mat4(1.0f), translation);
	model = glm::rotate(model, glm::radians(rotation_degrees), glm::vec3(0.0f, 1.0f, 0.0f));
	model = glm::scale(model, scale);
	return model;
}

inline glm::mat4 calculateViewProjection(const glm::vec3& eye, const glm::vec3& rotation_degrees,
                                         float fov_degrees, float near_plane, float far_plane,
                                         bool orthographic, float aspect_ratio) {
	// Локальные оси камеры: взгляд вдоль -Z, верх вдоль +Y.
	glm::mat4 camera = glm::translate(glm::mat4(1.0f), eye);
	camera = glm::rotate(camera, glm::radians(rotation_degrees.y), glm::vec3(0, 1, 0));
	camera = glm::rotate(camera, glm::radians(rotation_degrees.x), glm::vec3(1, 0, 0));
	camera = glm::rotate(camera, glm::radians(rotation_degrees.z), glm::vec3(0, 0, 1));
	const glm::mat4 view = glm::inverse(camera);

	const float vertical_fov = glm::radians(fov_degrees);
	glm::mat4 projection;
	if (orthographic) {
		// Сохраняем масштаб прежней ортографической проекции в точке взгляда камеры.
		const float half_height = glm::max(glm::length(eye), near_plane) * glm::tan(vertical_fov * 0.5f);
		const float half_width = half_height * aspect_ratio;
		projection = glm::ortho(-half_width, half_width, -half_height, half_height, near_plane, far_plane);
	} else {
		projection = glm::perspective(vertical_fov, aspect_ratio, near_plane, far_plane);
	}
	// GLM_FORCE_DEPTH_ZERO_TO_ONE задаёт глубину Vulkan [0, 1]; переворачиваем ось Y.
	projection[1][1] *= -1.0f;
	return projection * view;
}

} // namespace application
