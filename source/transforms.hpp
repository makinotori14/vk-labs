#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace application {

inline glm::mat4 calculateMvp(const glm::vec3& scale, const glm::vec3& translation,
                              float rotation_degrees, bool orthographic, float aspect_ratio) {
	// M = T * R * S: сначала масштабирование, затем поворот и перемещение.
	glm::mat4 model = glm::translate(glm::mat4(1.0f), translation);
	model = glm::rotate(model, glm::radians(rotation_degrees), glm::vec3(0.0f, 1.0f, 0.0f));
	model = glm::scale(model, scale);

	// Неподвижная камера смотрит на начало координат.
	const glm::vec3 eye(3.0f, 2.2f, 4.5f);
	const glm::mat4 view = glm::lookAt(eye, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

	constexpr float near_plane = 0.1f;
	constexpr float far_plane = 100.0f;
	const float vertical_fov = glm::radians(45.0f);
	glm::mat4 projection;
	if (orthographic) {
		// Сохраняем масштаб прежней ортографической проекции в точке взгляда камеры.
		const float half_height = glm::length(eye) * glm::tan(vertical_fov * 0.5f);
		const float half_width = half_height * aspect_ratio;
		projection = glm::ortho(-half_width, half_width, -half_height, half_height, near_plane, far_plane);
	} else {
		projection = glm::perspective(vertical_fov, aspect_ratio, near_plane, far_plane);
	}
	// GLM_FORCE_DEPTH_ZERO_TO_ONE задаёт глубину Vulkan [0, 1]; переворачиваем ось Y.
	projection[1][1] *= -1.0f;
	return projection * view * model;
}

} // namespace application
