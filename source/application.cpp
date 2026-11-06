#include "application.hpp"
#include "transforms.hpp"
#include "gpu_resources.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>

#include <imgui.h>

namespace application {

namespace {

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec2 uv;
};

// Вершины разделены между гранями, чтобы сохранить плоские нормали пирамиды.
const glm::vec3 positions[] = {
	{-1, -1,  1}, { 1, -1,  1}, { 0,  1,  0}, // Передняя грань.
	{ 1, -1,  1}, { 1, -1, -1}, { 0,  1,  0}, // Правая грань.
	{ 1, -1, -1}, {-1, -1, -1}, { 0,  1,  0}, // Задняя грань.
	{-1, -1, -1}, {-1, -1,  1}, { 0,  1,  0}, // Левая грань.
	{-1, -1, -1}, { 1, -1, -1}, { 1, -1,  1}, // Основание.
	{-1, -1, -1}, { 1, -1,  1}, {-1, -1,  1},
};

struct Mesh {
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
};

Mesh createMesh() {
	Mesh mesh;
	const glm::vec2 side_uv[] = {{0, 1}, {1, 1}, {0.5f, 0}};
	for (size_t i = 0; i < std::size(positions); i += 3) {
		const glm::vec3 normal = glm::normalize(glm::cross(
			positions[i + 1] - positions[i], positions[i + 2] - positions[i]));
		for (size_t j = 0; j < 3; ++j) {
			const auto position = positions[i + j];
			// Боковые грани получают треугольную развёртку; основание — плоскую по XZ.
			const glm::vec2 uv = i < 12 ? side_uv[j]
				: glm::vec2((position.x + 1.0f) * 0.5f, (1.0f - position.z) * 0.5f);
			const auto found = std::find_if(mesh.vertices.begin(), mesh.vertices.end(),
				[&](const Vertex& vertex) {
					return vertex.position == position && vertex.normal == normal && vertex.uv == uv;
				});
			const auto index = uint32_t(found - mesh.vertices.begin());
			if (found == mesh.vertices.end()) {
				mesh.vertices.push_back({position, normal, uv});
			}
			mesh.indices.push_back(index);
		}
	}
	return mesh;
}

struct alignas(16) UniformData {
	glm::mat4 model;
	glm::mat4 view_projection;
	glm::mat4 normal_matrix;
	glm::vec4 camera_position;
	glm::vec4 light_direction_intensity;
	glm::vec4 light_color;
	glm::vec4 ambient_color_intensity;
	glm::vec4 specular_shininess;
};

// Все поля совпадают с расположением std140 в обоих шейдерах.
static_assert(offsetof(UniformData, model) == 0);
static_assert(offsetof(UniformData, view_projection) == 64);
static_assert(offsetof(UniformData, normal_matrix) == 128);
static_assert(offsetof(UniformData, camera_position) == 192);
static_assert(offsetof(UniformData, light_direction_intensity) == 208);
static_assert(offsetof(UniformData, light_color) == 224);
static_assert(offsetof(UniformData, ambient_color_intensity) == 240);
static_assert(offsetof(UniformData, specular_shininess) == 256);
static_assert(sizeof(UniformData) == 272);

struct Controls {
	int projection = 0;
	float rotation_degrees = 0.0f;
	float scale[3] = {1.0f, 1.0f, 1.0f};
	float translation[3] = {0.0f, 0.0f, 0.0f};
	float camera_position[3] = {3.0f, 2.2f, 4.5f};
	float camera_rotation[3] = {-22.13f, 33.69f, 0.0f}; // Pitch, yaw, roll.
	float fov = 45.0f;
	float near_plane = 0.1f;
	float far_plane = 100.0f;
	float light_rotation[2] = {-45.0f, -35.0f}; // Наклон и азимут лучей.
	float light_color[3] = {1.0f, 1.0f, 1.0f};
	float light_intensity = 1.0f;
	float ambient_color[3] = {1.0f, 1.0f, 1.0f};
	float ambient_intensity = 0.1f;
	float specular[3] = {0.5f, 0.5f, 0.5f};
	float shininess = 32.0f;
};

graphics::resources::Buffer vertex_buffer;
graphics::resources::Buffer index_buffer;
graphics::resources::Texture albedo_texture;
uint32_t index_count;
VkDeviceSize uniform_stride;
// В сцене одна пирамида; массив UBO содержит один элемент с выровненным шагом.
constexpr uint32_t model_count = 1;
VkBuffer uniform_buffer;
VmaAllocation uniform_allocation;
void* uniform_mapped_data;
VkDescriptorSetLayout descriptor_set_layout;
VkDescriptorPool descriptor_pool;
VkDescriptorSet descriptor_set;
VkPipelineLayout pipeline_layout;
VkPipeline pipeline;
Controls controls;

// Углы задают ненулевое единичное направление распространения света.
glm::vec3 lightDirection() {
	const float pitch = glm::radians(controls.light_rotation[0]);
	const float yaw = glm::radians(controls.light_rotation[1]);
	return {glm::cos(pitch) * glm::sin(yaw), glm::sin(pitch),
	        -glm::cos(pitch) * glm::cos(yaw)};
}

bool createUniformResources() {
	auto& context = graphics::internal::context;
	VkPhysicalDeviceProperties properties{};
	vkGetPhysicalDeviceProperties(context.physical_device, &properties);
	const VkDeviceSize alignment = properties.limits.minUniformBufferOffsetAlignment;
	uniform_stride = (sizeof(UniformData) + alignment - 1) / alignment * alignment;
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = uniform_stride * model_count, .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	VmaAllocationInfo allocation{};
	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &uniform_buffer, &uniform_allocation, &allocation) != VK_SUCCESS) {
		return false;
	}
	uniform_mapped_data = allocation.pMappedData;

	const VkDescriptorSetLayoutBinding bindings[] = {
		{.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
		 .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT},
		{.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		 .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT},
	};
	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = uint32_t(std::size(bindings)), .pBindings = bindings,
	};
	if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
		return false;
	}
	const VkDescriptorPoolSize pool_sizes[] = {
		{.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .descriptorCount = 1},
		{.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1},
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1, .poolSizeCount = uint32_t(std::size(pool_sizes)), .pPoolSizes = pool_sizes,
	};
	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
		return false;
	}
	const VkDescriptorSetAllocateInfo set_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool, .descriptorSetCount = 1, .pSetLayouts = &descriptor_set_layout,
	};
	if (vkAllocateDescriptorSets(context.device, &set_info, &descriptor_set) != VK_SUCCESS) {
		return false;
	}
	const VkDescriptorBufferInfo descriptor_buffer = {
		.buffer = uniform_buffer, .offset = 0, .range = sizeof(UniformData),
	};
	const VkDescriptorImageInfo descriptor_image = {
		.sampler = albedo_texture.sampler, .imageView = albedo_texture.view,
		.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
	};
	const VkWriteDescriptorSet writes[] = {
		{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		 .dstSet = descriptor_set, .dstBinding = 0, .descriptorCount = 1,
		 .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, .pBufferInfo = &descriptor_buffer},
		{.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		 .dstSet = descriptor_set, .dstBinding = 1, .descriptorCount = 1,
		 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &descriptor_image},
	};
	vkUpdateDescriptorSets(context.device, uint32_t(std::size(writes)), writes, 0, nullptr);
	return true;
}

bool loadShader(const char* path, VkShaderModule& shader) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		std::cerr << "Не удалось открыть шейдер: " << path << '\n';
		return false;
	}
	const auto size = file.tellg();
	if (size <= 0 || size % sizeof(uint32_t) != 0) {
		std::cerr << "Некорректный шейдер SPIR-V: " << path << '\n';
		return false;
	}
	std::vector<uint32_t> code(static_cast<size_t>(size) / sizeof(uint32_t));
	file.seekg(0);
	if (!file.read(reinterpret_cast<char*>(code.data()), static_cast<std::streamsize>(size))) {
		return false;
	}
	const VkShaderModuleCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = static_cast<size_t>(size),
		.pCode = code.data(),
	};
	return vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &shader) == VK_SUCCESS;
}

bool createPipeline() {
	auto& context = graphics::internal::context;
	VkShaderModule vertex_shader = VK_NULL_HANDLE;
	VkShaderModule fragment_shader = VK_NULL_HANDLE;
	if (!loadShader(APP_SHADER_DIR "/pyramid.vert.spv", vertex_shader)) {
		return false;
	}
	if (!loadShader(APP_SHADER_DIR "/pyramid.frag.spv", fragment_shader)) {
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		return false;
	}

	const VkPipelineShaderStageCreateInfo stages[] = {
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		 .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vertex_shader, .pName = "main"},
		{.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		 .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragment_shader, .pName = "main"},
	};
	const VkVertexInputBindingDescription binding = {
		.binding = 0, .stride = sizeof(Vertex), .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};
	const VkVertexInputAttributeDescription attributes[] = {
		{.location = 0, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, position)},
		{.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, normal)},
		{.location = 2, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT, .offset = offsetof(Vertex, uv)},
	};
	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
		.vertexAttributeDescriptionCount = uint32_t(std::size(attributes)), .pVertexAttributeDescriptions = attributes,
	};
	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};
	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1, .scissorCount = 1,
	};
	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
		.lineWidth = 1.0f,
	};
	const VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};
	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};
	const VkPipelineColorBlendAttachmentState color_attachment = {
		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
	};
	const VkPipelineColorBlendStateCreateInfo color_blending = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1, .pAttachments = &color_attachment,
	};
	const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = uint32_t(std::size(dynamic_states)), .pDynamicStates = dynamic_states,
	};
	const VkGraphicsPipelineCreateInfo info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = uint32_t(std::size(stages)), .pStages = stages,
		.pVertexInputState = &vertex_input, .pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state, .pRasterizationState = &rasterization,
		.pMultisampleState = &multisampling, .pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blending, .pDynamicState = &dynamic_state,
		.layout = pipeline_layout, .renderPass = context.render_pass,
	};
	const VkResult result = vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline);
	vkDestroyShaderModule(context.device, fragment_shader, nullptr);
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);
	return result == VK_SUCCESS;
}

} // namespace

bool initialize() {
	const auto mesh = createMesh();
	index_count = uint32_t(mesh.indices.size());
	auto& context = graphics::internal::context;
	if (!graphics::resources::uploadBuffer(mesh.vertices.data(), mesh.vertices.size() * sizeof(Vertex),
	                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer) ||
	    !graphics::resources::uploadBuffer(mesh.indices.data(), mesh.indices.size() * sizeof(uint32_t),
	                                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer)) {
		std::cerr << "Не удалось загрузить вершины и индексы пирамиды в GPU\n";
		shutdown();
		return false;
	}
	if (!graphics::resources::loadTexture(APP_TEXTURE_DIR "/checker.png", albedo_texture)) {
		std::cerr << "Не удалось создать текстуру альбедо\n";
		shutdown();
		return false;
	}

	if (!createUniformResources()) {
		std::cerr << "Не удалось создать uniform buffer и дескрипторы сцены\n";
		shutdown();
		return false;
	}
	const VkPipelineLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1, .pSetLayouts = &descriptor_set_layout,
	};
	if (vkCreatePipelineLayout(context.device, &layout_info, nullptr, &pipeline_layout) != VK_SUCCESS ||
	    !createPipeline()) {
		std::cerr << "Не удалось создать графический конвейер пирамиды\n";
		shutdown();
		return false;
	}
	return true;
}

void shutdown() {
	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);
	vkDestroyPipeline(context.device, pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
	vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);
	if (uniform_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, uniform_buffer, uniform_allocation);
	}
	graphics::resources::destroy(vertex_buffer);
	graphics::resources::destroy(index_buffer);
	graphics::resources::destroy(albedo_texture);
	pipeline = VK_NULL_HANDLE;
	pipeline_layout = VK_NULL_HANDLE;
	uniform_buffer = VK_NULL_HANDLE;
	uniform_allocation = VK_NULL_HANDLE;
	uniform_mapped_data = nullptr;
	descriptor_pool = VK_NULL_HANDLE;
	descriptor_set_layout = VK_NULL_HANDLE;
	descriptor_set = VK_NULL_HANDLE;
}

void update([[maybe_unused]] double time) {
	ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(410, 620), ImGuiCond_FirstUseEver);
	ImGui::Begin("Pyramid - Texture mapping");
	if (ImGui::CollapsingHeader("Camera", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::DragFloat3("Camera position", controls.camera_position, 0.05f);
		ImGui::SliderFloat("Pitch", &controls.camera_rotation[0], -180.0f, 180.0f, "%.1f deg");
		ImGui::SliderFloat("Yaw", &controls.camera_rotation[1], -180.0f, 180.0f, "%.1f deg");
		ImGui::SliderFloat("Roll", &controls.camera_rotation[2], -180.0f, 180.0f, "%.1f deg");
		ImGui::Combo("Projection", &controls.projection, "Perspective\0Orthographic\0");
		ImGui::SliderFloat("Field of view", &controls.fov, 10.0f, 120.0f, "%.1f deg", ImGuiSliderFlags_AlwaysClamp);
		ImGui::DragFloat("Near plane", &controls.near_plane, 0.01f, 0.01f, controls.far_plane - 0.01f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::DragFloat("Far plane", &controls.far_plane, 0.5f, controls.near_plane + 0.01f, 1000.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
	}
	if (ImGui::CollapsingHeader("Directional light", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::SliderFloat("Light pitch", &controls.light_rotation[0], -90.0f, 90.0f, "%.1f deg");
		ImGui::SliderFloat("Light yaw", &controls.light_rotation[1], -180.0f, 180.0f, "%.1f deg");
		ImGui::ColorEdit3("Light color", controls.light_color);
		ImGui::SliderFloat("Light intensity", &controls.light_intensity, 0.0f, 5.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
	}
	if (ImGui::CollapsingHeader("Ambient light", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::ColorEdit3("Ambient color", controls.ambient_color);
		ImGui::SliderFloat("Ambient intensity", &controls.ambient_intensity, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
	}
	if (ImGui::CollapsingHeader("Material", ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::TextUnformatted("Albedo: checker.png");
		ImGui::ColorEdit3("Specular color", controls.specular);
		ImGui::SliderFloat("Shininess", &controls.shininess, 1.0f, 256.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
	}
	if (ImGui::CollapsingHeader("Model")) {
		ImGui::SliderFloat("Rotation", &controls.rotation_degrees, 0.0f, 360.0f, "%.1f deg");
		ImGui::SliderFloat3("Scale", controls.scale, 0.1f, 3.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::SliderFloat3("Position", controls.translation, -3.0f, 3.0f, "%.2f");
	}
	if (ImGui::Button("Reset")) {
		controls = {};
	}
	ImGui::End();
}

bool render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;
	// prepare() уже дождался fence предыдущего кадра. При одном кадре в работе
	// GPU больше не читает этот UBO, и его можно безопасно обновить.
	const glm::mat4 model = calculateModel(
		glm::vec3(controls.scale[0], controls.scale[1], controls.scale[2]),
		glm::vec3(controls.translation[0], controls.translation[1], controls.translation[2]),
		controls.rotation_degrees);
	const glm::vec3 eye(controls.camera_position[0], controls.camera_position[1], controls.camera_position[2]);
	const UniformData uniforms = {
		.model = model,
		.view_projection = calculateViewProjection(eye,
			glm::vec3(controls.camera_rotation[0], controls.camera_rotation[1], controls.camera_rotation[2]),
			controls.fov, controls.near_plane, controls.far_plane, controls.projection == 1,
			float(context.swapchain_extent.width) / float(context.swapchain_extent.height)),
		// Обратная транспонированная матрица сохраняет нормали при неравномерном масштабе.
		.normal_matrix = glm::transpose(glm::inverse(model)),
		.camera_position = glm::vec4(eye, 1.0f),
		.light_direction_intensity = glm::vec4(lightDirection(), controls.light_intensity),
		.light_color = glm::vec4(controls.light_color[0], controls.light_color[1], controls.light_color[2], 0.0f),
		.ambient_color_intensity = glm::vec4(controls.ambient_color[0], controls.ambient_color[1], controls.ambient_color[2], controls.ambient_intensity),
		.specular_shininess = glm::vec4(controls.specular[0], controls.specular[1], controls.specular[2], controls.shininess),
	};
	std::memcpy(uniform_mapped_data, &uniforms, sizeof(uniforms));
	if (vmaFlushAllocation(context.allocator, uniform_allocation, 0, sizeof(uniforms)) != VK_SUCCESS) {
		std::cerr << "Не удалось обновить uniform buffer сцены\n";
		return false;
	}
	if (vkResetCommandBuffer(fd.command_buffer, 0) != VK_SUCCESS) {
		return false;
	}
	const VkCommandBufferBeginInfo begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};
	if (vkBeginCommandBuffer(fd.command_buffer, &begin) != VK_SUCCESS) {
		return false;
	}
	const VkClearValue clear_values[] = {
		{.color = {{0.025f, 0.035f, 0.06f, 1.0f}}},
		{.depthStencil = {1.0f, 0}},
	};
	const VkRenderPassBeginInfo render_pass = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass, .framebuffer = fd.framebuffer,
		.renderArea = {.extent = context.swapchain_extent},
		.clearValueCount = uint32_t(std::size(clear_values)), .pClearValues = clear_values,
	};
	vkCmdBeginRenderPass(fd.command_buffer, &render_pass, VK_SUBPASS_CONTENTS_INLINE);
	const VkViewport viewport = {
		.width = float(context.swapchain_extent.width),
		.height = float(context.swapchain_extent.height), .minDepth = 0.0f, .maxDepth = 1.0f,
	};
	const VkRect2D scissor = {.extent = context.swapchain_extent};
	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	const VkDeviceSize offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer.handle, &offset);
	vkCmdBindIndexBuffer(fd.command_buffer, index_buffer.handle, 0, VK_INDEX_TYPE_UINT32);
	const uint32_t model_index = 0;
	const uint32_t dynamic_offset = uint32_t(model_index * uniform_stride);
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        pipeline_layout, 0, 1, &descriptor_set, 1, &dynamic_offset);
	vkCmdDrawIndexed(fd.command_buffer, index_count, 1, 0, 0, 0);
	vkCmdEndRenderPass(fd.command_buffer);
	return vkEndCommandBuffer(fd.command_buffer) == VK_SUCCESS;
}

} // namespace application
