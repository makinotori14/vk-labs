#include "application.hpp"
#include "transforms.hpp"

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
	float position[3];
	float color[3];
};

// Отдельные вершины задают каждой грани сплошной цвет без интерполяции между гранями.
constexpr Vertex vertices[] = {
	// Передняя грань.
	{{-1, -1,  1}, {0.12f, 0.42f, 0.90f}},
	{{ 1, -1,  1}, {0.12f, 0.42f, 0.90f}},
	{{ 0,  1,  0}, {0.12f, 0.42f, 0.90f}},
	// Правая грань.
	{{ 1, -1,  1}, {0.25f, 0.62f, 1.00f}},
	{{ 1, -1, -1}, {0.25f, 0.62f, 1.00f}},
	{{ 0,  1,  0}, {0.25f, 0.62f, 1.00f}},
	// Задняя грань.
	{{ 1, -1, -1}, {0.08f, 0.25f, 0.65f}},
	{{-1, -1, -1}, {0.08f, 0.25f, 0.65f}},
	{{ 0,  1,  0}, {0.08f, 0.25f, 0.65f}},
	// Левая грань.
	{{-1, -1, -1}, {0.40f, 0.72f, 1.00f}},
	{{-1, -1,  1}, {0.40f, 0.72f, 1.00f}},
	{{ 0,  1,  0}, {0.40f, 0.72f, 1.00f}},
	// Основание (два треугольника).
	{{-1, -1, -1}, {0.04f, 0.14f, 0.40f}},
	{{ 1, -1, -1}, {0.04f, 0.14f, 0.40f}},
	{{ 1, -1,  1}, {0.04f, 0.14f, 0.40f}},
	{{-1, -1, -1}, {0.04f, 0.14f, 0.40f}},
	{{ 1, -1,  1}, {0.04f, 0.14f, 0.40f}},
	{{-1, -1,  1}, {0.04f, 0.14f, 0.40f}},
};

struct alignas(16) UniformData {
	glm::mat4 mvp;
};

// Единственная mat4 соответствует блоку std140 в вершинном шейдере.
static_assert(offsetof(UniformData, mvp) == 0);
static_assert(sizeof(UniformData) == 64);

struct Controls {
	int projection = 0; // 0 — перспективная проекция, 1 — ортографическая.
	float rotation_degrees = 0.0f;
	float scale[3] = {1.0f, 1.0f, 1.0f};
	float translation[3] = {0.0f, 0.0f, 0.0f};
};

VkBuffer vertex_buffer;
VmaAllocation vertex_allocation;
VkBuffer uniform_buffer;
VmaAllocation uniform_allocation;
void* uniform_mapped_data;
VkDescriptorSetLayout descriptor_set_layout;
VkDescriptorPool descriptor_pool;
VkDescriptorSet descriptor_set;
VkPipelineLayout pipeline_layout;
VkPipeline pipeline;
Controls controls;

bool createUniformResources() {
	auto& context = graphics::internal::context;
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(UniformData), .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
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

	const VkDescriptorSetLayoutBinding binding = {
		.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};
	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1, .pBindings = &binding,
	};
	if (vkCreateDescriptorSetLayout(context.device, &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
		return false;
	}
	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = 1,
	};
	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size,
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
	const VkWriteDescriptorSet write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = descriptor_set, .dstBinding = 0, .descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &descriptor_buffer,
	};
	vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
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
		{.location = 1, .binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = offsetof(Vertex, color)},
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
	auto& context = graphics::internal::context;
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(vertices), .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo allocation_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};
	VmaAllocationInfo allocation{};
	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_info,
	                    &vertex_buffer, &vertex_allocation, &allocation) != VK_SUCCESS) {
		std::cerr << "Не удалось создать буфер вершин пирамиды\n";
		return false;
	}
	std::memcpy(allocation.pMappedData, vertices, sizeof(vertices));
	if (vmaFlushAllocation(context.allocator, vertex_allocation, 0, sizeof(vertices)) != VK_SUCCESS) {
		shutdown();
		return false;
	}

	if (!createUniformResources()) {
		std::cerr << "Не удалось создать uniform buffer и дескрипторы MVP\n";
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
	vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_allocation);
	pipeline = VK_NULL_HANDLE;
	pipeline_layout = VK_NULL_HANDLE;
	vertex_buffer = VK_NULL_HANDLE;
	vertex_allocation = VK_NULL_HANDLE;
	uniform_buffer = VK_NULL_HANDLE;
	uniform_allocation = VK_NULL_HANDLE;
	uniform_mapped_data = nullptr;
	descriptor_pool = VK_NULL_HANDLE;
	descriptor_set_layout = VK_NULL_HANDLE;
	descriptor_set = VK_NULL_HANDLE;
}

void update([[maybe_unused]] double time) {
	ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowSize(ImVec2(380, 0), ImGuiCond_FirstUseEver);
	ImGui::Begin("Pyramid", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
	ImGui::Combo("Projection", &controls.projection, "Perspective\0Orthographic\0");
	ImGui::SliderFloat("Rotation", &controls.rotation_degrees, 0.0f, 360.0f, "%.1f deg", ImGuiSliderFlags_AlwaysClamp);

	ImGui::SeparatorText("Proportions");
	ImGui::SliderFloat("Width (X)", &controls.scale[0], 0.1f, 3.0f, "%.2fx", ImGuiSliderFlags_AlwaysClamp);
	ImGui::SliderFloat("Height (Y)", &controls.scale[1], 0.1f, 3.0f, "%.2fx", ImGuiSliderFlags_AlwaysClamp);
	ImGui::SliderFloat("Depth (Z)", &controls.scale[2], 0.1f, 3.0f, "%.2fx", ImGuiSliderFlags_AlwaysClamp);

	ImGui::SeparatorText("Position");
	ImGui::SliderFloat("X", &controls.translation[0], -3.0f, 3.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
	ImGui::SliderFloat("Y", &controls.translation[1], -3.0f, 3.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
	ImGui::SliderFloat("Z", &controls.translation[2], -3.0f, 3.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
	if (ImGui::Button("Reset")) {
		controls = {};
	}
	ImGui::End();
}

bool render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;
	// prepare() уже дождался fence предыдущего кадра. При одном кадре в работе
	// GPU больше не читает этот UBO, и его можно безопасно обновить.
	const UniformData uniforms = {
		.mvp = calculateMvp(
			glm::vec3(controls.scale[0], controls.scale[1], controls.scale[2]),
			glm::vec3(controls.translation[0], controls.translation[1], controls.translation[2]),
			controls.rotation_degrees, controls.projection == 1,
			float(context.swapchain_extent.width) / float(context.swapchain_extent.height)),
	};
	std::memcpy(uniform_mapped_data, &uniforms, sizeof(uniforms));
	if (vmaFlushAllocation(context.allocator, uniform_allocation, 0, sizeof(uniforms)) != VK_SUCCESS) {
		std::cerr << "Не удалось обновить uniform buffer MVP\n";
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
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1, &vertex_buffer, &offset);
	vkCmdBindDescriptorSets(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
	                        pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);
	vkCmdDraw(fd.command_buffer, uint32_t(std::size(vertices)), 1, 0, 0);
	vkCmdEndRenderPass(fd.command_buffer);
	return vkEndCommandBuffer(fd.command_buffer) == VK_SUCCESS;
}

} // namespace application
