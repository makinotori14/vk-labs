#include "application.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <imgui.h>
#include <miniaudio.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace application {

namespace {

struct SceneUniformData {
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 projection;
};

struct ObjectUniformData {
	alignas(16) glm::mat4 model;
	alignas(16) glm::vec4 tint;
};

struct Vertex {
	glm::vec3 position;
	glm::vec3 color;
};

enum class ProjectionType {
	Perspective,
	Orthographic,
};

struct RenderObject {
	glm::vec3 position = {0.0f, 0.0f, 0.0f};
	glm::vec3 rotation_degrees = {0.0f, 0.0f, 0.0f};
	glm::vec3 scale = {1.0f, 1.0f, 1.0f};
	glm::vec4 tint = {1.0f, 1.0f, 1.0f, 1.0f};

	VkBuffer uniform_buffer = VK_NULL_HANDLE;
	VmaAllocation uniform_buffer_allocation = VK_NULL_HANDLE;
	void* uniform_buffer_mapped_data = nullptr;
	VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
};

static_assert(sizeof(SceneUniformData) == 128);
static_assert(sizeof(ObjectUniformData) == 80);
static_assert(sizeof(Vertex) == sizeof(float) * 6);

std::array<glm::vec3, 5> face_colors = {{
	{1.0f, 1.0f, 1.0f},
	{1.0f, 1.0f, 1.0f},
	{1.0f, 1.0f, 1.0f},
	{1.0f, 1.0f, 1.0f},
	{1.0f, 1.0f, 1.0f},
}};

std::array<Vertex, 18> vertices = {{
	// Front face
	{{ 0.0f,  0.8f,  0.0f}, face_colors[0]},
	{{-0.7f, -0.6f, -0.7f}, face_colors[0]},
	{{ 0.7f, -0.6f, -0.7f}, face_colors[0]},

	// Right face
	{{ 0.0f,  0.8f,  0.0f}, face_colors[1]},
	{{ 0.7f, -0.6f, -0.7f}, face_colors[1]},
	{{ 0.7f, -0.6f,  0.7f}, face_colors[1]},

	// Back face
	{{ 0.0f,  0.8f,  0.0f}, face_colors[2]},
	{{ 0.7f, -0.6f,  0.7f}, face_colors[2]},
	{{-0.7f, -0.6f,  0.7f}, face_colors[2]},

	// Left face
	{{ 0.0f,  0.8f,  0.0f}, face_colors[3]},
	{{-0.7f, -0.6f,  0.7f}, face_colors[3]},
	{{-0.7f, -0.6f, -0.7f}, face_colors[3]},

	// Base (two triangles)
	{{-0.7f, -0.6f, -0.7f}, face_colors[4]},
	{{-0.7f, -0.6f,  0.7f}, face_colors[4]},
	{{ 0.7f, -0.6f,  0.7f}, face_colors[4]},

	{{-0.7f, -0.6f, -0.7f}, face_colors[4]},
	{{ 0.7f, -0.6f,  0.7f}, face_colors[4]},
	{{ 0.7f, -0.6f, -0.7f}, face_colors[4]},
}};

VkPipelineLayout pipeline_layout;
VkPipeline graphics_pipeline;

VkDescriptorSetLayout scene_descriptor_set_layout;
VkDescriptorSetLayout object_descriptor_set_layout;
VkDescriptorPool descriptor_pool;
VkDescriptorSet scene_descriptor_set;
VkBuffer scene_uniform_buffer;
VmaAllocation scene_uniform_buffer_allocation;
void* scene_uniform_buffer_mapped_data;

VkBuffer vertex_buffer;
VmaAllocation vertex_buffer_allocation;
void* vertex_buffer_mapped_data;
bool vertex_colors_dirty = false;
float rotation_angle = 0.0f;

constexpr size_t render_object_count = 3;
std::array<RenderObject, render_object_count> render_objects = {{
	{
		.position = {-1.4f, 0.0f, 0.0f},
		.scale = {0.65f, 0.65f, 0.65f},
		.tint = {1.0f, 0.65f, 0.65f, 1.0f},
	},
	{
		.position = {0.0f, 0.0f, 0.0f},
		.tint = {0.75f, 1.0f, 0.75f, 1.0f},
	},
	{
		.position = {1.4f, 0.0f, 0.0f},
		.scale = {0.65f, 0.65f, 0.65f},
		.tint = {0.7f, 0.8f, 1.0f, 1.0f},
	},
}};
int selected_object_index = 1;
size_t animated_object_index = 1;

bool jump_animation_active = false;
float jump_animation_time = 0.0f;
float jump_height_offset = 0.0f;
float somersault_angle = 0.0f;

float jump_duration = 2.0f;
float jump_height = 1.5f;

bool jump_on_pause = false;

ma_engine audio_engine{};
ma_sound jump_sound{};
ma_sound pause_sound{};
bool audio_engine_initialized = false;
bool jump_sound_initialized = false;
bool pause_sound_initialized = false;

GLFWwindow* application_window;

const glm::vec3 default_camera_position = { 0.0f, 0.1f, 3.0f };
const glm::vec3 default_camera_forward = { 0.0f, 0.0f, -1.0f };
const float default_camera_yaw = glm::radians(-90.0f);
const float default_camera_pitch = 0.0f;

glm::vec3 camera_position = default_camera_position;
glm::vec3 camera_forward = default_camera_forward;
const glm::vec3 world_up = { 0.0f, 1.0f, 0.0f };

float camera_yaw = default_camera_yaw;
float camera_pitch = default_camera_pitch;
ProjectionType projection_type = ProjectionType::Perspective;

float perspective_fov_degrees = 70.0f;
constexpr float orthographic_half_height = 1.5f;
constexpr float near_plane = 0.1f;
constexpr float far_plane = 10.0f;

bool pyramid_rotates = false;

double previous_frame_time = -1.0;

bool isKeyPressed(int key) {
	return glfwGetKey(application_window, key) == GLFW_PRESS;
}

void resetCamera() {
	camera_position = default_camera_position;
	camera_forward = default_camera_forward;
	camera_yaw = default_camera_yaw;
	camera_pitch = default_camera_pitch;
}

void shutdownAudio() {
	if (pause_sound_initialized) {
		ma_sound_uninit(&pause_sound);
		pause_sound_initialized = false;
	}
	if (jump_sound_initialized) {
		ma_sound_uninit(&jump_sound);
		jump_sound_initialized = false;
	}
	if (audio_engine_initialized) {
		ma_engine_uninit(&audio_engine);
		audio_engine_initialized = false;
	}
}

bool initializeAudio() {
	if (ma_engine_init(nullptr, &audio_engine) != MA_SUCCESS) {
		std::cerr << "Failed to initialize miniaudio engine\n";
		return false;
	}
	audio_engine_initialized = true;

	constexpr ma_uint32 sound_flags =
		MA_SOUND_FLAG_STREAM | MA_SOUND_FLAG_NO_SPATIALIZATION;
	if (ma_sound_init_from_file(
			&audio_engine, "audio/jump.mp3", sound_flags,
			nullptr, nullptr, &jump_sound) != MA_SUCCESS) {
		std::cerr << "Failed to load audio/jump.mp3\n";
		shutdownAudio();
		return false;
	}
	jump_sound_initialized = true;

	if (ma_sound_init_from_file(
			&audio_engine, "audio/pause.mp3", sound_flags,
			nullptr, nullptr, &pause_sound) != MA_SUCCESS) {
		std::cerr << "Failed to load audio/pause.mp3\n";
		shutdownAudio();
		return false;
	}
	pause_sound_initialized = true;

	return true;
}

void stopAndRewind(ma_sound& sound) {
	ma_sound_stop(&sound);
	ma_sound_seek_to_pcm_frame(&sound, 0);
}

void startJumpAudio() {
	if (!jump_sound_initialized || !pause_sound_initialized) {
		return;
	}

	stopAndRewind(pause_sound);
	stopAndRewind(jump_sound);
	ma_sound_start(&jump_sound);
}

void pauseJumpAudio() {
	if (!jump_sound_initialized || !pause_sound_initialized) {
		return;
	}

	// ma_sound_stop() preserves the playback cursor, so jump.mp3 can resume.
	ma_sound_stop(&jump_sound);
	stopAndRewind(pause_sound);
	ma_sound_start(&pause_sound);
}

void resumeJumpAudio() {
	if (!jump_sound_initialized || !pause_sound_initialized) {
		return;
	}

	stopAndRewind(pause_sound);
	ma_sound_start(&jump_sound);
}

void finishJumpAudio() {
	if (jump_sound_initialized) {
		stopAndRewind(jump_sound);
	}
	if (pause_sound_initialized) {
		stopAndRewind(pause_sound);
	}
}

VkShaderModule loadShaderModule(const char* filename) {
	std::ifstream file(filename, std::ios::ate | std::ios::binary);
	if (!file.is_open()) {
		std::cerr << "Failed to open shader file: " << filename << '\n';
		return VK_NULL_HANDLE;
	}

	const std::streamsize file_size = file.tellg();
	if (file_size <= 0 || file_size % sizeof(uint32_t) != 0) {
		std::cerr << "Invalid SPIR-V shader file: " << filename << '\n';
		return VK_NULL_HANDLE;
	}

	std::vector<uint32_t> code(size_t(file_size) / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(code.data()), file_size);
	if (!file) {
		std::cerr << "Failed to read shader file: " << filename << '\n';
		return VK_NULL_HANDLE;
	}

	const VkShaderModuleCreateInfo shader_module_info = {
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size_t(file_size),
		.pCode = code.data(),
	};

	VkShaderModule shader_module = VK_NULL_HANDLE;
	if (vkCreateShaderModule(graphics::internal::context.device, &shader_module_info,
	                         nullptr, &shader_module) != VK_SUCCESS) {
		std::cerr << "Failed to create shader module: " << filename << '\n';
		return VK_NULL_HANDLE;
	}

	return shader_module;
}

glm::vec3 proceduralColor(const glm::vec3& position) {
	const glm::vec3 min_position = {-0.7f, -0.6f, -0.7f};
	const glm::vec3 max_position = { 0.7f,  0.8f,  0.7f};

	return glm::clamp(
		(position - min_position) / (max_position - min_position),
		0.0f, 1.0f);
}

void applyFaceColorsToVertices() {
	for (size_t vertex_index = 0; vertex_index < vertices.size(); ++vertex_index) {
		const size_t face_index = vertex_index < 12
			? vertex_index / 3
			: 4;

		const glm::vec3 base_color = proceduralColor(
			vertices[vertex_index].position);
		vertices[vertex_index].color = base_color * face_colors[face_index];
	}
}

bool uploadVertices() {
	if (vertex_buffer_mapped_data == nullptr) {
		std::cerr << "Vulkan vertex buffer is not mapped\n";
		return false;
	}

	std::memcpy(vertex_buffer_mapped_data, vertices.data(), sizeof(vertices));
	if (vmaFlushAllocation(graphics::internal::context.allocator,
	                       vertex_buffer_allocation,
	                       0, sizeof(vertices)) != VK_SUCCESS) {
		std::cerr << "Failed to flush Vulkan vertex buffer memory\n";
		return false;
	}

	return true;
}

bool createUniformDescriptorSetLayout(VkDescriptorSetLayout& layout) {
	const VkDescriptorSetLayoutBinding uniform_binding = {
		.binding = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
	};

	const VkDescriptorSetLayoutCreateInfo layout_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.bindingCount = 1,
		.pBindings = &uniform_binding,
	};

	return vkCreateDescriptorSetLayout(
		graphics::internal::context.device,
		&layout_info, nullptr, &layout) == VK_SUCCESS;
}

bool createMappedUniformBuffer(VkDeviceSize size,
	                            VkBuffer& buffer,
	                            VmaAllocation& allocation,
	                            void*& mapped_data) {
	auto& context = graphics::internal::context;

	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size,
		.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_create_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo allocation_info = {};
	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_create_info,
	                    &buffer, &allocation, &allocation_info) != VK_SUCCESS ||
	    allocation_info.pMappedData == nullptr) {
		if (buffer != VK_NULL_HANDLE) {
			vmaDestroyBuffer(context.allocator, buffer, allocation);
			buffer = VK_NULL_HANDLE;
			allocation = VK_NULL_HANDLE;
		}
		return false;
	}

	mapped_data = allocation_info.pMappedData;
	return true;
}

void writeUniformDescriptor(VkDescriptorSet set,
	                        VkBuffer buffer,
	                        VkDeviceSize size) {
	const VkDescriptorBufferInfo buffer_info = {
		.buffer = buffer,
		.offset = 0,
		.range = size,
	};

	const VkWriteDescriptorSet descriptor_write = {
		.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = set,
		.dstBinding = 0,
		.descriptorCount = 1,
		.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.pBufferInfo = &buffer_info,
	};

	vkUpdateDescriptorSets(
		graphics::internal::context.device,
		1, &descriptor_write, 0, nullptr);
}

bool uploadUniformBuffer(VmaAllocation allocation,
	                     void* mapped_data,
	                     const void* data,
	                     size_t size) {
	if (mapped_data == nullptr) {
		std::cerr << "Vulkan uniform buffer is not mapped\n";
		return false;
	}

	std::memcpy(mapped_data, data, size);
	if (vmaFlushAllocation(graphics::internal::context.allocator,
	                       allocation, 0, size) != VK_SUCCESS) {
		std::cerr << "Failed to flush Vulkan uniform buffer memory\n";
		return false;
	}

	return true;
}

void destroyDescriptorResources() {
	auto& context = graphics::internal::context;

	if (descriptor_pool != VK_NULL_HANDLE) {
		vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
		descriptor_pool = VK_NULL_HANDLE;
		scene_descriptor_set = VK_NULL_HANDLE;
		for (RenderObject& object : render_objects) {
			object.descriptor_set = VK_NULL_HANDLE;
		}
	}

	if (scene_uniform_buffer != VK_NULL_HANDLE) {
		vmaDestroyBuffer(context.allocator, scene_uniform_buffer,
		                 scene_uniform_buffer_allocation);
		scene_uniform_buffer = VK_NULL_HANDLE;
		scene_uniform_buffer_allocation = VK_NULL_HANDLE;
		scene_uniform_buffer_mapped_data = nullptr;
	}

	for (RenderObject& object : render_objects) {
		if (object.uniform_buffer != VK_NULL_HANDLE) {
			vmaDestroyBuffer(context.allocator, object.uniform_buffer,
			                 object.uniform_buffer_allocation);
			object.uniform_buffer = VK_NULL_HANDLE;
			object.uniform_buffer_allocation = VK_NULL_HANDLE;
			object.uniform_buffer_mapped_data = nullptr;
		}
	}

	if (object_descriptor_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(
			context.device, object_descriptor_set_layout, nullptr);
		object_descriptor_set_layout = VK_NULL_HANDLE;
	}
	if (scene_descriptor_set_layout != VK_NULL_HANDLE) {
		vkDestroyDescriptorSetLayout(
			context.device, scene_descriptor_set_layout, nullptr);
		scene_descriptor_set_layout = VK_NULL_HANDLE;
	}
}

bool createDescriptorResources() {
	auto& context = graphics::internal::context;

	if (!createUniformDescriptorSetLayout(scene_descriptor_set_layout) ||
	    !createUniformDescriptorSetLayout(object_descriptor_set_layout)) {
		std::cerr << "Failed to create Vulkan descriptor set layouts\n";
		destroyDescriptorResources();
		return false;
	}

	if (!createMappedUniformBuffer(
			sizeof(SceneUniformData), scene_uniform_buffer,
			scene_uniform_buffer_allocation, scene_uniform_buffer_mapped_data)) {
		std::cerr << "Failed to create Vulkan scene uniform buffer\n";
		destroyDescriptorResources();
		return false;
	}

	for (RenderObject& object : render_objects) {
		if (!createMappedUniformBuffer(
				sizeof(ObjectUniformData), object.uniform_buffer,
				object.uniform_buffer_allocation,
				object.uniform_buffer_mapped_data)) {
			std::cerr << "Failed to create Vulkan object uniform buffer\n";
			destroyDescriptorResources();
			return false;
		}
	}

	const uint32_t descriptor_count = uint32_t(render_objects.size()) + 1;
	const VkDescriptorPoolSize pool_size = {
		.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
		.descriptorCount = descriptor_count,
	};

	const VkDescriptorPoolCreateInfo pool_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.maxSets = descriptor_count,
		.poolSizeCount = 1,
		.pPoolSizes = &pool_size,
	};

	if (vkCreateDescriptorPool(context.device, &pool_info, nullptr,
	                           &descriptor_pool) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan descriptor pool\n";
		destroyDescriptorResources();
		return false;
	}

	const VkDescriptorSetAllocateInfo scene_allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = 1,
		.pSetLayouts = &scene_descriptor_set_layout,
	};

	if (vkAllocateDescriptorSets(context.device, &scene_allocate_info,
	                             &scene_descriptor_set) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan scene descriptor set\n";
		destroyDescriptorResources();
		return false;
	}

	std::array<VkDescriptorSetLayout, render_object_count> object_layouts;
	object_layouts.fill(object_descriptor_set_layout);
	std::array<VkDescriptorSet, render_object_count> object_sets;

	const VkDescriptorSetAllocateInfo object_allocate_info = {
		.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = descriptor_pool,
		.descriptorSetCount = uint32_t(object_sets.size()),
		.pSetLayouts = object_layouts.data(),
	};

	if (vkAllocateDescriptorSets(context.device, &object_allocate_info,
	                             object_sets.data()) != VK_SUCCESS) {
		std::cerr << "Failed to allocate Vulkan object descriptor sets\n";
		destroyDescriptorResources();
		return false;
	}

	writeUniformDescriptor(
		scene_descriptor_set, scene_uniform_buffer, sizeof(SceneUniformData));
	for (size_t i = 0; i < render_objects.size(); ++i) {
		render_objects[i].descriptor_set = object_sets[i];
		writeUniformDescriptor(
			render_objects[i].descriptor_set,
			render_objects[i].uniform_buffer,
			sizeof(ObjectUniformData));
	}

	return true;
}

bool createVertexBuffer() {
	auto& context = graphics::internal::context;

	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(vertices),
		.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};

	const VmaAllocationCreateInfo allocation_create_info = {
		.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
		         VMA_ALLOCATION_CREATE_MAPPED_BIT,
		.usage = VMA_MEMORY_USAGE_AUTO,
	};

	VmaAllocationInfo allocation_info;
	if (vmaCreateBuffer(context.allocator, &buffer_info, &allocation_create_info,
	                    &vertex_buffer, &vertex_buffer_allocation,
	                    &allocation_info) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan vertex buffer\n";
		return false;
	}

	if (allocation_info.pMappedData == nullptr) {
		std::cerr << "Failed to map Vulkan vertex buffer\n";
		vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_buffer_allocation);
		vertex_buffer = VK_NULL_HANDLE;
		vertex_buffer_allocation = VK_NULL_HANDLE;
		return false;
	}

	applyFaceColorsToVertices();
	vertex_buffer_mapped_data = allocation_info.pMappedData;
	if (!uploadVertices()) {
		vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_buffer_allocation);
		vertex_buffer = VK_NULL_HANDLE;
		vertex_buffer_allocation = VK_NULL_HANDLE;
		vertex_buffer_mapped_data = nullptr;
		return false;
	}

	return true;
}

} // namespace

bool initialize(GLFWwindow* window) {
	application_window = window;

	auto& context = graphics::internal::context;

	const VkShaderModule vertex_shader = loadShaderModule("shaders/triangle.vert.spv");
	if (vertex_shader == VK_NULL_HANDLE) {
		return false;
	}

	const VkShaderModule fragment_shader = loadShaderModule("shaders/triangle.frag.spv");
	if (fragment_shader == VK_NULL_HANDLE) {
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		return false;
	}

	if (!createDescriptorResources()) {
		vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		return false;
	}

	const VkPipelineShaderStageCreateInfo shader_stages[] = {
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader,
			.pName = "main",
		},
		{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader,
			.pName = "main",
		},
	};

	const VkVertexInputBindingDescription vertex_binding = {
		.binding = 0,
		.stride = sizeof(Vertex),
		.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
	};

	const VkVertexInputAttributeDescription vertex_attributes[] = {
		{
			.location = 0,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = uint32_t(offsetof(Vertex, position)),
		},
		{
			.location = 1,
			.binding = 0,
			.format = VK_FORMAT_R32G32B32_SFLOAT,
			.offset = uint32_t(offsetof(Vertex, color)),
		},
	};

	const VkPipelineVertexInputStateCreateInfo vertex_input = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
		.vertexBindingDescriptionCount = 1,
		.pVertexBindingDescriptions = &vertex_binding,
		.vertexAttributeDescriptionCount = uint32_t(
			sizeof(vertex_attributes) / sizeof(vertex_attributes[0])),
		.pVertexAttributeDescriptions = vertex_attributes,
	};

	const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
		.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
	};

	const VkPipelineViewportStateCreateInfo viewport_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
		.viewportCount = 1,
		.scissorCount = 1,
	};

	const VkPipelineRasterizationStateCreateInfo rasterization = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
		.polygonMode = VK_POLYGON_MODE_FILL,
		.cullMode = VK_CULL_MODE_NONE,
		.frontFace = VK_FRONT_FACE_CLOCKWISE,
		.lineWidth = 1.0f,
	};

	const VkPipelineMultisampleStateCreateInfo multisampling = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
		.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
	};

	const VkPipelineDepthStencilStateCreateInfo depth_stencil = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
		.depthTestEnable = VK_TRUE,
		.depthWriteEnable = VK_TRUE,
		.depthCompareOp = VK_COMPARE_OP_LESS,
	};

	const VkPipelineColorBlendAttachmentState color_blend_attachment = {
		.blendEnable = VK_FALSE,

		.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
		                  VK_COLOR_COMPONENT_G_BIT |
		                  VK_COLOR_COMPONENT_B_BIT |
		                  VK_COLOR_COMPONENT_A_BIT,
	};

	const VkPipelineColorBlendStateCreateInfo color_blending = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
		.attachmentCount = 1,
		.pAttachments = &color_blend_attachment,
	};

	const VkDynamicState dynamic_states[] = {
		VK_DYNAMIC_STATE_VIEWPORT,
		VK_DYNAMIC_STATE_SCISSOR,
	};

	const VkPipelineDynamicStateCreateInfo dynamic_state = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
		.dynamicStateCount = uint32_t(sizeof(dynamic_states) / sizeof(dynamic_states[0])),
		.pDynamicStates = dynamic_states,
	};

	const VkDescriptorSetLayout descriptor_set_layouts[] = {
		scene_descriptor_set_layout,
		object_descriptor_set_layout,
	};

	const VkPipelineLayoutCreateInfo pipeline_layout_info = {
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = uint32_t(
			sizeof(descriptor_set_layouts) / sizeof(descriptor_set_layouts[0])),
		.pSetLayouts = descriptor_set_layouts,
	};

	if (vkCreatePipelineLayout(context.device, &pipeline_layout_info, nullptr,
	                           &pipeline_layout) != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan pipeline layout\n";
		vkDestroyShaderModule(context.device, fragment_shader, nullptr);
		vkDestroyShaderModule(context.device, vertex_shader, nullptr);
		destroyDescriptorResources();
		return false;
	}

	const VkGraphicsPipelineCreateInfo pipeline_info = {
		.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
		.stageCount = uint32_t(sizeof(shader_stages) / sizeof(shader_stages[0])),
		.pStages = shader_stages,
		.pVertexInputState = &vertex_input,
		.pInputAssemblyState = &input_assembly,
		.pViewportState = &viewport_state,
		.pRasterizationState = &rasterization,
		.pMultisampleState = &multisampling,
		.pDepthStencilState = &depth_stencil,
		.pColorBlendState = &color_blending,
		.pDynamicState = &dynamic_state,
		.layout = pipeline_layout,
		.renderPass = context.render_pass,
		.subpass = 0,
	};

	const VkResult pipeline_result = vkCreateGraphicsPipelines(
		context.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &graphics_pipeline);

	vkDestroyShaderModule(context.device, fragment_shader, nullptr);
	vkDestroyShaderModule(context.device, vertex_shader, nullptr);

	if (pipeline_result != VK_SUCCESS) {
		std::cerr << "Failed to create Vulkan graphics pipeline\n";
		vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
		pipeline_layout = VK_NULL_HANDLE;
		destroyDescriptorResources();
		return false;
	}

	if (!createVertexBuffer()) {
		vkDestroyPipeline(context.device, graphics_pipeline, nullptr);
		vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
		graphics_pipeline = VK_NULL_HANDLE;
		pipeline_layout = VK_NULL_HANDLE;
		destroyDescriptorResources();
		return false;
	}

	if (!initializeAudio()) {
		std::cerr << "Audio is disabled, graphics will continue to work\n";
	}

	return true;
}

void shutdown() {
	shutdownAudio();

	auto& context = graphics::internal::context;
	vkQueueWaitIdle(context.graphics_queue);

	vmaDestroyBuffer(context.allocator, vertex_buffer, vertex_buffer_allocation);
	vkDestroyPipeline(context.device, graphics_pipeline, nullptr);
	vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
	destroyDescriptorResources();

	vertex_buffer_mapped_data = nullptr;
	application_window = nullptr;
}
void drawOverlay(const char* text) {
    const ImGuiIO& io = ImGui::GetIO();
    const ImVec2 text_size = ImGui::CalcTextSize(text);

    const ImVec2 position = {
        (io.DisplaySize.x - text_size.x) * 0.5f,
        20.0f
    };

    ImGui::GetForegroundDrawList()->AddText(
        position,
        IM_COL32(255, 255, 255, 255),
        text
    );
}
void drawInterface() {
    ImGui::SetNextWindowSize(
        ImVec2(320.0f, 180.0f),
        ImGuiCond_FirstUseEver
    );

	ImGui::Begin("Menu");

	if (ImGui::Button("Reset cam pos")) {
		resetCamera();
	}

	ImGui::SeparatorText("Projection");

	if (ImGui::RadioButton(
			"Perspective", projection_type == ProjectionType::Perspective)) {
		projection_type = ProjectionType::Perspective;
	}
	ImGui::SameLine();
	if (ImGui::RadioButton(
			"Orthographic", projection_type == ProjectionType::Orthographic)) {
		projection_type = ProjectionType::Orthographic;
	}

	const char* object_names[] = {"Left", "Center", "Right"};
	ImGui::Combo(
		"Object", &selected_object_index,
		object_names, int(sizeof(object_names) / sizeof(object_names[0])));
	RenderObject& selected_object = render_objects[size_t(selected_object_index)];

	ImGui::SeparatorText("Selected object");
	ImGui::DragFloat3("Position", &selected_object.position.x, 0.01f);
	ImGui::DragFloat3(
		"Rotation", &selected_object.rotation_degrees.x,
		1.0f, -180.0f, 180.0f, "%.1f deg");
	ImGui::DragFloat3(
		"Scale", &selected_object.scale.x,
		0.01f, 0.01f, 5.0f);
	ImGui::ColorEdit3("Object tint", &selected_object.tint.x);

	ImGui::SeparatorText("Face color tints");
	bool colors_changed = false;
	colors_changed |= ImGui::ColorEdit3("Front", &face_colors[0].x);
	colors_changed |= ImGui::ColorEdit3("Right", &face_colors[1].x);
	colors_changed |= ImGui::ColorEdit3("Back", &face_colors[2].x);
	colors_changed |= ImGui::ColorEdit3("Left", &face_colors[3].x);
	colors_changed |= ImGui::ColorEdit3("Base", &face_colors[4].x);

	if (colors_changed) {
		applyFaceColorsToVertices();
		vertex_colors_dirty = true;
	}

	if (ImGui::Button("Rotation")) {
		pyramid_rotates ^= 1;
	}
	ImGui::DragFloat("FOV", &perspective_fov_degrees, 1.0f, 10.0f, 170.0f, "%.1f deg");

	ImGui::SeparatorText("Animation");
	if (ImGui::Button(
			jump_animation_active ? "Jumping..." : "Jump and flip") &&
		!jump_animation_active) {
		jump_animation_active = true;
		jump_on_pause = false;
		jump_animation_time = 0.0f;
		animated_object_index = size_t(selected_object_index);
		startJumpAudio();
	}

	if (jump_animation_active) {
		if (ImGui::Button("Pause/Resume")) {
			jump_on_pause ^= 1;
			if (jump_on_pause) {
				pauseJumpAudio();
			} else {
				resumeJumpAudio();
			}
		}
	}

	if (!jump_animation_active || (jump_animation_active && jump_on_pause)) {
		ImGui::DragFloat("Jump Height", &jump_height, .1f, 1.5f, 10.0f, "%.3f");
		ImGui::DragFloat("Duration", &jump_duration, .1f, 2.0f, 10.0f, "%.3fs");
	}

	ImGui::End();
}

void update(double time) {
	float delta_time = 0.0f;
	if (previous_frame_time >= 0.0) {
		delta_time = static_cast<float>(time - previous_frame_time);
	}
	previous_frame_time = time;
	delta_time = glm::min(delta_time, 0.1f);

	if (!ImGui::GetIO().WantCaptureKeyboard) {
		const float rotation_speed = glm::radians(90.0f);

		if (isKeyPressed(GLFW_KEY_H)) {
			camera_yaw -= rotation_speed * delta_time;
		}
		if (isKeyPressed(GLFW_KEY_L)) {
			camera_yaw += rotation_speed * delta_time;
		}
		if (isKeyPressed(GLFW_KEY_J)) {
			camera_pitch -= rotation_speed * delta_time;
		}
		if (isKeyPressed(GLFW_KEY_K)) {
			camera_pitch += rotation_speed * delta_time;
		}

		camera_pitch = glm::clamp(
			camera_pitch, glm::radians(-89.0f), glm::radians(89.0f));

		const glm::vec3 direction = {
			glm::cos(camera_pitch) * glm::cos(camera_yaw),
			glm::sin(camera_pitch),
			glm::cos(camera_pitch) * glm::sin(camera_yaw),
		};
		camera_forward = glm::normalize(direction);

		const glm::vec3 movement_forward = glm::normalize(glm::vec3(
			camera_forward.x, 0.0f, camera_forward.z));
		const glm::vec3 camera_right = glm::normalize(
			glm::cross(movement_forward, world_up));

		const float movement_speed = 2.0f;
		const float movement_distance = movement_speed * delta_time;

		if (isKeyPressed(GLFW_KEY_W)) {
			camera_position += movement_forward * movement_distance;
		}
		if (isKeyPressed(GLFW_KEY_S)) {
			camera_position -= movement_forward * movement_distance;
		}
		if (isKeyPressed(GLFW_KEY_A)) {
			camera_position -= camera_right * movement_distance;
		}
		if (isKeyPressed(GLFW_KEY_D)) {
			camera_position += camera_right * movement_distance;
		}
		if (isKeyPressed(GLFW_KEY_SPACE)) {
			camera_position += world_up * movement_distance;
		}
		if (isKeyPressed(GLFW_KEY_LEFT_SHIFT)) {
			camera_position -= world_up * movement_distance;
		}
		if (isKeyPressed(GLFW_KEY_U)) {
			resetCamera();
		}
	}

	if (pyramid_rotates) {
		rotation_angle += delta_time * 0.8f;
	}

	if (jump_animation_active && jump_on_pause) {
		drawOverlay("SWAG is on Pause");
	}

	if (jump_animation_active && !jump_on_pause) {
		drawOverlay("SWAG");
		jump_animation_time += delta_time;

		const float t = glm::clamp(
			jump_animation_time / jump_duration, 0.0f, 1.0f);

		jump_height_offset = 4.0f * jump_height * t * (1.0f - t);

		const float smooth_t = t * t * (3.0f - 2.0f * t);
		somersault_angle = glm::radians(360.0f) * smooth_t;

		if (t >= 1.0f) {
			jump_animation_active = false;
			jump_animation_time = 0.0f;
			jump_height_offset = 0.0f;
			somersault_angle = 0.0f;
			finishJumpAudio();
		}
	}

	drawInterface();
}

void render(const graphics::internal::FrameData& fd) {
	auto& context = graphics::internal::context;
	if (vertex_colors_dirty && uploadVertices()) {
		vertex_colors_dirty = false;
	}

	if (vkResetCommandBuffer(fd.command_buffer, 0) != VK_SUCCESS) {
		std::cerr << "Failed to reset Vulkan command buffer\n";
		return;
	}

	const VkCommandBufferBeginInfo command_buffer_begin = {
		.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
		.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
	};

	if (vkBeginCommandBuffer(fd.command_buffer, &command_buffer_begin) != VK_SUCCESS) {
		std::cerr << "Failed to begin Vulkan command buffer\n";
		return;
	}

	const VkClearValue clear_values[] = {
		{ .color = {{ 0.01f, 0.3f, 0.35f, 1.0f }} },
		{ .depthStencil = { 1.0f, 0 } },
	};

	const VkRenderPassBeginInfo render_pass_begin = {
		.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
		.renderPass = context.render_pass,
		.framebuffer = fd.framebuffer,
		.renderArea = { .extent = context.swapchain_extent },
		.clearValueCount = uint32_t(sizeof(clear_values) / sizeof(clear_values[0])),
		.pClearValues = clear_values,
	};

	vkCmdBeginRenderPass(fd.command_buffer, &render_pass_begin, VK_SUBPASS_CONTENTS_INLINE);

	const VkViewport viewport = {
		.x = 0.0f,
		.y = 0.0f,
		.width = float(context.swapchain_extent.width),
		.height = float(context.swapchain_extent.height),
		.minDepth = 0.0f,
		.maxDepth = 1.0f,
	};

	const VkRect2D scissor = {
		.extent = context.swapchain_extent,
	};

	vkCmdSetViewport(fd.command_buffer, 0, 1, &viewport);
	vkCmdSetScissor(fd.command_buffer, 0, 1, &scissor);
	vkCmdBindPipeline(fd.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

	const VkDeviceSize vertex_buffer_offset = 0;
	vkCmdBindVertexBuffers(fd.command_buffer, 0, 1,
	                       &vertex_buffer, &vertex_buffer_offset);

	const glm::mat4 view = glm::lookAt(
		camera_position,
		camera_position + camera_forward,
		world_up);

	const float aspect = float(context.swapchain_extent.width) /
	                     float(context.swapchain_extent.height);

	glm::mat4 projection;
	if (projection_type == ProjectionType::Perspective) {
		projection = glm::perspective(
			glm::radians(perspective_fov_degrees), aspect, near_plane, far_plane);
	} else {
		const float orthographic_half_width = orthographic_half_height * aspect;
		projection = glm::ortho(
			-orthographic_half_width,
			 orthographic_half_width,
			-orthographic_half_height,
			 orthographic_half_height,
			 near_plane,
			 far_plane);
	}
	projection[1][1] *= -1.0f;

	const SceneUniformData scene_uniform_data = {
		.view = view,
		.projection = projection,
	};
	uploadUniformBuffer(
		scene_uniform_buffer_allocation,
		scene_uniform_buffer_mapped_data,
		&scene_uniform_data,
		sizeof(scene_uniform_data));

	vkCmdBindDescriptorSets(
		fd.command_buffer,
		VK_PIPELINE_BIND_POINT_GRAPHICS,
		pipeline_layout,
		0, 1, &scene_descriptor_set,
		0, nullptr);

	for (size_t object_index = 0;
	     object_index < render_objects.size();
	     ++object_index) {
		RenderObject& object = render_objects[object_index];
		glm::vec3 position = object.position;
		float object_somersault_angle = 0.0f;
		if (object_index == animated_object_index) {
			position.y += jump_height_offset;
			object_somersault_angle = somersault_angle;
		}

		glm::mat4 model = glm::translate(glm::mat4(1.0f), position);
		model = glm::rotate(
			model,
			glm::radians(object.rotation_degrees.x) + object_somersault_angle,
			glm::vec3(1.0f, 0.0f, 0.0f));
		model = glm::rotate(
			model,
			glm::radians(object.rotation_degrees.y) + rotation_angle,
			glm::vec3(0.0f, 1.0f, 0.0f));
		model = glm::rotate(
			model,
			glm::radians(object.rotation_degrees.z),
			glm::vec3(0.0f, 0.0f, 1.0f));
		model = glm::scale(model, object.scale);

		const ObjectUniformData object_uniform_data = {
			.model = model,
			.tint = object.tint,
		};
		uploadUniformBuffer(
			object.uniform_buffer_allocation,
			object.uniform_buffer_mapped_data,
			&object_uniform_data,
			sizeof(object_uniform_data));

		vkCmdBindDescriptorSets(
			fd.command_buffer,
			VK_PIPELINE_BIND_POINT_GRAPHICS,
			pipeline_layout,
			1, 1, &object.descriptor_set,
			0, nullptr);
		vkCmdDraw(fd.command_buffer, uint32_t(vertices.size()), 1, 0, 0);
	}

	vkCmdEndRenderPass(fd.command_buffer);

	if (vkEndCommandBuffer(fd.command_buffer) != VK_SUCCESS) {
		std::cerr << "Failed to record Vulkan command buffer\n";
	}
}

} // namespace application
