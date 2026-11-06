#include "gpu_resources.hpp"

#include <cstring>
#include <iostream>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#include <stb_image.h>

namespace graphics::resources {

namespace {

// Короткоживущий командный пул используется только при начальной загрузке.
struct UploadCommands {
	VkCommandPool pool = VK_NULL_HANDLE;
	VkCommandBuffer buffer = VK_NULL_HANDLE;

	~UploadCommands() {
		vkDestroyCommandPool(internal::context.device, pool, nullptr);
	}

	bool begin() {
		auto& context = internal::context;
		const VkCommandPoolCreateInfo pool_info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
			.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
			.queueFamilyIndex = context.graphics_queue_index,
		};
		if (vkCreateCommandPool(context.device, &pool_info, nullptr, &pool) != VK_SUCCESS) {
			return false;
		}
		const VkCommandBufferAllocateInfo allocate_info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
			.commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
		};
		if (vkAllocateCommandBuffers(context.device, &allocate_info, &buffer) != VK_SUCCESS) {
			return false;
		}
		const VkCommandBufferBeginInfo begin_info = {
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};
		return vkBeginCommandBuffer(buffer, &begin_info) == VK_SUCCESS;
	}

	bool finish() {
		if (vkEndCommandBuffer(buffer) != VK_SUCCESS) {
			return false;
		}
		const VkSubmitInfo submit = {
			.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
			.commandBufferCount = 1, .pCommandBuffers = &buffer,
		};
		auto queue = internal::context.graphics_queue;
		if (vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS) {
			return false;
		}
		// После завершения копирования staging-память и командный пул можно освободить.
		return vkQueueWaitIdle(queue) == VK_SUCCESS;
	}
};

struct StagingBuffer {
	Buffer buffer;
	~StagingBuffer() { destroy(buffer); }

	bool initialize(const void* data, VkDeviceSize size) {
		const VkBufferCreateInfo buffer_info = {
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = size, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
			.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		};
		const VmaAllocationCreateInfo allocation_info = {
			.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
			.usage = VMA_MEMORY_USAGE_AUTO,
		};
		auto allocator = internal::context.allocator;
		VmaAllocationInfo allocation{};
		if (vmaCreateBuffer(allocator, &buffer_info, &allocation_info,
		                    &buffer.handle, &buffer.allocation, &allocation) != VK_SUCCESS) {
			return false;
		}
		std::memcpy(allocation.pMappedData, data, static_cast<size_t>(size));
		return vmaFlushAllocation(allocator, buffer.allocation, 0, size) == VK_SUCCESS;
	}
};

} // namespace

bool uploadBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& buffer) {
	StagingBuffer staging;
	if (!staging.initialize(data, size)) {
		return false;
	}
	const VkBufferCreateInfo buffer_info = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = size, .usage = usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE,
	};
	const VmaAllocationCreateInfo allocation_info = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
		.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	};
	if (vmaCreateBuffer(internal::context.allocator, &buffer_info, &allocation_info,
	                    &buffer.handle, &buffer.allocation, nullptr) != VK_SUCCESS) {
		return false;
	}
	UploadCommands commands;
	if (!commands.begin()) {
		return false;
	}
	const VkBufferCopy copy = {.size = size};
	vkCmdCopyBuffer(commands.buffer, staging.buffer.handle, buffer.handle, 1, &copy);
	const VkBufferMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
		.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.dstAccessMask = usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT
			? VK_ACCESS_INDEX_READ_BIT : VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.buffer = buffer.handle, .offset = 0, .size = size,
	};
	vkCmdPipelineBarrier(commands.buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
	                     VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 0, nullptr, 1, &barrier, 0, nullptr);
	return commands.finish();
}

bool loadTexture(const char* path, Texture& texture) {
	int width = 0;
	int height = 0;
	std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels(
		stbi_load(path, &width, &height, nullptr, STBI_rgb_alpha), stbi_image_free);
	if (!pixels) {
		std::cerr << "Не удалось загрузить текстуру " << path << ": " << stbi_failure_reason() << '\n';
		return false;
	}
	const VkDeviceSize size = VkDeviceSize(width) * VkDeviceSize(height) * 4;
	StagingBuffer staging;
	if (!staging.initialize(pixels.get(), size)) {
		return false;
	}
	auto& context = internal::context;
	const VkImageCreateInfo image_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
		.imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_SRGB,
		.extent = {uint32_t(width), uint32_t(height), 1},
		.mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
		.tiling = VK_IMAGE_TILING_OPTIMAL,
		.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
		.sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
	};
	const VmaAllocationCreateInfo allocation_info = {
		.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
		.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
	};
	if (vmaCreateImage(context.allocator, &image_info, &allocation_info,
	                   &texture.image, &texture.allocation, nullptr) != VK_SUCCESS) {
		return false;
	}
	UploadCommands commands;
	if (!commands.begin()) {
		return false;
	}
	VkImageMemoryBarrier barrier = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
		.srcAccessMask = 0, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
		.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
		.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
		.image = texture.image,
		.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1},
	};
	vkCmdPipelineBarrier(commands.buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
	                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
	const VkBufferImageCopy copy = {
		.imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .layerCount = 1},
		.imageExtent = image_info.extent,
	};
	vkCmdCopyBufferToImage(commands.buffer, staging.buffer.handle, texture.image,
	                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
	barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
	barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
	barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
	vkCmdPipelineBarrier(commands.buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
	                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
	if (!commands.finish()) {
		return false;
	}
	const VkImageViewCreateInfo view_info = {
		.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
		.image = texture.image, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = image_info.format,
		.subresourceRange = barrier.subresourceRange,
	};
	if (vkCreateImageView(context.device, &view_info, nullptr, &texture.view) != VK_SUCCESS) {
		return false;
	}
	const VkSamplerCreateInfo sampler_info = {
		.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
		.magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
		.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
		.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
		.maxLod = 0.0f,
	};
	return vkCreateSampler(context.device, &sampler_info, nullptr, &texture.sampler) == VK_SUCCESS;
}

void destroy(Buffer& buffer) {
	if (buffer.handle != VK_NULL_HANDLE) {
		vmaDestroyBuffer(internal::context.allocator, buffer.handle, buffer.allocation);
	}
	buffer = {};
}

void destroy(Texture& texture) {
	auto& context = internal::context;
	vkDestroySampler(context.device, texture.sampler, nullptr);
	vkDestroyImageView(context.device, texture.view, nullptr);
	if (texture.image != VK_NULL_HANDLE) {
		vmaDestroyImage(context.allocator, texture.image, texture.allocation);
	}
	texture = {};
}

} // namespace graphics::resources
