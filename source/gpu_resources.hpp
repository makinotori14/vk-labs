#pragma once

#include "graphics_internal.hpp"

namespace graphics::resources {

struct Buffer {
	VkBuffer handle = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
};

struct Texture {
	VkImage image = VK_NULL_HANDLE;
	VmaAllocation allocation = nullptr;
	VkImageView view = VK_NULL_HANDLE;
	VkSampler sampler = VK_NULL_HANDLE;
};

bool uploadBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage, Buffer& buffer);
bool loadTexture(const char* path, Texture& texture);
void destroy(Buffer& buffer);
void destroy(Texture& texture);

} // namespace graphics::resources
