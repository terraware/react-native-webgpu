#pragma once

#include <memory>
#include <shared_mutex>
#include <unordered_map>
#include <utility>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#include <pthread.h>
#endif

#include "webgpu/webgpu_cpp.h"

namespace rnwgpu {

struct NativeInfo {
  void *nativeSurface;
  int width;
  int height;
};

struct Size {
  int width;
  int height;
};

class SurfaceInfo {
public:
  SurfaceInfo(wgpu::Instance gpu, int width, int height)
      : gpu(std::move(gpu)), width(width), height(height) {}

  ~SurfaceInfo() { surface = nullptr; }

  void reconfigure(int newWidth, int newHeight) {
    wgpu::Surface surfaceCopy;
    wgpu::SurfaceConfiguration configCopy;
    bool hasSurface = false;
    {
      std::unique_lock<std::shared_mutex> lock(_mutex);
      config.width = newWidth;
      config.height = newHeight;
      if (surface) {
        surfaceCopy = surface;
        configCopy = config;
        hasSurface = true;
      } else {
        _createOffscreenTexture();
      }
    }
    if (hasSurface) {
      _configureSurfaceOnMainThread(surfaceCopy, configCopy);
    }
  }

  void configure(wgpu::SurfaceConfiguration &newConfig) {
    wgpu::Surface surfaceCopy;
    wgpu::SurfaceConfiguration configCopy;
    bool hasSurface = false;
    {
      std::unique_lock<std::shared_mutex> lock(_mutex);
      config = newConfig;
      config.width = width;
      config.height = height;
      config.presentMode = wgpu::PresentMode::Fifo;
      if (surface) {
        surfaceCopy = surface;
        configCopy = config;
        hasSurface = true;
      } else {
        _createOffscreenTexture();
      }
    }
    if (hasSurface) {
      _configureSurfaceOnMainThread(surfaceCopy, configCopy);
    }
  }

  void unconfigure() {
    std::unique_lock<std::shared_mutex> lock(_mutex);
    if (surface) {
      surface.Unconfigure();
    } else {
      texture = nullptr;
    }
  }

  void *switchToOffscreen() {
    std::unique_lock<std::shared_mutex> lock(_mutex);
    // We only do this if the onscreen surface is configured.
    auto isConfigured = config.device != nullptr;
    if (isConfigured) {
      wgpu::TextureDescriptor textureDesc;
      textureDesc.usage = wgpu::TextureUsage::RenderAttachment |
                          wgpu::TextureUsage::CopySrc |
                          wgpu::TextureUsage::TextureBinding;
      textureDesc.format = config.format;
      textureDesc.size.width = config.width;
      textureDesc.size.height = config.height;
      texture = config.device.CreateTexture(&textureDesc);
    }
    surface = nullptr;
    return nativeSurface;
  }

  void switchToOnscreen(void *newNativeSurface, wgpu::Surface newSurface) {
    std::unique_lock<std::shared_mutex> lock(_mutex);
    nativeSurface = newNativeSurface;
    surface = std::move(newSurface);
    // If we are comming from an offscreen context, we need to configure the new
    // surface
    if (texture != nullptr) {
      config.usage = config.usage | wgpu::TextureUsage::CopyDst;
      // switchToOnscreen is called from the main thread (MetalView), so
      // this calls surface.Configure directly without dispatching.
#ifdef __APPLE__
      dispatch_assert_queue_debug(dispatch_get_main_queue());
#endif
      surface.configure(config);
      // We flush the offscreen texture to the onscreen one
      // TODO: there is a faster way to do this without validation?
      wgpu::CommandEncoderDescriptor encoderDesc;
      auto device = config.device;
      wgpu::CommandEncoder encoder = device.CreateCommandEncoder(&encoderDesc);

      wgpu::TexelCopyTextureInfo sourceTexture = {};
      sourceTexture.texture = texture;

      wgpu::TexelCopyTextureInfo destinationTexture = {};
      wgpu::SurfaceTexture surfaceTexture;
      surface.GetCurrentTexture(&surfaceTexture);
      destinationTexture.texture = surfaceTexture.texture;

      wgpu::Extent3D size = {sourceTexture.texture.GetWidth(),
                             sourceTexture.texture.GetHeight(),
                             sourceTexture.texture.GetDepthOrArrayLayers()};

      encoder.CopyTextureToTexture(&sourceTexture, &destinationTexture, &size);

      wgpu::CommandBuffer commands = encoder.Finish();
      wgpu::Queue queue = device.GetQueue();
      queue.Submit(1, &commands);
      surface.Present();
      texture = nullptr;
    }
  }

  void resize(int newWidth, int newHeight) {
    std::unique_lock<std::shared_mutex> lock(_mutex);
    width = newWidth;
    height = newHeight;
  }

  void present() {
    std::unique_lock<std::shared_mutex> lock(_mutex);
    if (surface) {
      surface.Present();
    }
  }

  wgpu::Texture getCurrentTexture() {
    std::shared_lock<std::shared_mutex> lock(_mutex);
    if (surface) {
      wgpu::SurfaceTexture surfaceTexture;
      surface.GetCurrentTexture(&surfaceTexture);
      return surfaceTexture.texture;
    } else {
      return texture;
    }
  }

  NativeInfo getNativeInfo() {
    std::shared_lock<std::shared_mutex> lock(_mutex);
    return {.nativeSurface = nativeSurface, .width = width, .height = height};
  }

  Size getSize() {
    std::shared_lock<std::shared_mutex> lock(_mutex);
    return {.width = width, .height = height};
  }

  wgpu::SurfaceConfiguration getConfig() {
    std::shared_lock<std::shared_mutex> lock(_mutex);
    return config;
  }

  wgpu::Device getDevice() {
    std::shared_lock<std::shared_mutex> lock(_mutex);
    return config.device;
  }

private:
  // Calls surface.Configure on the main thread. On Apple, if we are not
  // already on the main thread, dispatches synchronously to avoid modifying
  // CAMetalLayer properties from a background thread (Dawn's Metal SwapChain
  // initialization touches the layer). Parameters are taken by value so the
  // caller can release _mutex before calling this, avoiding deadlock.
  static void _configureSurfaceOnMainThread(wgpu::Surface s,
                                            wgpu::SurfaceConfiguration c) {
#ifdef __APPLE__
    if (!pthread_main_np()) {
      dispatch_sync(dispatch_get_main_queue(), ^{
        s.Configure(&c);
      });
      return;
    }
#endif
    s.Configure(&c);
  }

  // Creates an offscreen texture from the current config. Must be called
  // with _mutex held.
  void _createOffscreenTexture() {
    wgpu::TextureDescriptor textureDesc;
    textureDesc.format = config.format;
    textureDesc.size.width = config.width;
    textureDesc.size.height = config.height;
    textureDesc.usage = wgpu::TextureUsage::RenderAttachment |
                        wgpu::TextureUsage::CopySrc |
                        wgpu::TextureUsage::TextureBinding;
    texture = config.device.CreateTexture(&textureDesc);
  }

  mutable std::shared_mutex _mutex;
  void *nativeSurface = nullptr;
  wgpu::Surface surface = nullptr;
  wgpu::Texture texture = nullptr;
  wgpu::Instance gpu;
  wgpu::SurfaceConfiguration config;
  int width;
  int height;
};

class SurfaceRegistry {
public:
  static SurfaceRegistry &getInstance() {
    static SurfaceRegistry instance;
    return instance;
  }

  SurfaceRegistry(const SurfaceRegistry &) = delete;
  SurfaceRegistry &operator=(const SurfaceRegistry &) = delete;

  std::shared_ptr<SurfaceInfo> getSurfaceInfo(int id) {
    std::shared_lock<std::shared_mutex> lock(_mutex);
    auto it = _registry.find(id);
    if (it != _registry.end()) {
      return it->second;
    }
    return nullptr;
  }

  void removeSurfaceInfo(int id) {
    std::unique_lock<std::shared_mutex> lock(_mutex);
    _registry.erase(id);
  }

  std::shared_ptr<SurfaceInfo> addSurfaceInfo(int id, wgpu::Instance gpu,
                                              int width, int height) {
    std::unique_lock<std::shared_mutex> lock(_mutex);
    auto info = std::make_shared<SurfaceInfo>(gpu, width, height);
    _registry[id] = info;
    return info;
  }

  std::shared_ptr<SurfaceInfo>
  getSurfaceInfoOrCreate(int id, wgpu::Instance gpu, int width, int height) {
    std::unique_lock<std::shared_mutex> lock(_mutex);
    auto it = _registry.find(id);
    if (it != _registry.end()) {
      return it->second;
    }
    auto info = std::make_shared<SurfaceInfo>(gpu, width, height);
    _registry[id] = info;
    return info;
  }

private:
  SurfaceRegistry() = default;
  mutable std::shared_mutex _mutex;
  std::unordered_map<int, std::shared_ptr<SurfaceInfo>> _registry;
};

} // namespace rnwgpu
