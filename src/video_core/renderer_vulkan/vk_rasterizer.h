// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <optional>
#include <span>
#include "common/recursive_lock.h"
#include "common/shared_first_mutex.h"
#include "video_core/buffer_cache/buffer_cache.h"
#include "video_core/page_manager.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/texture_cache.h"

namespace AmdGpu {
struct Liverpool;
}

namespace Core {
class MemoryManager;
}

namespace Vulkan {

class GraphicsPipeline;
class Runtime;

/**
 * Guest occlusion queries counted by the GPU. The guest dumps the depth block sample counters
 * (ZPASS_DONE) before the draws it tests, at a 16 byte aligned address, and after them, 8 bytes
 * further; the sum of the differences of the counter pairs is the number of samples that passed
 * the depth test. The counters run all the time on the GPU, so queries may overlap or nest (a
 * query around a whole pass containing smaller queries). Here Vulkan occlusion queries count the
 * samples of consecutive segments of the command stream, shared by all the open guest queries: a
 * segment ends at each dump and at the end of each rendering scope (Vulkan queries begun in a
 * rendering scope must end in it, and only one may be active). A guest query reports the
 * segments between its two dumps, one per counter pair.
 */
class OcclusionQueries final : public RenderingObserver {
public:
    static constexpr u32 MaxCounterPairs = 16;
    static constexpr u32 MaxOpenQueries = 16;

    explicit OcclusionQueries(const Instance& instance, Scheduler& scheduler);
    ~OcclusionQueries() override;

    void OnBeginRendering(const CommandRecorder& cmdbuf) override;
    void OnEndRendering(const CommandRecorder& cmdbuf) override;

    /// Begin dump: counts the samples of the following draws for the query at this address.
    /// Returns false when it cannot be counted (too many open queries).
    bool Begin(VAddr address, u32 num_pairs);

    /// End dump of the query that began at begin_address: stops counting it and fills the Vulkan
    /// queries that counted its samples, oldest first. Returns false when no counted query began
    /// there, or when some of its samples were not counted (too many segments).
    bool End(VAddr begin_address,
             boost::container::static_vector<u32, MaxCounterPairs>& out_queries);

    /// Stops counting all open queries (real queries turned off).
    void AbandonAll();

    /// Starts counting again for the open queries after an end dump, if a rendering scope is
    /// active (otherwise the next one starts counting).
    void Resume();

    /// End of a guest frame: queries open for several frames are forgotten (never ended), so
    /// that they do not keep segments and open slots forever.
    void OnFrameEnd();

    /// The queries are read by the current command buffer: not reused before it completes.
    void MarkUsed(std::span<const u32> queries);

    [[nodiscard]] bool HasOpen() const noexcept {
        return !open.empty();
    }

    /// Whether a counted query begun at this address is open.
    [[nodiscard]] bool IsOpen(VAddr address) const noexcept {
        return std::any_of(open.begin(), open.end(),
                           [address](const OpenQuery& q) { return q.address == address; });
    }

    [[nodiscard]] u32 NumOpen() const noexcept {
        return static_cast<u32>(open.size());
    }

    [[nodiscard]] vk::QueryPool Pool() const noexcept {
        return *pool;
    }

private:
    struct OpenQuery {
        VAddr address;
        u32 max_segments;  ///< Counter pairs of the query: one segment each.
        u32 first_segment; ///< Index in segments of its first segment.
        u32 frame;         ///< Guest frame of its begin dump.
        bool incomplete;   ///< Some of its samples were not counted.
    };

    /// Forgets the segments that no open query needs any more.
    void DropOldSegments();

    void BeginSegment(const CommandRecorder& cmdbuf);
    void EndSegment(const CommandRecorder& cmdbuf);
    /// Ends the active segment: the samples before and after a dump are counted apart.
    void Cut();

    static constexpr u32 NumQueries = 4096;
    static constexpr u32 MaxSegments = 256;

    const Instance& instance;
    Scheduler& scheduler;
    vk::UniqueQueryPool pool;
    vk::QueryControlFlags control_flags{};
    /// GPU tick of the command buffer that last used each query (reused when it completed).
    std::array<u64, NumQueries> query_ticks{};
    u32 next_query{};
    bool active{}; ///< The last segment is counting in the current rendering scope.
    u32 frame{};   ///< Guest frames ended.
    boost::container::static_vector<OpenQuery, MaxOpenQueries> open;
    /// Vulkan queries of the segments since the oldest open query began, in order.
    boost::container::static_vector<u32, MaxSegments> segments;
};

/// What the command processor writes for a guest occlusion counter dump.
enum class OcclusionDumpResult {
    Emulated,          ///< Emulated counters at the dump address (all samples visible).
    EmulatedWithBegin, ///< Emulated counters at the dump address and 8 bytes before it.
    Counted,           ///< Written (or to be written) by the GPU, nothing to write.
};

class Rasterizer {
public:
    explicit Rasterizer(const Instance& instance, Scheduler& scheduler, Runtime& runtime,
                        AmdGpu::Liverpool* liverpool);
    ~Rasterizer();

    [[nodiscard]] Scheduler& GetScheduler() noexcept {
        return scheduler;
    }

    [[nodiscard]] Runtime& GetRuntime() noexcept {
        return runtime;
    }

    [[nodiscard]] VideoCore::BufferCache& GetBufferCache() noexcept {
        return buffer_cache;
    }

    [[nodiscard]] VideoCore::TextureCache& GetTextureCache() noexcept {
        return texture_cache;
    }

    void Draw(bool is_indexed, u32 index_offset = 0);
    void DrawIndirect(bool is_indexed, VAddr arg_address, u32 offset, u32 size, u32 max_count,
                      VAddr count_address, u16 vertex_sgpr_offset, u16 instance_sgpr_offset);

    void DispatchDirect();
    void DispatchIndirect(VAddr address, u32 offset, u32 size);

    void ScopeMarker(fmt::string_view fmt, fmt::format_args args, auto&& func) {
        if (host_markers_enabled) {
            ScopeMarkerBegin(fmt::vformat(fmt, args));
            func();
            ScopeMarkerEnd();
        } else {
            func();
        }
    }

    void ScopeMarkerBegin(const std::string_view& str, bool from_guest = false);
    void ScopeMarkerEnd(bool from_guest = false);
    void ScopedMarkerInsert(const std::string_view& str, bool from_guest = false);
    void ScopedMarkerInsertColor(const std::string_view& str, const u32 color,
                                 bool from_guest = false);

    void FillBuffer(VAddr address, u32 num_bytes, u32 value, bool is_gds);
    void CopyBuffer(VAddr dst, VAddr src, u32 num_bytes, bool dst_gds, bool src_gds);
    u32 ReadDataFromGds(u32 gsd_offset);
    bool InvalidateMemory(VAddr addr, u64 size, bool assume_locks = false);
    /// Marks buffer memory a game thread is about to write as CPU modified ahead of it.
    void InvalidateBuffersAhead(VAddr addr, u64 size);
    bool ReadMemory(VAddr addr, u64 size, bool assume_locks = false);
    /// Notes memory written past its page protection, like fence values.
    void OnBackingWritten(VAddr addr, u64 size);
    bool IsMapped(VAddr addr, u64 size);
    void MapMemory(VAddr addr, u64 size);
    void RegisterMemory(VAddr addr, u64 size);
    void UnmapMemory(VAddr addr, u64 size);

    u64 Flush();
    void Finish();
    void OnSubmit();

    /// Handles a ZPASS_DONE counter dump of the command processor (see OcclusionQueries).
    OcclusionDumpResult OcclusionDump(VAddr address, u32 num_pairs);

    /// Whether the memory is the result of a counted occlusion query; value is set to what
    /// the 32 bit word holds when known without reading it back from the GPU.
    bool IsCountedOcclusionResult(VAddr address, std::optional<u32>& value) const;
    void OnFence();

    PipelineCache& GetPipelineCache() {
        return pipeline_cache;
    }

    template <typename Func>
    void ForEachMappedRangeInRange(VAddr addr, u64 size, Func&& func) {
        const auto range = decltype(mapped_ranges)::interval_type::right_open(addr, addr + size);
        Common::RecursiveSharedLock lock{mapped_ranges_mutex};
        for (const auto& mapped_range : (mapped_ranges & range)) {
            func(mapped_range);
        }
    }

    std::thread::id GetGpuCommandProcessorThread();
#ifdef __linux__
    u32 GetGpuCommandProcessorThreadId();
#endif

private:
    void PrepareRenderState(const GraphicsPipeline* pipeline);
    RenderState BeginRendering(const GraphicsPipeline* pipeline);
    void Resolve();
    void DepthStencilCopy(bool is_depth, bool is_stencil);
    void EliminateFastClear();

    void UpdateDynamicState(const GraphicsPipeline* pipeline, bool is_indexed) const;
    void UpdateViewportScissorState() const;
    void UpdateDepthStencilState() const;
    void UpdatePrimitiveState(bool is_indexed) const;
    void UpdateRasterizationState() const;
    void UpdateColorBlendingState(const GraphicsPipeline* pipeline) const;

    bool FilterDraw();

    void BindBuffers(const Shader::Info& stage, Shader::Backend::Bindings& binding,
                     Shader::PushData& push_data);
    void BindTextures(const Shader::Info& stage, Shader::Backend::Bindings& binding);
    bool BindResources(const Pipeline* pipeline);
    void BindComputePipeline(const ComputePipeline& pipeline);

    void BindVertexBuffers(const GraphicsPipeline* pipeline);
    void BindIndexBuffer(u32 index_offset = 0);

    void ResetBindings(bool is_compute);

    /// Returns true if a dispatch of one of the game's compute rings could run on the second
    /// queue, as far as the shader tells, before its resources are bound.
    bool CanBindOnAsync(const ComputePipeline& pipeline) const;

    /// Returns true if the dispatch bound can run on the second queue, making that wait for the
    /// graphics work submitted that writes what it reads or touches what it writes. False if
    /// binding it recorded graphics work, or graphics work not submitted yet is such.
    bool OrderOnAsync(u64 commands_before);

    /// Records the dispatch bound on the second queue.
    void DispatchOnAsync(const ComputePipeline& pipeline, u32 dim_x, u32 dim_y, u32 dim_z);

    bool IsComputeMetaClear(const Pipeline* pipeline);
    bool IsComputeImageCopy(const Pipeline* pipeline);
    bool IsComputeImageClear(const Pipeline* pipeline);

private:
    friend class VideoCore::BufferCache;

    const Instance& instance;
    Scheduler& scheduler;
    Runtime& runtime;
    VideoCore::PageManager page_manager;
    VideoCore::BufferCache buffer_cache;
    VideoCore::TextureCache texture_cache;
    AmdGpu::Liverpool* liverpool;
    Core::MemoryManager* memory;
    boost::icl::interval_set<VAddr> mapped_ranges;
    Common::SharedFirstMutex mapped_ranges_mutex;
    PipelineCache pipeline_cache;
    const bool host_markers_enabled;
    const bool guest_markers_enabled;
    OcclusionQueries occlusion{instance, scheduler};

    /// Forgets the results of counted occlusion queries in this range (written again).
    void ForgetCountedResults(VAddr address, u64 size);
    struct CountedResults {
        VAddr address;
        u32 num_pairs;  ///< Counter pairs written by the GPU.
        u32 num_counts; ///< Pairs whose end counter holds samples counted by the GPU.
    };
    static constexpr size_t MaxCountedResults = 64;
    /// Results of the last occlusion queries counted by the GPU, most recent last.
    boost::container::static_vector<CountedResults, MaxCountedResults> counted_results;

    /// A texture bound by a stage. Only the view and the type of the descriptor it was found with
    /// are needed to bind it, and copying all of it, nearly 400 bytes, was a good part of binding
    /// each texture. What it is made from is kept to find the image again if it was replaced.
    struct ImageBinding {
        VideoCore::ImageId image_id;
        VideoCore::ImageViewInfo view_info;
        VideoCore::TextureCache::BindingType type;
        AmdGpu::Image tsharp;
        const Shader::ImageResource* resource;
        u32 level;
    };
    std::array<ImageBinding, Shader::NUM_IMAGES> image_bindings;
    struct TargetBinding {
        VideoCore::ImageId image_id;
        VideoCore::TextureCache::ImageDesc desc;
    };
    std::array<TargetBinding, AmdGpu::NUM_COLOR_BUFFERS> cb_descs;
    std::pair<VideoCore::ImageId, VideoCore::TextureCache::ImageDesc> db_desc;

    boost::container::static_vector<vk::DescriptorImageInfo, Shader::NUM_IMAGES> image_infos;
    boost::container::static_vector<vk::DescriptorBufferInfo, Shader::NUM_BUFFERS> buffer_infos;

    struct BoundBuffer {
        const VideoCore::Buffer* buffer;
        u64 offset;
        u32 size;
        bool is_written;
    };
    /// Shader buffers, vertex buffers and the index buffer.
    boost::container::static_vector<BoundBuffer, Shader::NUM_BUFFERS + MaxVertexBufferCount + 1>
        bound_buffers;
    boost::container::static_vector<VideoCore::ImageId, Shader::NUM_IMAGES> bound_images;

    u32 set_write_index{};
    Pipeline::DescriptorWrites set_writes;
    Shader::PushData push_data;

    /// The vertex input last set in the command buffer, valid unless the dynamic state of the
    /// scheduler says it is dirty.
    VertexInputs<vk::VertexInputAttributeDescription2EXT> last_vertex_attributes;
    VertexInputs<vk::VertexInputBindingDescription2EXT> last_vertex_bindings;

    /// Everything the dynamic state of a draw is built from besides the registers covered by
    /// their version, as of the last draw that built it.
    struct DynamicStateInputs {
        u64 regs_version{};
        std::array<vk::ColorComponentFlags, AmdGpu::NUM_COLOR_BUFFERS> write_masks{};
        bool feedback_loop{};
        bool is_indexed{};

        bool operator==(const DynamicStateInputs&) const = default;
    };
    mutable DynamicStateInputs last_dynamic_inputs{};

    /// Everything the render targets of a draw are found from besides the registers covered by
    /// their version, as of the last draw that found them.
    struct RenderTargetInputs {
        u64 regs_version{};
        u64 image_generation{};
        std::array<u32, AmdGpu::NUM_COLOR_BUFFERS> cb_extents{};
        u32 db_extent{};
        u32 mrt_mask{};

        bool operator==(const RenderTargetInputs&) const = default;
    };
    RenderTargetInputs last_render_target_inputs{};

    bool attachment_feedback_loop{};
    bool needs_barrier{};
    /// Whether the draw being made reads memory in ways no accesses are kept for.
    bool untracked_access{};
    /// Whether the last work recorded was a dispatch, to count switches between draws and
    /// dispatches.
    bool last_work_compute{};
    /// The scheduler of the second queue the game's compute rings run on, if one is used, and
    /// the last of its ticks the graphics queue waits for.
    std::unique_ptr<Scheduler> async_scheduler;
    u64 async_tick_waited{};
    u64 graphics_tick_waited{};
};

} // namespace Vulkan
