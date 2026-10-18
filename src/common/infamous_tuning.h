// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include "common/logging/log.h"
#include "common/path_util.h"
#include "common/types.h"

namespace Common::Infamous {

/// Settings read from perf_tuning.ini in the user directory (the file of the AYOUB builds, only
/// the keys below are used here), reloaded when the file changes.
struct Tuning {
    /// Guest occlusion queries are counted by the GPU (lights behind walls, flares hidden by
    /// geometry), instead of reporting every sample visible.
    std::atomic<bool> real_occlusion{true};
    /// Divides the progress of the GPU clock the guest reads (1 = host time).
    std::atomic<u32> gpu_clock_div{1};
    /// Locks the GPU clock to the guest frame, in microseconds a frame (0 = host time).
    std::atomic<u32> gpu_clock_frame_us{0};
    /// Logs the small render passes and constant buffers of the exposure chain (diagnostic).
    std::atomic<u32> lum_debug{0};
};

inline Tuning& GetTuning() {
    static Tuning tuning;
    return tuning;
}

/// Guest frames submitted, for the clock locked to the frame.
inline std::atomic<u64>& GuestFrameCounter() {
    static std::atomic<u64> counter{0};
    return counter;
}

/// Reads perf_tuning.ini if it changed since it was last read. Cheap to call often.
inline void RefreshTuning() {
    static std::filesystem::file_time_type last_write{};
    static bool had_file = false;
    static std::chrono::steady_clock::time_point last_check{};
    const auto now = std::chrono::steady_clock::now();
    if (last_check != std::chrono::steady_clock::time_point{} &&
        now - last_check < std::chrono::seconds(1)) {
        return;
    }
    last_check = now;

    const auto path = Common::FS::GetUserPath(Common::FS::PathType::UserDir) / "perf_tuning.ini";
    std::error_code error;
    const auto write_time = std::filesystem::last_write_time(path, error);
    const bool has_file = !error;
    if (has_file == had_file && (!has_file || write_time == last_write)) {
        return;
    }
    had_file = has_file;
    last_write = has_file ? write_time : std::filesystem::file_time_type{};
    if (!has_file) {
        return;
    }

    auto& tuning = GetTuning();
    const auto trim = [](std::string text) {
        const auto first = text.find_first_not_of(" \t\r");
        const auto last = text.find_last_not_of(" \t\r");
        return first == std::string::npos ? std::string{} : text.substr(first, last - first + 1);
    };
    std::ifstream file{path};
    std::string line;
    while (std::getline(file, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) {
            line.resize(comment);
        }
        const auto equal = line.find('=');
        if (equal == std::string::npos) {
            continue;
        }
        const std::string key = trim(line.substr(0, equal));
        const std::string value = trim(line.substr(equal + 1));
        if (value.empty()) {
            continue;
        }
        if (key == "real_occlusion") {
            tuning.real_occlusion.store(value != "0", std::memory_order_relaxed);
        } else if (key == "gpu_clock_div") {
            tuning.gpu_clock_div.store(
                static_cast<u32>(std::max<unsigned long>(std::strtoul(value.c_str(), nullptr, 10), 1)),
                std::memory_order_relaxed);
        } else if (key == "lum_debug") {
            tuning.lum_debug.store(static_cast<u32>(std::strtoul(value.c_str(), nullptr, 10)),
                                   std::memory_order_relaxed);
        } else if (key == "gpu_clock_frame_us") {
            tuning.gpu_clock_frame_us.store(
                static_cast<u32>(std::strtoul(value.c_str(), nullptr, 10)),
                std::memory_order_relaxed);
        }
    }
    LOG_INFO(Render,
             "perf_tuning.ini: real_occlusion={} gpu_clock_div={} gpu_clock_frame_us={} "
             "lum_debug={}",
             tuning.real_occlusion.load(), tuning.gpu_clock_div.load(),
             tuning.gpu_clock_frame_us.load(), tuning.lum_debug.load());
}

} // namespace Common::Infamous
