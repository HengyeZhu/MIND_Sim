#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mind_sim::macro::sim {

struct ScalarBuffer {
    std::vector<float> values{};

    ScalarBuffer() = default;
    explicit ScalarBuffer(std::size_t exposure_count): values(exposure_count, 0.0F) {}

    [[nodiscard]] std::size_t size() const noexcept {
        return values.size();
    }

    [[nodiscard]] float get(int exposure_id, std::string_view context) const {
        if (exposure_id < 0 || static_cast<std::size_t>(exposure_id) >= values.size()) {
            throw std::runtime_error(std::string(context) + " exposure index out of range");
        }
        return values[static_cast<std::size_t>(exposure_id)];
    }

    [[nodiscard]] float& at(int exposure_id, std::string_view context) {
        if (exposure_id < 0 || static_cast<std::size_t>(exposure_id) >= values.size()) {
            throw std::runtime_error(std::string(context) + " exposure index out of range");
        }
        return values[static_cast<std::size_t>(exposure_id)];
    }

    void fill(float value) {
        std::fill(values.begin(), values.end(), value);
    }
};

struct ExposureRecord {
    int roi_count{0};
    int exposure_count{0};
    std::vector<int> roi_indices{};
    std::vector<float> values{};

    [[nodiscard]] std::size_t recorded_roi_count() const noexcept {
        return roi_indices.size();
    }

    [[nodiscard]] std::size_t sample_count() const noexcept {
        const auto width = roi_indices.size() * static_cast<std::size_t>(exposure_count);
        return width == 0 ? 0 : values.size() / width;
    }
};

struct MacroSimulationResult {
    std::vector<double> times{};
    ExposureRecord exposures{};
};

}  // namespace mind_sim::macro::sim
