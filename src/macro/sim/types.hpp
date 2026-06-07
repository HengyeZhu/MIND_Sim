#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace mind_sim::macro::sim {

struct ScalarBuffer {
    std::vector<double> values{};

    ScalarBuffer() = default;
    explicit ScalarBuffer(std::size_t output_count): values(output_count, 0.0) {}

    [[nodiscard]] std::size_t size() const noexcept {
        return values.size();
    }

    [[nodiscard]] double get(int output_id, std::string_view context) const {
        if (output_id < 0 || static_cast<std::size_t>(output_id) >= values.size()) {
            throw std::runtime_error(std::string(context) + " output index out of range");
        }
        return values[static_cast<std::size_t>(output_id)];
    }

    [[nodiscard]] double& at(int output_id, std::string_view context) {
        if (output_id < 0 || static_cast<std::size_t>(output_id) >= values.size()) {
            throw std::runtime_error(std::string(context) + " output index out of range");
        }
        return values[static_cast<std::size_t>(output_id)];
    }

    void fill(double value) {
        std::fill(values.begin(), values.end(), value);
    }
};

struct RecordTable {
    int roi_count{0};
    int output_count{0};
    std::vector<int> roi_indices{};
    std::vector<int> output_indices{};
    std::vector<double> values{};

    [[nodiscard]] std::size_t recorded_roi_count() const noexcept {
        return roi_indices.size();
    }

    [[nodiscard]] std::size_t recorded_output_count() const noexcept {
        return output_indices.size();
    }

    [[nodiscard]] std::size_t sample_count() const noexcept {
        const auto width = roi_indices.size() * static_cast<std::size_t>(output_count);
        return width == 0 ? 0 : values.size() / width;
    }
};

struct MacroSimulationResult {
    std::vector<double> times{};
    RecordTable records{};
};

}  // namespace mind_sim::macro::sim
