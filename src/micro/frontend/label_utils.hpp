#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace mind_micro_labels {

inline constexpr std::string_view kAllLabel = "all";

[[nodiscard]] inline bool is_all_label(std::string_view name) {
    return name == kAllLabel;
}

[[nodiscard]] inline bool is_soma_label(std::string_view name) {
    return name == "soma";
}

[[nodiscard]] inline bool is_axon_label(std::string_view name) {
    return name == "axon";
}

[[nodiscard]] inline bool is_dend_label(std::string_view name) {
    return name == "dend";
}

[[nodiscard]] inline bool is_apic_label(std::string_view name) {
    return name == "apic";
}

}  // namespace mind_micro_labels
