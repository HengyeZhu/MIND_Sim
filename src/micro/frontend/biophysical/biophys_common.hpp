#pragma once

#include "biophys_param_types.hpp"
#include "morph/cell_template_layout.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace mind_micro_biophysical {

enum class ObjectOpKind {
    VInit = 0,
    Cm,
    Ra,
};

[[nodiscard]] inline const char* section_property_name(ObjectOpKind kind) {
    if (kind == ObjectOpKind::VInit) {
        return "v_init";
    }
    if (kind == ObjectOpKind::Cm) {
        return "cm";
    }
    if (kind == ObjectOpKind::Ra) {
        return "ra";
    }
    throw std::runtime_error("unsupported object section property");
}

inline void validate_section_property_value(ObjectOpKind kind, double value) {
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            std::string("section ") + section_property_name(kind) + " must be finite");
    }
    if (kind != ObjectOpKind::VInit && value <= 0.0) {
        throw std::runtime_error(
            std::string("section ") + section_property_name(kind) + " must be positive and finite");
    }
}

template <typename T>
inline void sort_unique_inplace(std::vector<T>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

[[nodiscard]] inline std::vector<std::size_t> normalize_section_group_indices(
    const mind_micro_model::CellTemplateMorphLayout& morph,
    int gid,
    const std::vector<std::size_t>& section_indices) {
    std::vector<std::size_t> normalized_indices{};
    normalized_indices.reserve(section_indices.size());
    for (const auto section_index : section_indices) {
        normalized_indices.push_back(mind_micro_model::require_section_index(morph, gid, section_index));
    }
    sort_unique_inplace(normalized_indices);
    if (normalized_indices.empty()) {
        throw std::runtime_error("section group resolved to no sections");
    }
    return normalized_indices;
}

template <typename Fn>
inline void for_each_section_segment_original_node(const mind_micro_model::CellTemplateMorphLayout& morph,
                                                   int gid,
                                                   std::size_t section_index,
                                                   Fn&& fn) {
    const auto& tpl = mind_micro_model::require_cell_template(morph, gid);
    const auto sec_index = mind_micro_model::require_section_index(morph, gid, section_index);
    const auto base = tpl.section_node_base_id[sec_index];
    const auto nseg = tpl.section_nseg[sec_index];
    for (std::int32_t segment = 0; segment < nseg; ++segment) {
        const auto original = mind_micro_model::map_template_node_to_original(morph, gid, base + segment);
        if (original < 0) {
            throw std::runtime_error("segment node resolved to invalid original index");
        }
        fn(static_cast<std::size_t>(original));
    }
}

}  // namespace mind_micro_biophysical
