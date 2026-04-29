#pragma once

#include "morph/dat_to_section.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mind_micro_model {

inline constexpr std::string_view kAllSectionLabel = "all";

struct CellTemplateInfo {
    std::string name{};
    int cell_base{0};
    int num_cells{0};

    std::size_t tpl_nodes_per_cell{0};
    std::int32_t tpl_soma_node{-1};

    std::vector<std::string> label_names{};
    std::unordered_map<std::string, std::size_t> label_index{};
    std::vector<std::vector<mind_micro_morph::section_id>> sections_by_label{};
    std::vector<std::int32_t> section_label_u_by_sec{};
    std::vector<std::int32_t> section_node_base_id{};
    std::vector<std::int32_t> section_nseg{};
    std::vector<std::int32_t> section_parent_node_id{};
};

struct CellTemplateMorphLayout {
    int num_cells_total{0};
    std::size_t nnode{0};
    std::vector<CellTemplateInfo> templates{};
    std::vector<std::size_t> cell_template_id{};
    std::vector<std::size_t> cell_nonroot_base{};
};

[[nodiscard]] inline const CellTemplateInfo& require_cell_template(const CellTemplateMorphLayout& morph, int gid) {
    if (gid < 0 || gid >= morph.num_cells_total) {
        throw std::runtime_error("gid out of range: " + std::to_string(gid));
    }
    const auto cell = static_cast<std::size_t>(gid);
    if (cell >= morph.cell_template_id.size()) {
        throw std::runtime_error("cell template id mapping is unavailable for gid=" + std::to_string(gid));
    }
    const auto tpl_id = morph.cell_template_id[cell];
    if (tpl_id >= morph.templates.size()) {
        throw std::runtime_error("template id out of range for gid=" + std::to_string(gid));
    }
    return morph.templates[tpl_id];
}

[[nodiscard]] inline std::size_t require_section_index(const CellTemplateMorphLayout& morph,
                                                       int gid,
                                                       std::size_t section_index) {
    const auto& tpl = require_cell_template(morph, gid);
    if (section_index >= tpl.section_nseg.size()) {
        throw std::runtime_error(
            "section index out of range for gid=" + std::to_string(gid) +
            ": " + std::to_string(section_index));
    }
    return section_index;
}

[[nodiscard]] inline std::size_t section_count(const CellTemplateMorphLayout& morph, int gid) {
    return require_cell_template(morph, gid).section_nseg.size();
}

inline void require_label_exists(const CellTemplateMorphLayout& morph, int gid, const std::string& label) {
    const auto& tpl = require_cell_template(morph, gid);
    if (label == kAllSectionLabel) {
        return;
    }
    if (tpl.label_index.find(label) == tpl.label_index.end()) {
        throw std::runtime_error("unknown label '" + label + "' for gid=" + std::to_string(gid));
    }
}

[[nodiscard]] inline std::vector<std::size_t> section_indices_for_label(const CellTemplateMorphLayout& morph,
                                                                        int gid,
                                                                        const std::string& label) {
    const auto& tpl = require_cell_template(morph, gid);
    std::vector<std::size_t> out;
    if (label == kAllSectionLabel) {
        out.reserve(tpl.section_nseg.size());
        for (std::size_t i = 0; i < tpl.section_nseg.size(); ++i) {
            out.push_back(i);
        }
        return out;
    }
    const auto it = tpl.label_index.find(label);
    if (it == tpl.label_index.end()) {
        throw std::runtime_error("unknown label '" + label + "' for gid=" + std::to_string(gid));
    }
    const auto label_index = it->second;
    if (label_index >= tpl.sections_by_label.size()) {
        throw std::runtime_error("section label index out of range");
    }
    out.reserve(tpl.sections_by_label[label_index].size());
    for (const auto sec_id : tpl.sections_by_label[label_index]) {
        out.push_back(static_cast<std::size_t>(sec_id));
    }
    return out;
}

[[nodiscard]] inline std::string section_label(const CellTemplateMorphLayout& morph,
                                               int gid,
                                               std::size_t section_index) {
    const auto sec_index = require_section_index(morph, gid, section_index);
    const auto& tpl = require_cell_template(morph, gid);
    if (sec_index >= tpl.section_label_u_by_sec.size()) {
        throw std::runtime_error("section label lookup index out of range");
    }
    const auto label_u = tpl.section_label_u_by_sec[sec_index];
    if (label_u < 0) {
        return "";
    }
    const auto label_index = static_cast<std::size_t>(label_u);
    if (label_index >= tpl.label_names.size()) {
        throw std::runtime_error("section label name index out of range");
    }
    return tpl.label_names[label_index];
}

[[nodiscard]] inline std::int32_t map_template_node_to_original(const CellTemplateMorphLayout& morph,
                                                                int gid,
                                                                std::int32_t template_node_index) {
    if (template_node_index < 0) {
        return -1;
    }
    if (template_node_index == 0) {
        return static_cast<std::int32_t>(gid);
    }
    const auto cell_u = static_cast<std::size_t>(gid);
    const auto base = morph.cell_nonroot_base[cell_u];
    const auto idx = base + (static_cast<std::size_t>(template_node_index) - 1);
    return static_cast<std::int32_t>(idx);
}

}  // namespace mind_micro_model
