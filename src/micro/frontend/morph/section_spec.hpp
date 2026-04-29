#pragma once

#include "morph/dat_to_section.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mind_micro_frontend {

using SectionNseg = std::vector<std::int32_t>;

struct SectionPt3d {
    float x_um{};
    float y_um{};
    float z_um{};
    float diam_um{};
};

struct SectionSpec {
    SectionSpec() = default;
    explicit SectionSpec(std::string name_)
        : name(std::move(name_)) {}
    SectionSpec(std::string name_, std::string label_)
        : name(std::move(name_)),
          label(std::move(label_)) {
        if (label.empty()) {
            throw std::runtime_error("section label name is empty");
        }
        if (label == "all") {
            throw std::runtime_error("section label name 'all' is reserved");
        }
    }

    std::string name{};
    std::string parent_name{};
    double parentx{1.0};
    std::string label{};
    std::int32_t nseg{1};
    double L_um{0.0};
    double diam_um{0.0};
    std::vector<SectionPt3d> pt3d{};
};

[[nodiscard]] mind_micro_morph::Morph build_morph_from_sections(const std::vector<SectionSpec>& sections,
                                                             SectionNseg& out_nseg);
void apply_section_nseg(mind_micro_morph::Morph& morph, const SectionNseg& section_nseg);
[[nodiscard]] std::vector<SectionSpec> section_specs_from_morph(const mind_micro_morph::Morph& morph);
[[nodiscard]] std::vector<SectionSpec> delete_subtree_specs(const std::vector<SectionSpec>& sections,
                                                            std::string_view section_name);
[[nodiscard]] std::vector<SectionSpec> delete_label_specs(const std::vector<SectionSpec>& sections,
                                                          std::string_view label);

[[nodiscard]] std::vector<SectionSpec> load_swc_sections(const std::string& swc_file);
[[nodiscard]] std::vector<SectionSpec> load_asc_sections(const std::string& asc_file);

}  // namespace mind_micro_frontend
