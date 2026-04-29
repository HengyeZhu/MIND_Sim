#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace mind_micro_morph {

using origin_id = std::int32_t;
using swcdata_id = std::uint32_t;
using section_id = std::uint32_t;

inline constexpr swcdata_id invalid_swcdata_id = std::numeric_limits<swcdata_id>::max();
inline constexpr section_id invalid_section_id = std::numeric_limits<section_id>::max();

struct SwcData {
    // Keep pre-pt3d morphology coordinates/diameter in double precision so
    // parser and sectionification decisions can follow NEURON Import3d style.
    std::array<double, 3> xyz{};  // microns
    double d_um{};                // diameter (um)
    std::int8_t label{};
};

// High-performance SoA layout for pt3d points:
//  - xyz + diameter stored as float (matches NEURON Pt3d precision).
//  - arc stored as double for stable cumulative arclength.
//
// Sections reference subranges via Section::pt3d_offset + Section::pt3d_count.
struct Pt3dSoA {
    std::vector<float> x{};
    std::vector<float> y{};
    std::vector<float> z{};
    std::vector<float> d{};
    std::vector<double> arc{};

    [[nodiscard]] std::size_t size() const noexcept {
        return arc.size();
    }
    [[nodiscard]] bool empty() const noexcept {
        return arc.empty();
    }

    void clear() {
        x.clear();
        y.clear();
        z.clear();
        d.clear();
        arc.clear();
    }
};

struct Section {
    std::size_t offset{};  // into Morph::datas
    std::size_t count{};   // swc point ids used for pt3dadd (pre-sphere_rep)

    swcdata_id head_swc_id{invalid_swcdata_id};    // swc point id of the first point of the section
    swcdata_id parent_swc_id{invalid_swcdata_id};  // swc point id of the parent point (if any)

    section_id parent_sec_id{invalid_section_id};
    double parentx{1.0};     // connection position on parent section in [0,1]
    bool wire_first{false};  // matches Import3d_Section.first for SWC

    // Section label name (e.g. "soma", "axon", "dend", "apic").
    std::string label{};
    std::string name{};    // user-facing unique section name
    double rallbranch{1.0};
    // Discretization parameter (NEURON nseg). This is intentionally left unset by SWC import
    // and is expected to be populated by an explicit nseg policy stage before building nodes.
    std::int32_t nseg{0};

    std::size_t pt3d_offset{};  // into Morph::pt3d
    std::size_t pt3d_count{};   // number of pt3d points after sphere_rep
    double L_um{};              // pt3d arclength (um)
    // For NEURON-style "n3d==0" cylinders (no pt3d points): store the constant diameter here.
    // This keeps full double precision (NEURON section `diam` is double), while pt3d diameters
    // are stored as float.
    double diam_um{0.0};

    std::vector<section_id> children{};  // ordered like NEURON sibling list
};

struct Morph {
    std::vector<SwcData> swc{};
    std::vector<swcdata_id> datas{};
    std::vector<Section> sections{};
    Pt3dSoA pt3d{};

    // Fast lookup for NEURON-style workflows like:
    //   for (sec in sections_with_label[label_name]) { insert(channel); }
    //
    // labels are stored in label_names, and sections_by_label uses the same ordering.
    std::vector<std::string> label_names{};
    std::unordered_map<std::string, std::size_t> label_index{};
    std::vector<std::vector<section_id>> sections_by_label{};
};

[[nodiscard]] inline std::size_t ensure_section_label_index(std::vector<std::string>& label_names,
                                                            std::unordered_map<std::string, std::size_t>& label_index,
                                                            std::vector<std::vector<section_id>>& sections_by_label,
                                                            std::string_view label_name) {
    const auto key = std::string(label_name);
    if (key.empty()) {
        throw std::runtime_error("section label name is empty");
    }
    if (key == "all") {
        throw std::runtime_error("section label name 'all' is reserved");
    }
    if (auto it = label_index.find(key); it != label_index.end()) {
        return it->second;
    }
    const auto idx = label_names.size();
    label_names.push_back(key);
    label_index.emplace(key, idx);
    while (sections_by_label.size() <= idx) {
        sections_by_label.emplace_back();
    }
    return idx;
}

[[nodiscard]] inline std::size_t ensure_section_label_index(Morph& morph, std::string_view label_name) {
    return ensure_section_label_index(morph.label_names, morph.label_index, morph.sections_by_label, label_name);
}

Morph load_swc_morphology(const std::string& swc_path);
Morph load_asc_morphology(const std::string& asc_path);

// Return section ids for a given label. Unknown labels return an empty vector.
[[nodiscard]] std::vector<section_id> sections_with_label(const Morph& morph, std::string_view label);

}  // namespace mind_micro_morph
