// Module 2/4: Mutable Morph/Section helpers (user interaction layer)

#include "mutable_section.hpp"

#include "label_utils.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stack>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mind_micro_morph {
namespace {

void delete_marked_sections(Morph& morph, std::span<const std::uint8_t> marked_for_deletion) {
    std::size_t survivors = 0;
    for (const auto marked : marked_for_deletion) {
        if (!marked) {
            ++survivors;
        }
    }
    if (survivors == morph.sections.size()) {
        return;
    }

    // Build old->new section id mapping.
    std::vector<section_id> old2new(morph.sections.size(), invalid_section_id);
    std::vector<Section> new_sections;
    std::vector<std::string> new_label_names;
    std::unordered_map<std::string, std::size_t> new_label_index;
    std::vector<std::vector<section_id>> new_sections_by_label;

    for (section_id old_section_id = 0; old_section_id < morph.sections.size(); ++old_section_id) {
        if (marked_for_deletion[static_cast<std::size_t>(old_section_id)]) {
            continue;
        }
        const auto old_section_index = static_cast<std::size_t>(old_section_id);
        const auto new_section_id = static_cast<section_id>(new_sections.size());
        old2new[old_section_index] = new_section_id;
        new_sections.push_back(morph.sections[old_section_index]);
        new_sections.back().children.clear();

        const auto label_u = ensure_section_label_index(
            new_label_names, new_label_index, new_sections_by_label, morph.sections[old_section_index].label);
        new_sections_by_label[label_u].push_back(new_section_id);
    }

    // Compact datas and pt3d while fixing per-section offsets and parent pointers.
    std::vector<swcdata_id> new_datas;
    Pt3dSoA new_pt3d{};

    for (std::size_t new_section_index = 0; new_section_index < new_sections.size(); ++new_section_index) {
        auto& sec = new_sections[new_section_index];

        if (sec.parent_sec_id != invalid_section_id) {
            const auto mapped = old2new[static_cast<std::size_t>(sec.parent_sec_id)];
            if (mapped == invalid_section_id) {
                sec.parent_sec_id = invalid_section_id;
                sec.parentx = 1.0;
            } else {
                sec.parent_sec_id = mapped;
            }
        }

        const auto begin = sec.offset;
        const auto end = sec.offset + sec.count;
        sec.offset = new_datas.size();
        new_datas.insert(new_datas.end(), morph.datas.begin() + begin, morph.datas.begin() + end);
        sec.count = end - begin;

        if (sec.pt3d_count != 0) {
            const auto pbegin = sec.pt3d_offset;
            const auto pend = sec.pt3d_offset + sec.pt3d_count;
            const auto new_off = new_pt3d.size();
            new_pt3d.x.insert(new_pt3d.x.end(), morph.pt3d.x.begin() + pbegin, morph.pt3d.x.begin() + pend);
            new_pt3d.y.insert(new_pt3d.y.end(), morph.pt3d.y.begin() + pbegin, morph.pt3d.y.begin() + pend);
            new_pt3d.z.insert(new_pt3d.z.end(), morph.pt3d.z.begin() + pbegin, morph.pt3d.z.begin() + pend);
            new_pt3d.d.insert(new_pt3d.d.end(), morph.pt3d.d.begin() + pbegin, morph.pt3d.d.begin() + pend);
            new_pt3d.arc.insert(new_pt3d.arc.end(), morph.pt3d.arc.begin() + pbegin, morph.pt3d.arc.begin() + pend);
            sec.pt3d_offset = new_off;
            sec.pt3d_count = pend - pbegin;
        } else {
            sec.pt3d_offset = 0;
            sec.pt3d_count = 0;
            // Preserve sec.L_um for implicit cylinders (NEURON n3d==0).
        }
    }

    morph.sections = std::move(new_sections);
    morph.datas = std::move(new_datas);
    morph.pt3d = std::move(new_pt3d);
    morph.label_names = std::move(new_label_names);
    morph.label_index = std::move(new_label_index);
    morph.sections_by_label = std::move(new_sections_by_label);
    rebuild_section_children(morph);
}

}  // namespace

void rebuild_section_children(Morph& morph) {
    for (auto& sec : morph.sections) {
        sec.children.clear();
    }

    auto add_child_sorted = [&](section_id parent_section_id, section_id child_section_id) {
        if (parent_section_id == invalid_section_id) {
            return;
        }
        auto& children = morph.sections[parent_section_id].children;
        const double x = morph.sections[child_section_id].parentx;
        auto it = children.begin();
        for (; it != children.end(); ++it) {
            const double x0 = morph.sections[*it].parentx;
            if (x <= x0) {
                children.insert(it, child_section_id);
                return;
            }
        }
        children.push_back(child_section_id);
    };

    for (section_id child_section_id = 0; child_section_id < morph.sections.size(); ++child_section_id) {
        const auto parent_section_id = morph.sections[child_section_id].parent_sec_id;
        if (parent_section_id != invalid_section_id) {
            add_child_sorted(parent_section_id, child_section_id);
        }
    }
}

section_id append_section(Morph& morph, Section sec, std::span<const swcdata_id> point_ids) {
    sec.offset = morph.datas.size();
    morph.datas.insert(morph.datas.end(), point_ids.begin(), point_ids.end());
    sec.count = point_ids.size();

    sec.pt3d_offset = 0;
    sec.pt3d_count = 0;
    sec.L_um = 0.0;
    sec.children.clear();

    const auto new_section_id = static_cast<section_id>(morph.sections.size());
    if (sec.label.empty()) {
        throw std::runtime_error("section label name is empty");
    }
    morph.sections.push_back(std::move(sec));

    // Maintain label->section lookup incrementally for NEURON-style insertion planning.
    const auto label_u = ensure_section_label_index(morph, morph.sections[new_section_id].label);
    morph.sections_by_label[label_u].push_back(new_section_id);

    const auto parent_section_id = morph.sections[new_section_id].parent_sec_id;
    if (parent_section_id != invalid_section_id) {
        auto& child_section_ids = morph.sections[parent_section_id].children;
        const double x = morph.sections[new_section_id].parentx;
        auto it = child_section_ids.begin();
        for (; it != child_section_ids.end(); ++it) {
            const double x0 = morph.sections[*it].parentx;
            if (x <= x0) {
                child_section_ids.insert(it, new_section_id);
                return new_section_id;
            }
        }
        child_section_ids.push_back(new_section_id);
    }

    return new_section_id;
}

NeuronStylizedPt3d3 neuron_stylized_pts3(float start_x_um, double L_um, float diam_um) {

    const float x0 = start_x_um;
    const float x2 = static_cast<float>(static_cast<double>(x0) + L_um);
    // Match NEURON's define_shape stylized 3d: compute the middle x from rounded endpoints.
    const float x1 = static_cast<float>((static_cast<double>(x0) + static_cast<double>(x2)) * 0.5);
    const double arc1 = static_cast<double>(x1 - x0);
    return NeuronStylizedPt3d3{
        .arc_um = {0.0, arc1, L_um},
        .diam_um = {diam_um, diam_um, diam_um},
        .end_x_um = x2,
    };
}

section_id append_stylized_section(Morph& morph,
                                  const std::string& name,
                                  section_id parent_section_id,
                                  double parentx,
                                  std::string label,
                                  float start_x_um,
                                  double L_um,
                                  float diam_um) {
    const auto geom = neuron_stylized_pts3(start_x_um, L_um, diam_um);
    const std::array<float, 3> x_um = {start_x_um,
                                       static_cast<float>((static_cast<double>(start_x_um) + geom.end_x_um) * 0.5),
                                       geom.end_x_um};
    const std::array<float, 3> y_um = {0.0f, 0.0f, 0.0f};
    const std::array<float, 3> z_um = {0.0f, 0.0f, 0.0f};

    const auto head_id = static_cast<swcdata_id>(morph.swc.size());
    morph.swc.push_back(SwcData{
        .xyz = {static_cast<double>(start_x_um), 0.0, 0.0},
        .d_um = diam_um,
        .label = 0,
    });

    Section sec{};
    sec.name = name;
    sec.head_swc_id = head_id;
    sec.parent_swc_id = invalid_swcdata_id;
    sec.parent_sec_id = parent_section_id;
    sec.parentx = (parent_section_id == invalid_section_id) ? 1.0 : parentx;
    sec.wire_first = true;
    sec.label = std::move(label);
    sec.rallbranch = 1.0;
    // `nseg` is assigned later by an explicit nseg policy stage.
    sec.nseg = 0;

    const std::array<swcdata_id, 1> point_ids = {head_id};
    const auto new_section_id = append_section(morph, std::move(sec), std::span<const swcdata_id>(point_ids));
    set_section_pt3d(morph,
                     new_section_id,
                     std::span<const float>(x_um),
                     std::span<const float>(y_um),
                     std::span<const float>(z_um),
                     std::span<const double>(geom.arc_um),
                     std::span<const float>(geom.diam_um));
    return new_section_id;
}

section_id append_cylindrical_section_no_pt3d(Morph& morph,
                                             const std::string& name,
                                             section_id parent_section_id,
                                             double parentx,
                                             std::string label,
                                             double L_um,
                                             double diam_um) {
    const auto head_id = static_cast<swcdata_id>(morph.swc.size());
    morph.swc.push_back(SwcData{
        .xyz = {0.0, 0.0, 0.0},
        .d_um = diam_um,
        .label = 0,
    });

    Section sec{};
    sec.name = name;
    sec.head_swc_id = head_id;
    sec.parent_swc_id = invalid_swcdata_id;
    sec.parent_sec_id = parent_section_id;
    sec.parentx = (parent_section_id == invalid_section_id) ? 1.0 : parentx;
    sec.wire_first = true;
    sec.label = std::move(label);
    sec.rallbranch = 1.0;
    // `nseg` is assigned later by an explicit nseg policy stage.
    sec.nseg = 0;
    sec.diam_um = diam_um;

    const std::array<swcdata_id, 1> point_ids = {head_id};
    const auto new_section_id = append_section(morph, std::move(sec), std::span<const swcdata_id>(point_ids));
    // Preserve the caller-provided length; this section intentionally has no pt3d.
    morph.sections[static_cast<std::size_t>(new_section_id)].L_um = (L_um <= 1e-9) ? 1e-9 : L_um;
    return new_section_id;
}

void set_section_pt3d(Morph& morph,
                      section_id target_section_id,
                      std::span<const float> x_um,
                      std::span<const float> y_um,
                      std::span<const float> z_um,
                      std::span<const double> arc_um,
                      std::span<const float> diam_um) {
    const auto section_index = static_cast<std::size_t>(target_section_id);
    auto& sec = morph.sections[section_index];

    double prev = arc_um[0];
    for (std::size_t i = 1; i < arc_um.size(); ++i) {
        const double a = arc_um[i];
        prev = a;
    }

    const std::size_t offset = morph.pt3d.size();
    morph.pt3d.x.insert(morph.pt3d.x.end(), x_um.begin(), x_um.end());
    morph.pt3d.y.insert(morph.pt3d.y.end(), y_um.begin(), y_um.end());
    morph.pt3d.z.insert(morph.pt3d.z.end(), z_um.begin(), z_um.end());
    morph.pt3d.d.insert(morph.pt3d.d.end(), diam_um.begin(), diam_um.end());
    morph.pt3d.arc.insert(morph.pt3d.arc.end(), arc_um.begin(), arc_um.end());

    sec.pt3d_offset = offset;
    sec.pt3d_count = arc_um.size();
    const double L_um = arc_um.back();
    sec.L_um = (L_um <= 1e-9) ? 1e-9 : L_um;
}

section_id append_section_with_pt3d(Morph& morph,
                                    const std::string& name,
                                    section_id parent_section_id,
                                    double parentx,
                                    std::string label,
                                    std::span<const Pt3dPoint> points) {
    std::vector<double> arc_um(points.size());
    std::vector<float> diam_um(points.size());
    std::vector<float> x_um(points.size());
    std::vector<float> y_um(points.size());
    std::vector<float> z_um(points.size());

    arc_um[0] = 0.0;
    diam_um[0] = static_cast<float>(points[0].diam_um);
    x_um[0] = static_cast<float>(points[0].x_um);
    y_um[0] = static_cast<float>(points[0].y_um);
    z_um[0] = static_cast<float>(points[0].z_um);
    double acc = 0.0;
    for (std::size_t i = 1; i < points.size(); ++i) {
        const auto& p0 = points[i - 1];
        const auto& p1 = points[i];
        const double dx = (p1.x_um - p0.x_um);
        const double dy = (p1.y_um - p0.y_um);
        const double dz = (p1.z_um - p0.z_um);
        acc += std::sqrt(dx * dx + dy * dy + dz * dz);
        arc_um[i] = acc;
        diam_um[i] = static_cast<float>(p1.diam_um);
        x_um[i] = static_cast<float>(p1.x_um);
        y_um[i] = static_cast<float>(p1.y_um);
        z_um[i] = static_cast<float>(p1.z_um);
    }

    const auto head_id = static_cast<swcdata_id>(morph.swc.size());
    // Store the first point coordinates in SWC for debugging; node build uses pt3d directly.
    morph.swc.push_back(SwcData{
        .xyz = {points[0].x_um, points[0].y_um, points[0].z_um},
        .d_um = points[0].diam_um,
        .label = 0,
    });

    Section sec{};
    sec.name = name;
    sec.head_swc_id = head_id;
    sec.parent_swc_id = invalid_swcdata_id;
    sec.parent_sec_id = parent_section_id;
    sec.parentx = (parent_section_id == invalid_section_id) ? 1.0 : parentx;
    sec.wire_first = true;
    sec.label = std::move(label);
    sec.rallbranch = 1.0;
    // `nseg` is assigned later by an explicit nseg policy stage.
    sec.nseg = 0;

    const std::array<swcdata_id, 1> point_ids = {head_id};
    const auto new_section_id = append_section(morph, std::move(sec), std::span<const swcdata_id>(point_ids));
    set_section_pt3d(morph,
                     new_section_id,
                     std::span<const float>(x_um),
                     std::span<const float>(y_um),
                     std::span<const float>(z_um),
                     std::span<const double>(arc_um),
                     std::span<const float>(diam_um));
    return new_section_id;
}

void delete_section_subtree(Morph& morph, section_id subtree_root_section_id) {
    const auto subtree_root_index = static_cast<std::size_t>(subtree_root_section_id);

    // Mark all sections in the subtree.
    std::vector<std::uint8_t> marked_for_deletion(morph.sections.size(), 0);
    std::stack<section_id> dfs_stack;
    dfs_stack.push(subtree_root_section_id);
    marked_for_deletion[subtree_root_index] = 1;
    while (!dfs_stack.empty()) {
        const auto current_section_id = dfs_stack.top();
        dfs_stack.pop();
        const auto current_section_index = static_cast<std::size_t>(current_section_id);
        for (const auto child_section_id : morph.sections[current_section_index].children) {
            const auto child_section_index = static_cast<std::size_t>(child_section_id);
            if (marked_for_deletion[child_section_index]) {
                continue;
            }
            marked_for_deletion[child_section_index] = 1;
            dfs_stack.push(child_section_id);
        }
    }
    delete_marked_sections(morph, marked_for_deletion);
}

void delete_label(Morph& morph, std::string_view label) {
    std::vector<std::uint8_t> marked_for_deletion(morph.sections.size(), 0);
    std::stack<section_id> dfs_stack;
    for (std::size_t i = 0; i < morph.sections.size(); ++i) {
        if (morph.sections[i].label == label) {
            if (!marked_for_deletion[i]) {
                marked_for_deletion[i] = 1;
                dfs_stack.push(static_cast<section_id>(i));
            }
        }
    }
    if (dfs_stack.empty()) {
        return;
    }
    while (!dfs_stack.empty()) {
        const auto current_section_id = dfs_stack.top();
        dfs_stack.pop();
        const auto current_section_index = static_cast<std::size_t>(current_section_id);
        for (const auto child_section_id : morph.sections[current_section_index].children) {
            const auto child_section_index = static_cast<std::size_t>(child_section_id);
            if (marked_for_deletion[child_section_index]) {
                continue;
            }
            marked_for_deletion[child_section_index] = 1;
            dfs_stack.push(child_section_id);
        }
    }
    delete_marked_sections(morph, marked_for_deletion);
}

}  // namespace mind_micro_morph
