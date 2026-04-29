#include "morph/section_spec.hpp"

#include "morph/mutable_section.hpp"
#include "morph/section_to_node.hpp"
#include "label_utils.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mind_micro_frontend {
namespace {

struct SectionBuildPlan {
    std::vector<std::size_t> order{};
    std::unordered_map<std::string, std::size_t> name_to_spec{};
};

[[nodiscard]] std::vector<std::vector<std::size_t>> build_section_children(
    const std::vector<SectionSpec>& sections,
    const SectionBuildPlan& plan) {
    std::vector<std::vector<std::size_t>> out(sections.size());
    for (std::size_t child = 0; child < sections.size(); ++child) {
        const auto& parent_name = sections[child].parent_name;
        if (parent_name.empty()) {
            continue;
        }
        const auto parent_it = plan.name_to_spec.find(parent_name);
        if (parent_it == plan.name_to_spec.end()) {
            throw std::runtime_error("unknown parent section: " + parent_name);
        }
        out[parent_it->second].push_back(child);
    }
    return out;
}

void mark_section_subtree(std::size_t root,
                          const std::vector<std::vector<std::size_t>>& children,
                          std::vector<std::uint8_t>& marked_for_deletion) {
    if (root >= marked_for_deletion.size()) {
        throw std::runtime_error("section subtree root index out of range");
    }
    if (marked_for_deletion[root]) {
        return;
    }

    std::vector<std::size_t> stack;
    stack.push_back(root);
    marked_for_deletion[root] = 1;
    while (!stack.empty()) {
        const auto current = stack.back();
        stack.pop_back();
        for (const auto child : children[current]) {
            if (marked_for_deletion[child]) {
                continue;
            }
            marked_for_deletion[child] = 1;
            stack.push_back(child);
        }
    }
}

[[nodiscard]] std::vector<SectionSpec> filter_section_specs(
    const std::vector<SectionSpec>& sections,
    const SectionBuildPlan& plan,
    std::span<const std::uint8_t> marked_for_deletion) {
    std::vector<SectionSpec> out;
    out.reserve(sections.size());
    for (const auto idx : plan.order) {
        if (!marked_for_deletion[idx]) {
            out.push_back(sections[idx]);
        }
    }
    return out;
}

void fill_base_section_spec_from_morph(const mind_micro_morph::Morph& morph,
                                       std::size_t sec_u,
                                       SectionSpec& spec) {
    if (sec_u >= morph.sections.size()) {
        throw std::runtime_error("section index out of range while building SectionSpec");
    }
    const auto& sec = morph.sections[sec_u];
    spec.name = sec.name;
    if (sec.parent_sec_id != mind_micro_morph::invalid_section_id) {
        const auto parent_u = static_cast<std::size_t>(sec.parent_sec_id);
        if (parent_u >= morph.sections.size()) {
            throw std::runtime_error("section parent id out of range while building SectionSpec");
        }
        spec.parent_name = morph.sections[parent_u].name;
    }
    spec.parentx = sec.parentx;
    spec.label = sec.label;
    spec.nseg = sec.nseg;
    spec.L_um = sec.L_um;
    spec.diam_um = sec.diam_um;
}

[[nodiscard]] SectionBuildPlan prepare_section_build(const std::vector<SectionSpec>& sections) {
    SectionBuildPlan plan{};
    for (std::size_t i = 0; i < sections.size(); ++i) {
        const auto& sec = sections[i];
        if (sec.name.empty()) {
            throw std::runtime_error("section name must not be empty");
        }
        const auto [_, inserted] = plan.name_to_spec.emplace(sec.name, i);
        if (!inserted) {
            throw std::runtime_error("duplicate section name: " + sec.name);
        }
    }

    std::vector<std::uint8_t> state(sections.size(), 0);
    std::function<void(std::size_t)> dfs = [&](std::size_t idx) {
        if (state[idx] == 2) {
            return;
        }
        if (state[idx] == 1) {
            throw std::runtime_error("cyclic parent section reference at: " + sections[idx].name);
        }
        state[idx] = 1;
        const auto& sec = sections[idx];
        if (!sec.parent_name.empty()) {
            const auto it = plan.name_to_spec.find(sec.parent_name);
            if (it == plan.name_to_spec.end()) {
                throw std::runtime_error("unknown parent section: " + sec.parent_name);
            }
            dfs(it->second);
        }
        state[idx] = 2;
        plan.order.push_back(idx);
    };

    for (std::size_t i = 0; i < sections.size(); ++i) {
        dfs(i);
    }
    return plan;
}

[[nodiscard]] mind_micro_morph::section_id append_section_with_spec_pt3d(
    mind_micro_morph::Morph& morph,
    const SectionSpec& spec,
    mind_micro_morph::section_id parent) {
    const auto& first = spec.pt3d.front();

    const auto head_id = static_cast<mind_micro_morph::swcdata_id>(morph.swc.size());
    morph.swc.push_back(mind_micro_morph::SwcData{
        .xyz = {
            static_cast<double>(first.x_um),
            static_cast<double>(first.y_um),
            static_cast<double>(first.z_um),
        },
        .d_um = static_cast<double>(first.diam_um),
        .label = 0,
    });

    mind_micro_morph::Section sec{};
    sec.name = spec.name;
    sec.head_swc_id = head_id;
    sec.parent_swc_id = mind_micro_morph::invalid_swcdata_id;
    sec.parent_sec_id = parent;
    sec.parentx = (parent == mind_micro_morph::invalid_section_id) ? 1.0 : spec.parentx;
    sec.wire_first = true;
    sec.label = spec.label;
    sec.rallbranch = 1.0;
    sec.nseg = 0;

    const std::array<mind_micro_morph::swcdata_id, 1> point_ids = {head_id};
    const auto new_id = mind_micro_morph::append_section(
        morph,
        std::move(sec),
        std::span<const mind_micro_morph::swcdata_id>(point_ids.data(), point_ids.size()));

    const auto section_index = static_cast<std::size_t>(new_id);
    auto& out_sec = morph.sections[section_index];
    out_sec.pt3d_offset = morph.pt3d.size();
    out_sec.pt3d_count = spec.pt3d.size();

    double acc = 0.0;
    morph.pt3d.x.push_back(first.x_um);
    morph.pt3d.y.push_back(first.y_um);
    morph.pt3d.z.push_back(first.z_um);
    morph.pt3d.d.push_back(first.diam_um);
    morph.pt3d.arc.push_back(0.0);

    for (std::size_t i = 1; i < spec.pt3d.size(); ++i) {
        const auto& p0 = spec.pt3d[i - 1];
        const auto& p1 = spec.pt3d[i];
        const double dx = static_cast<double>(p1.x_um) - static_cast<double>(p0.x_um);
        const double dy = static_cast<double>(p1.y_um) - static_cast<double>(p0.y_um);
        const double dz = static_cast<double>(p1.z_um) - static_cast<double>(p0.z_um);
        acc += std::sqrt(dx * dx + dy * dy + dz * dz);
        morph.pt3d.x.push_back(p1.x_um);
        morph.pt3d.y.push_back(p1.y_um);
        morph.pt3d.z.push_back(p1.z_um);
        morph.pt3d.d.push_back(p1.diam_um);
        morph.pt3d.arc.push_back(acc);
    }
    out_sec.L_um = (acc <= 1e-9) ? 1e-9 : acc;
    return new_id;
}

}  // namespace

mind_micro_morph::Morph build_morph_from_sections(const std::vector<SectionSpec>& sections,
                                               SectionNseg& out_nseg) {
    const auto plan = prepare_section_build(sections);
    mind_micro_morph::Morph morph{};
    out_nseg.clear();

    std::unordered_map<std::string, mind_micro_morph::section_id> name_to_id;
    for (const auto idx : plan.order) {
        const auto& spec = sections[idx];
        const mind_micro_morph::section_id parent = spec.parent_name.empty()
                                                     ? mind_micro_morph::invalid_section_id
                                                     : name_to_id.find(spec.parent_name)->second;
        mind_micro_morph::section_id new_id = mind_micro_morph::invalid_section_id;
        if (!spec.pt3d.empty()) {
            new_id = append_section_with_spec_pt3d(morph, spec, parent);
        } else {
            new_id = mind_micro_morph::append_cylindrical_section_no_pt3d(
                morph,
                spec.name,
                parent,
                spec.parentx,
                spec.label,
                spec.L_um,
                spec.diam_um);
        }
        name_to_id.emplace(spec.name, new_id);
        out_nseg.push_back(spec.nseg);
    }
    return morph;
}

void apply_section_nseg(mind_micro_morph::Morph& morph, const SectionNseg& section_nseg) {
    if (section_nseg.empty()) {
        for (auto& sec : morph.sections) {
            sec.nseg = 1;
        }
        return;
    }
    for (std::size_t i = 0; i < morph.sections.size(); ++i) {
        morph.sections[i].nseg = section_nseg[i];
    }
}

std::vector<SectionSpec> section_specs_from_morph(const mind_micro_morph::Morph& morph) {
    std::vector<SectionSpec> out;
    for (mind_micro_morph::section_id sid = 0; sid < morph.sections.size(); ++sid) {
        const auto sec_u = static_cast<std::size_t>(sid);
        SectionSpec spec{};
        fill_base_section_spec_from_morph(morph, sec_u, spec);
        const auto& sec = morph.sections[sec_u];

        if (sec.pt3d_count <= 0) {
            out.push_back(std::move(spec));
            continue;
        }

        const auto begin = sec.pt3d_offset;
        const auto end = sec.pt3d_offset + sec.pt3d_count;
        if (end > morph.pt3d.x.size() || end > morph.pt3d.y.size() || end > morph.pt3d.z.size() ||
            end > morph.pt3d.d.size()) {
            throw std::runtime_error("section_specs_from_morph: pt3d range out of bounds");
        }

        for (std::size_t i = begin; i < end; ++i) {
            spec.pt3d.push_back(SectionPt3d{
                morph.pt3d.x[i],
                morph.pt3d.y[i],
                morph.pt3d.z[i],
                morph.pt3d.d[i],
            });
        }
        out.push_back(std::move(spec));
    }
    return out;
}

std::vector<SectionSpec> delete_subtree_specs(const std::vector<SectionSpec>& sections,
                                              std::string_view section_name) {
    const auto plan = prepare_section_build(sections);
    const auto target_it = plan.name_to_spec.find(std::string(section_name));
    if (target_it == plan.name_to_spec.end()) {
        throw std::runtime_error("delete_subtree: section not found: " + std::string(section_name));
    }

    const auto children = build_section_children(sections, plan);
    std::vector<std::uint8_t> marked_for_deletion(sections.size(), 0);
    mark_section_subtree(target_it->second, children, marked_for_deletion);
    return filter_section_specs(sections, plan, marked_for_deletion);
}

std::vector<SectionSpec> delete_label_specs(const std::vector<SectionSpec>& sections, std::string_view label) {
    const auto plan = prepare_section_build(sections);
    const auto children = build_section_children(sections, plan);
    std::vector<std::uint8_t> marked_for_deletion(sections.size(), 0);
    for (std::size_t i = 0; i < sections.size(); ++i) {
        if (sections[i].label == label) {
            mark_section_subtree(i, children, marked_for_deletion);
        }
    }
    return filter_section_specs(sections, plan, marked_for_deletion);
}

std::vector<SectionSpec> load_swc_sections(const std::string& swc_file) {
    mind_micro_morph::Morph morph = mind_micro_morph::load_swc_morphology(swc_file);
    mind_micro_morph::build_pt3d_geometry(morph);

    std::vector<SectionSpec> out;
    for (std::size_t i = 0; i < morph.sections.size(); ++i) {
        SectionSpec spec{};
        fill_base_section_spec_from_morph(morph, i, spec);
        const auto& sec = morph.sections[i];
        spec.nseg = 1;

        const bool implicit_cylinder = (sec.count == 1 && sec.L_um > 0.0 && sec.pt3d_count == 0);
        if (implicit_cylinder) {
            out.push_back(std::move(spec));
            continue;
        }

        const bool fix_first_diam =
            (!sec.wire_first && sec.parent_sec_id != mind_micro_morph::invalid_section_id &&
             mind_micro_labels::is_soma_label(morph.sections[sec.parent_sec_id].label) &&
             !mind_micro_labels::is_soma_label(sec.label));

        if (sec.count == 1) {
            const auto swc_id = morph.datas[sec.offset];
            const auto& s = morph.swc[static_cast<std::size_t>(swc_id)];
            const float d_um = static_cast<float>(s.d_um);
            const float y = static_cast<float>(s.xyz[1]);
            const float z = static_cast<float>(s.xyz[2]);
            const float half = 0.5f * d_um;
            const float x0 = static_cast<float>(s.xyz[0]) - half;
            const float x1 = static_cast<float>(s.xyz[0]);
            const float x2 = static_cast<float>(s.xyz[0]) + half;
            spec.pt3d.push_back(SectionPt3d{x0, y, z, d_um});
            spec.pt3d.push_back(SectionPt3d{x1, y, z, d_um});
            spec.pt3d.push_back(SectionPt3d{x2, y, z, d_um});
            out.push_back(std::move(spec));
            continue;
        }

        const auto second_id = morph.datas[sec.offset + 1];
        const float first_diam = static_cast<float>(morph.swc[static_cast<std::size_t>(second_id)].d_um);
        for (std::size_t j = 0; j < sec.count; ++j) {
            const auto swc_id = morph.datas[sec.offset + j];
            const auto& s = morph.swc[static_cast<std::size_t>(swc_id)];
            float d_um = static_cast<float>(s.d_um);
            if (fix_first_diam && j == 0) {
                d_um = first_diam;
            }
            spec.pt3d.push_back(SectionPt3d{
                static_cast<float>(s.xyz[0]),
                static_cast<float>(s.xyz[1]),
                static_cast<float>(s.xyz[2]),
                d_um,
            });
        }
        out.push_back(std::move(spec));
    }
    return out;
}

std::vector<SectionSpec> load_asc_sections(const std::string& asc_file) {
    mind_micro_morph::Morph morph = mind_micro_morph::load_asc_morphology(asc_file);

    std::vector<SectionSpec> out;
    for (std::size_t i = 0; i < morph.sections.size(); ++i) {
        SectionSpec spec{};
        fill_base_section_spec_from_morph(morph, i, spec);
        const auto& sec = morph.sections[i];
        spec.nseg = 1;

        if (sec.pt3d_count >= 2) {
            for (std::size_t j = 0; j < sec.pt3d_count; ++j) {
                const auto idx = sec.pt3d_offset + j;
                spec.pt3d.push_back(SectionPt3d{
                    morph.pt3d.x[idx],
                    morph.pt3d.y[idx],
                    morph.pt3d.z[idx],
                    morph.pt3d.d[idx],
                });
            }
        } else if (sec.pt3d_count == 1) {
            const auto idx = sec.pt3d_offset;
            const float d_um = morph.pt3d.d[idx];
            const float half = 0.5f * d_um;
            const float x = morph.pt3d.x[idx];
            const float y = morph.pt3d.y[idx];
            const float z = morph.pt3d.z[idx];
            spec.pt3d.push_back(SectionPt3d{x - half, y, z, d_um});
            spec.pt3d.push_back(SectionPt3d{x, y, z, d_um});
            spec.pt3d.push_back(SectionPt3d{x + half, y, z, d_um});
        } else {
            if (!(spec.L_um > 0.0) || !std::isfinite(spec.L_um)) {
                spec.L_um = 1.0;
            }
            if (!(spec.diam_um > 0.0) || !std::isfinite(spec.diam_um)) {
                spec.diam_um = 1.0;
            }
        }

        out.push_back(std::move(spec));
    }
    return out;
}

}  // namespace mind_micro_frontend
