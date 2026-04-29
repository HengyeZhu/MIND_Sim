#include "section_distance.hpp"
#include "section_to_node.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mind_micro_morph {
namespace {

[[nodiscard]] int section_depth(const SectionDistanceLayout& layout,
                                std::size_t sec_idx,
                                std::vector<int>& depth_cache) {
    if (sec_idx >= depth_cache.size()) {
        return -1;
    }
    if (depth_cache[sec_idx] >= 0) {
        return depth_cache[sec_idx];
    }

    std::vector<std::size_t> chain;
    std::int32_t current = static_cast<std::int32_t>(sec_idx);
    while (current >= 0) {
        const auto cur_sec = static_cast<std::size_t>(current);
        if (cur_sec >= depth_cache.size()) {
            return -1;
        }
        if (depth_cache[cur_sec] >= 0) {
            break;
        }
        chain.push_back(cur_sec);
        if (chain.size() > depth_cache.size()) {
            return -1;
        }
        current = layout.section_parent_section_id[cur_sec];
    }

    int depth = -1;
    if (current >= 0) {
        const auto cur_sec = static_cast<std::size_t>(current);
        depth = depth_cache[cur_sec];
        if (depth < 0) {
            return -1;
        }
    }
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        depth += 1;
        depth_cache[*it] = depth;
    }
    return depth_cache[sec_idx];
}

[[nodiscard]] double node_dist_um(const SectionDistanceLayout& layout,
                                  std::size_t sec_idx,
                                  std::int32_t template_node) {
    if (sec_idx >= layout.section_nseg.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const int nseg = layout.section_nseg[sec_idx];
    if (nseg <= 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    if (template_node == layout.section_parent_node_id[sec_idx]) {
        return 0.0;
    }

    const double L_um = layout.section_length_um[sec_idx];
    const double safe_length_um = (std::isfinite(L_um) && L_um > 0.0) ? L_um : 0.0;
    const int base = layout.section_node_base_id[sec_idx];
    const int inode = static_cast<int>(template_node - base);
    if (inode == nseg) {
        return safe_length_um;
    }
    if (inode < 0 || inode >= nseg) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double ratio = (static_cast<double>(inode) + 0.5) / static_cast<double>(nseg);
    return safe_length_um * ratio;
}

struct DiamInterpState {
    std::size_t pt_index{0};
    double prev_arc_um{0.0};
    double prev_diam_um{0.0};
    double node_step_um{0.0};
};

[[nodiscard]] double segment_diam_um_from_pt3d_neuron_style(const SectionDistanceLayout& layout,
                                                            std::size_t p_offset,
                                                            std::size_t p_count,
                                                            std::int32_t nseg,
                                                            std::int32_t seg_index) {
    if (p_count < 2 || nseg <= 0 || seg_index < 0 || seg_index >= nseg) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    if (p_offset + p_count > layout.pt3d_arc_um.size() || p_offset + p_count > layout.pt3d_diam_um.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const double arc0 = layout.pt3d_arc_um[p_offset];
    const double arc_last = layout.pt3d_arc_um[p_offset + p_count - 1];
    const double total = arc_last - arc0;
    if (!std::isfinite(total) || total <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    auto arc_rel = [&](std::size_t local_idx) -> double {
        return layout.pt3d_arc_um[p_offset + local_idx] - arc0;
    };
    auto diam_abs = [&](std::size_t local_idx) -> double {
        return std::fabs(layout.pt3d_diam_um[p_offset + local_idx]);
    };

    const auto nnode = static_cast<std::int32_t>(nseg + 1);
    DiamInterpState st{};
    double seg_diam_um = std::numeric_limits<double>::quiet_NaN();

    for (std::int32_t inode = 0; inode <= seg_index; ++inode) {
        if (inode == 0) {
            st.pt_index = 0;
            st.prev_arc_um = arc_rel(0);
            st.prev_diam_um = diam_abs(0);
            st.node_step_um = arc_rel(p_count - 1) / static_cast<double>(nnode - 1);
        }

        double arc_start_um = static_cast<double>(inode) * st.node_step_um;
        double diam_integral = 0.0;

        // Match section_to_node.cpp::diam_from_list_segment traversal.
        for (std::int32_t half = 0; half < 2; ++half) {
            const double arc_mid_um = arc_start_um + st.node_step_um / 2.0;
            for (;;) {
                const std::size_t next_pt = st.pt_index + 1;
                const double arc_j_um = arc_rel(st.pt_index);
                const double arc_jp_um = arc_rel(next_pt);

                double arc_next_um = 0.0;
                double diam_next_um = 0.0;
                std::size_t next_index = st.pt_index;

                if (arc_jp_um > arc_mid_um || next_pt == p_count - 1) {
                    const double denom = arc_jp_um - arc_j_um;
                    const double frac =
                        (std::fabs(denom) < 1e-10) ? 1.0 : ((arc_mid_um - arc_j_um) / denom);
                    arc_next_um = arc_mid_um;
                    diam_next_um = (1.0 - frac) * diam_abs(st.pt_index) + frac * diam_abs(next_pt);
                    next_index = st.pt_index;
                } else {
                    arc_next_um = arc_jp_um;
                    diam_next_um = diam_abs(next_pt);
                    next_index = next_pt;
                }

                const double raw_delta_um = (arc_next_um - st.prev_arc_um);
                diam_integral += (diam_next_um + st.prev_diam_um) * raw_delta_um;

                st.prev_arc_um = arc_next_um;
                st.prev_diam_um = diam_next_um;
                if (st.pt_index == next_index) {
                    break;
                }
                st.pt_index = next_index;
            }
            arc_start_um = arc_mid_um;
        }

        seg_diam_um = (st.node_step_um == 0.0) ? 0.0 : (diam_integral * 0.5 / st.node_step_um);
    }

    return seg_diam_um;
}

bool resolve_diam_um_by_loc_internal(const SectionDistanceLayout& layout,
                                     std::size_t sec_idx,
                                     double loc,
                                     double* out_diam_um) {
    if (out_diam_um == nullptr || !std::isfinite(loc) || !layout.valid) {
        return false;
    }
    const auto default_diam =
        (sec_idx < layout.section_diam_um.size() && std::isfinite(layout.section_diam_um[sec_idx]) &&
         layout.section_diam_um[sec_idx] > 0.0)
            ? layout.section_diam_um[sec_idx]
            : 1.0;
    if (sec_idx >= layout.section_nseg.size()) {
        *out_diam_um = default_diam;
        return true;
    }
    const std::int32_t nseg = layout.section_nseg[sec_idx];
    if (nseg <= 0) {
        *out_diam_um = default_diam;
        return true;
    }

    if (sec_idx >= layout.section_pt3d_offset.size() || sec_idx >= layout.section_pt3d_count.size()) {
        *out_diam_um = default_diam;
        return true;
    }

    const int p_offset_i = layout.section_pt3d_offset[sec_idx];
    const int p_count_i = layout.section_pt3d_count[sec_idx];
    if (p_offset_i < 0 || p_count_i <= 0) {
        *out_diam_um = default_diam;
        return true;
    }

    const auto p_offset = static_cast<std::size_t>(p_offset_i);
    const auto p_count = static_cast<std::size_t>(p_count_i);
    if (p_offset + p_count > layout.pt3d_arc_um.size() || p_offset + p_count > layout.pt3d_diam_um.size()) {
        *out_diam_um = default_diam;
        return true;
    }

    if (p_count == 1) {
        *out_diam_um =
            (std::isfinite(layout.pt3d_diam_um[p_offset]) && layout.pt3d_diam_um[p_offset] > 0.0)
                ? layout.pt3d_diam_um[p_offset]
                : 1.0;
        return true;
    }

    std::int32_t seg_idx = 0;
    if (!resolve_segment_index_for_nseg(nseg, loc, &seg_idx)) {
        *out_diam_um = default_diam;
        return true;
    }

    const double seg_diam = segment_diam_um_from_pt3d_neuron_style(layout, p_offset, p_count, nseg, seg_idx);
    if (!std::isfinite(seg_diam)) {
        *out_diam_um = default_diam;
        return true;
    }
    *out_diam_um = (std::isfinite(seg_diam) && seg_diam > 0.0) ? seg_diam : 1.0;
    return true;
}

bool resolve_section_location_by_index_internal(const SectionDistanceLayout& layout,
                                                std::size_t sec_idx,
                                                double loc,
                                                SectionLocationResolveInfo* out) {
    if (out == nullptr || !layout.valid || !std::isfinite(loc)) {
        return false;
    }
    if (sec_idx >= layout.section_nseg.size() ||
        sec_idx >= layout.section_node_base_id.size() ||
        sec_idx >= layout.section_parent_node_id.size()) {
        return false;
    }

    std::int32_t segment_index = -1;
    if (!resolve_segment_index_for_nseg(layout.section_nseg[sec_idx], loc, &segment_index)) {
        return false;
    }

    std::int32_t template_node = -1;
    if (!resolve_template_node_by_section_loc(layout.section_node_base_id,
                                              layout.section_nseg,
                                              layout.section_parent_node_id,
                                              sec_idx,
                                              loc,
                                              &template_node)) {
        return false;
    }

    double diam_um = std::numeric_limits<double>::quiet_NaN();
    if (!resolve_diam_um_by_loc_internal(layout, sec_idx, loc, &diam_um) || !std::isfinite(diam_um)) {
        return false;
    }

    *out = SectionLocationResolveInfo{
        .section_index = sec_idx,
        .segment_index = segment_index,
        .template_node = template_node,
        .requested_loc = loc,
        .diam_um = diam_um,
    };
    return true;
}

}  // namespace

SectionDistanceLayout build_section_distance_layout(const Morph& morph) {
    if (morph.sections.empty()) {
        throw std::runtime_error("build_section_distance_layout requires non-empty sections");
    }

    SectionDistanceLayout out{};
    out.section_parent_section_id = std::vector<std::int32_t>(morph.sections.size(), -1);
    out.section_length_um = std::vector<double>(morph.sections.size(), 0.0);
    out.section_diam_um = std::vector<double>(morph.sections.size(), 0.0);
    out.section_pt3d_offset = std::vector<std::int32_t>(morph.sections.size(), -1);
    out.section_pt3d_count = std::vector<std::int32_t>(morph.sections.size(), 0);

    for (section_id sid = 0; sid < morph.sections.size(); ++sid) {
        const auto sec_index = static_cast<std::size_t>(sid);
        const auto& sec = morph.sections[sec_index];
        out.section_name_to_index.emplace(sec.name, sec_index);
        if (sec.parent_sec_id != invalid_section_id) {
            out.section_parent_section_id[sec_index] = static_cast<std::int32_t>(sec.parent_sec_id);
        }

        const double length_um = sec.L_um;
        out.section_length_um[sec_index] = (std::isfinite(length_um) && length_um > 0.0) ? length_um : 0.0;

        const double diam_um = sec.diam_um;
        out.section_diam_um[sec_index] = (std::isfinite(diam_um) && diam_um > 0.0) ? diam_um : 0.0;

        if (sec.pt3d_count <= 0) {
            continue;
        }

        const auto p_offset = sec.pt3d_offset;
        const auto p_count = sec.pt3d_count;
        if (p_offset + p_count > morph.pt3d.arc.size() || p_offset + p_count > morph.pt3d.d.size()) {
            throw std::runtime_error("build_section_distance_layout: pt3d range out of bounds for section '" +
                                     sec.name + "'");
        }

        const auto out_offset = static_cast<std::int32_t>(out.pt3d_arc_um.size());
        out.section_pt3d_offset[sec_index] = out_offset;
        out.section_pt3d_count[sec_index] = static_cast<std::int32_t>(p_count);
        for (std::size_t k = 0; k < p_count; ++k) {
            const auto idx = p_offset + k;
            out.pt3d_arc_um.push_back(morph.pt3d.arc[idx]);
            out.pt3d_diam_um.push_back(static_cast<double>(morph.pt3d.d[idx]));
        }
    }

    NodeBuildConfig cfg{};
    cfg.force_single_root = true;
    cfg.min_diam_um = 1e-6;
    auto built = build_nodes_neuron_compatible_with_layout(morph, cfg);
    out.section_node_base_id = std::move(built.layout.section_node_base_id);
    out.section_nseg = std::move(built.layout.section_nseg);
    out.section_parent_node_id = std::move(built.layout.section_parent_node_id);

    const auto nsec = out.section_nseg.size();
    if (nsec == 0) {
        throw std::runtime_error("build_section_distance_layout: empty node layout");
    }
    if (out.section_node_base_id.size() != nsec ||
        out.section_parent_node_id.size() != nsec ||
        out.section_parent_section_id.size() != nsec ||
        out.section_length_um.size() != nsec) {
        throw std::runtime_error("build_section_distance_layout: inconsistent section layout vectors");
    }
    for (const auto& [_, sec_idx] : out.section_name_to_index) {
        if (sec_idx >= nsec) {
            throw std::runtime_error("build_section_distance_layout: section name index out of range");
        }
    }

    out.valid = true;
    return out;
}

bool resolve_section_location(const SectionDistanceLayout& layout,
                              const std::string& section_name,
                              double loc,
                              SectionLocationResolveInfo* out) {
    if (out == nullptr || !layout.valid) {
        return false;
    }
    *out = SectionLocationResolveInfo{};
    const auto sec_it = layout.section_name_to_index.find(section_name);
    if (sec_it == layout.section_name_to_index.end()) {
        return false;
    }
    return resolve_section_location_by_index_internal(layout, sec_it->second, loc, out);
}

bool resolve_segment_index_for_nseg(std::int32_t nseg, double loc, std::int32_t* out_segment_index) {
    if (out_segment_index == nullptr || !std::isfinite(loc) || nseg <= 0) {
        return false;
    }
    if (loc < 0.0 || loc > 1.0) {
        return false;
    }
    int seg = static_cast<int>(std::floor(loc * static_cast<double>(nseg)));
    if (seg == static_cast<int>(nseg)) {
        seg = static_cast<int>(nseg) - 1;
    }
    seg = std::clamp(seg, 0, static_cast<int>(nseg) - 1);
    *out_segment_index = static_cast<std::int32_t>(seg);
    return true;
}

double distance_between_points_um(const SectionDistanceLayout& layout,
                                  const SectionDistancePoint& p1,
                                  const SectionDistancePoint& p2) {
    if (!layout.valid) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const std::size_t nsec = layout.section_nseg.size();
    if (p1.section_index >= nsec || p2.section_index >= nsec ||
        p1.template_node < 0 || p2.template_node < 0) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    int sec1 = static_cast<int>(p1.section_index);
    int sec2 = static_cast<int>(p2.section_index);
    std::int32_t node1 = p1.template_node;
    std::int32_t node2 = p2.template_node;
    double distance_um = 0.0;

    std::vector<int> depth_cache(nsec, -1);

    while (sec1 != sec2) {
        if (sec1 < 0) {
            const auto sec2_idx = static_cast<std::size_t>(sec2);
            const double step = node_dist_um(layout, sec2_idx, node2);
            if (!std::isfinite(step)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            distance_um += step;
            node2 = layout.section_parent_node_id[sec2_idx];
            sec2 = layout.section_parent_section_id[sec2_idx];
            continue;
        }
        if (sec2 < 0) {
            const auto sec1_idx = static_cast<std::size_t>(sec1);
            const double step = node_dist_um(layout, sec1_idx, node1);
            if (!std::isfinite(step)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            distance_um += step;
            node1 = layout.section_parent_node_id[sec1_idx];
            sec1 = layout.section_parent_section_id[sec1_idx];
            continue;
        }

        const auto sec1_idx = static_cast<std::size_t>(sec1);
        const auto sec2_idx = static_cast<std::size_t>(sec2);
        const int depth1 = section_depth(layout, sec1_idx, depth_cache);
        const int depth2 = section_depth(layout, sec2_idx, depth_cache);
        if (depth1 < 0 || depth2 < 0) {
            return std::numeric_limits<double>::quiet_NaN();
        }

        if (depth1 > depth2) {
            const double step = node_dist_um(layout, sec1_idx, node1);
            if (!std::isfinite(step)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            distance_um += step;
            node1 = layout.section_parent_node_id[sec1_idx];
            sec1 = layout.section_parent_section_id[sec1_idx];
        } else {
            const double step = node_dist_um(layout, sec2_idx, node2);
            if (!std::isfinite(step)) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            distance_um += step;
            node2 = layout.section_parent_node_id[sec2_idx];
            sec2 = layout.section_parent_section_id[sec2_idx];
        }
    }

    if (sec1 < 0) {
        if (node1 != node2) {
            return kDisconnectedDistanceUm;
        }
        return distance_um;
    }

    if (node1 != node2) {
        const auto common_sec = static_cast<std::size_t>(sec1);
        const double x1 = node_dist_um(layout, common_sec, node1);
        const double x2 = node_dist_um(layout, common_sec, node2);
        if (!std::isfinite(x1) || !std::isfinite(x2)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        distance_um += std::fabs(x1 - x2);
    }
    return distance_um;
}

double distance_between_locs_um(const SectionDistanceLayout& layout,
                                const std::string& section_name1,
                                double loc1,
                                const std::string& section_name2,
                                double loc2) {
    SectionLocationResolveInfo p1{};
    SectionLocationResolveInfo p2{};
    if (!resolve_section_location(layout, section_name1, loc1, &p1) ||
        !resolve_section_location(layout, section_name2, loc2, &p2)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return distance_between_points_um(
        layout,
        SectionDistancePoint{.section_index = p1.section_index, .template_node = p1.template_node},
        SectionDistancePoint{.section_index = p2.section_index, .template_node = p2.template_node});
}

}  // namespace mind_micro_morph
