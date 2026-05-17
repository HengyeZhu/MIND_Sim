// Module 3/4: Section geometry + Section->Node lowering (pt3d/arc + area/ri/a/b)

#include "section_to_node.hpp"

#include "label_utils.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <numbers>
#include <span>
#include <stdexcept>
#include <vector>

namespace mind_micro_morph {
namespace {

inline constexpr double pi = std::numbers::pi_v<double>;
inline constexpr double arc_interp_denom_eps = 1e-10;
inline constexpr double min_positive_delta_um = 1e-15;
inline constexpr double huge_resistance_Mohm = 1e30;
inline constexpr double infinite_resistance_cutoff_Mohm = 1e29;
inline constexpr double ri_scale = (4.0 * 0.01);

struct SegmentGeom {
    double area_um2{};
    double rinv_uS{};
    double ri_Mohm{};
    double diam_um{};
    double rright_Mohm{};
};

struct DiamFromListInterpState {
    // Current pt3d sample index used by the piecewise-linear interpolation.
    std::size_t pt_index{};
    // Previous interpolation point on the arclength axis and its diameter.
    double prev_arc_um{};
    double prev_diam_um{};
    // Uniform node spacing along arc (um): L_um/(nnode-1).
    double node_step_um{};
};

SegmentGeom diam_from_list_segment(std::span<const double> arc_um,
                                   std::span<const float> diam_um,
                                   std::int32_t inode,
                                   std::int32_t nnode,
                                   double rparent_Mohm,
                                   DiamFromListInterpState& st) {
    const std::size_t npt = arc_um.size();
    if (inode == 0) {
        st.pt_index = 0;
        st.prev_arc_um = arc_um[0];
        st.prev_diam_um = std::fabs(static_cast<double>(diam_um[0]));
        st.node_step_um = arc_um[npt - 1] / static_cast<double>(nnode - 1);
    }

    auto diam_abs = [&](std::size_t pt_index) -> double {
        return std::fabs(static_cast<double>(diam_um[pt_index]));
    };

    double arc_start_um = static_cast<double>(inode) * st.node_step_um;
    double diam_integral = 0.0;
    double area_integral = 0.0;

    double rleft_Mohm = 0.0;
    double rright_Mohm = 0.0;

    // Integrate the left and right half of the segment separately (NEURON-style),
    // advancing a piecewise-linear diameter interpolation along the arc axis.
    for (std::int32_t half = 0; half < 2; ++half) {
        double ri_integral = 0.0;
        const double arc_mid_um = arc_start_um + st.node_step_um / 2.0;

        for (;;) {
            // Invariant: st.pt_index always points to the left sample, so st.pt_index+1 is valid.
            const std::size_t next_pt = st.pt_index + 1;
            const double arc_j_um = arc_um[st.pt_index];
            const double arc_jp_um = arc_um[next_pt];

            double arc_next_um{};
            double diam_next_um{};
            std::size_t next_index{};

            if (arc_jp_um > arc_mid_um || next_pt == npt - 1) {
                const double denom = arc_jp_um - arc_j_um;
                const double frac =
                    (std::fabs(denom) < arc_interp_denom_eps) ? 1.0 : ((arc_mid_um - arc_j_um) / denom);
                arc_next_um = arc_mid_um;
                diam_next_um = (1.0 - frac) * diam_abs(st.pt_index) + frac * diam_abs(next_pt);
                next_index = st.pt_index;  // stop once we reach the half-step position
            } else {
                arc_next_um = arc_jp_um;
                diam_next_um = diam_abs(next_pt);
                next_index = next_pt;  // keep walking forward through samples
            }

            const double raw_delta_um = (arc_next_um - st.prev_arc_um);
            diam_integral += (diam_next_um + st.prev_diam_um) * raw_delta_um;

            double delta_um = raw_delta_um;
            if (delta_um < min_positive_delta_um) {
                delta_um = min_positive_delta_um;
            }

            double inv_cond_term = diam_next_um * st.prev_diam_um / delta_um;
            if (inv_cond_term == 0.0) {
                inv_cond_term = min_positive_delta_um;
            }
            ri_integral += 1.0 / inv_cond_term;

            double tmp = 0.5 * (diam_next_um - st.prev_diam_um);
            tmp = std::sqrt(delta_um * delta_um + tmp * tmp);
            area_integral += (diam_next_um + st.prev_diam_um) * tmp;

            st.prev_arc_um = arc_next_um;
            st.prev_diam_um = diam_next_um;
            if (st.pt_index == next_index) {
                break;
            }
            st.pt_index = next_index;
        }

        const double half_Mohm = ri_integral * 1.0 / pi * ri_scale;
        if (half == 0) {
            rleft_Mohm = half_Mohm;
        } else {
            rright_Mohm = half_Mohm;
        }
        arc_start_um = arc_mid_um;
    }

    // Match NEURON + our Python construction exactly:
    //  - store ri with NEURON ri() semantics (computed via 1/rinv with rounding)
    //  - store rinv as Python does (computed as 1/ri with rounding)
    const double ri_total = rparent_Mohm + rleft_Mohm;
    const double rinv_true_uS = (ri_total == 0.0) ? 0.0 : (1.0 / ri_total);
    const double ri_Mohm = (rinv_true_uS == 0.0) ? huge_resistance_Mohm : (1.0 / rinv_true_uS);
    const double rinv_uS = (ri_Mohm >= infinite_resistance_cutoff_Mohm) ? 0.0 : (1.0 / ri_Mohm);

    const double seg_diam_um =
        (st.node_step_um == 0.0) ? 0.0 : (diam_integral * 0.5 / st.node_step_um);
    const double area_um2 = area_integral * 0.5 * pi;

    return SegmentGeom{
        .area_um2 = area_um2,
        .rinv_uS = rinv_uS,
        .ri_Mohm = ri_Mohm,
        .diam_um = seg_diam_um,
        .rright_Mohm = rright_Mohm,
    };
}

struct SectionCableData {
    std::int32_t nseg{};
    std::vector<double> area{};
    std::vector<double> rinv{};
    std::vector<double> ri{};
    std::vector<double> diam{};
};

SectionCableData compute_section_cable_data(const Morph& morph, const Section& sec) {
    const std::int32_t nnode = sec.nseg + 1;
    SectionCableData out{};
    out.nseg = sec.nseg;

    // Support NEURON-style "n3d==0" sections (no pt3d): treat them as uniform cylinders
    // with length `sec.L_um` and constant diameter stored on the dummy head SWC point.
    if (sec.pt3d_count < 2) {
        double diam_um = std::fabs(sec.diam_um);
        if (!(diam_um > 0.0)) {
            const auto head_index = static_cast<std::size_t>(sec.head_swc_id);
            diam_um = std::fabs(static_cast<double>(morph.swc[head_index].d_um));
        }

        const double L_um = sec.L_um;
        const double dx_um = L_um / static_cast<double>(sec.nseg);

        out.area = std::vector<double>(static_cast<std::size_t>(nnode), 100.0);
        out.rinv = std::vector<double>(static_cast<std::size_t>(nnode), 0.0);
        out.ri = std::vector<double>(static_cast<std::size_t>(nnode), huge_resistance_Mohm);
        out.diam = std::vector<double>(static_cast<std::size_t>(sec.nseg), diam_um);

        // Uniform cylinder: each segment has constant diameter and length dx.
        const double seg_area_um2 = pi * diam_um * dx_um;
        const double rhalf_Mohm = 1.0e-2 * 1.0 * (dx_um / 2.0) /
                                  (pi * diam_um * diam_um / 4.0);

        double rparent = 0.0;
        for (std::int32_t inode = 0; inode < sec.nseg; ++inode) {
            const double ri_total = rparent + rhalf_Mohm;
            const double rinv_true_uS = (ri_total == 0.0) ? 0.0 : (1.0 / ri_total);
            const double ri_Mohm = (rinv_true_uS == 0.0) ? huge_resistance_Mohm : (1.0 / rinv_true_uS);
            const double rinv_uS = (ri_Mohm >= infinite_resistance_cutoff_Mohm) ? 0.0 : (1.0 / ri_Mohm);

            out.area[static_cast<std::size_t>(inode)] = seg_area_um2;
            out.rinv[static_cast<std::size_t>(inode)] = rinv_uS;
            out.ri[static_cast<std::size_t>(inode)] = ri_Mohm;
            rparent = rhalf_Mohm;
        }

        // Boundary node: area stays at 100; rinv and ri follow nrn_area_ri.
        const auto boundary = static_cast<std::size_t>(sec.nseg);
        const double rinv_true = (rparent == 0.0) ? 0.0 : (1.0 / rparent);
        out.ri[boundary] = (rinv_true == 0.0) ? huge_resistance_Mohm : (1.0 / rinv_true);
        out.rinv[boundary] = (out.ri[boundary] >= infinite_resistance_cutoff_Mohm) ? 0.0 : (1.0 / out.ri[boundary]);
        return out;
    }

    auto arc_um = std::span<const double>(morph.pt3d.arc).subspan(sec.pt3d_offset, sec.pt3d_count);
    auto diam_um = std::span<const float>(morph.pt3d.d).subspan(sec.pt3d_offset, sec.pt3d_count);

    out.area = std::vector<double>(static_cast<std::size_t>(nnode), 100.0);
    out.rinv = std::vector<double>(static_cast<std::size_t>(nnode), 0.0);
    out.ri = std::vector<double>(static_cast<std::size_t>(nnode), 1e30);
    out.diam = std::vector<double>(static_cast<std::size_t>(sec.nseg), 0.0);

    double rparent = 0.0;
    DiamFromListInterpState st{};
    for (std::int32_t inode = 0; inode < sec.nseg; ++inode) {
        const auto seg = diam_from_list_segment(arc_um, diam_um, inode, nnode, rparent, st);
        out.area[static_cast<std::size_t>(inode)] = seg.area_um2;
        out.rinv[static_cast<std::size_t>(inode)] = seg.rinv_uS;
        out.ri[static_cast<std::size_t>(inode)] = seg.ri_Mohm;
        out.diam[static_cast<std::size_t>(inode)] = seg.diam_um;
        rparent = seg.rright_Mohm;
    }

    // Boundary node: area stays at 100; rinv and ri follow nrn_area_ri.
    const auto boundary = static_cast<std::size_t>(sec.nseg);
    const double rinv_true = (rparent == 0.0) ? 0.0 : (1.0 / rparent);
    out.ri[boundary] = (rinv_true == 0.0) ? 1e30 : (1.0 / rinv_true);
    out.rinv[boundary] = (out.ri[boundary] >= 1.0e29) ? 0.0 : (1.0 / out.ri[boundary]);

    return out;
}

}  // namespace

void build_pt3d_geometry(Morph& morph) {
    // Build pt3d geometry for sections that don't already have it.
    //
    // Important: some higher-level morph editing APIs (e.g. delete_axon_BPO / stylized sections,
    // subtree copy, etc.) attach explicit pt3d to newly created sections and may use a single
    // dummy SWC point (sec.count==1) to satisfy internal datas invariants. In that case we must
    // NOT clobber the existing pt3d by regenerating from SWC (which would treat count==1 as a
    // sphere and collapse the intended cable length to the diameter).

    for (section_id current_section_id = 0; current_section_id < morph.sections.size(); ++current_section_id) {
        auto& sec = morph.sections[current_section_id];

        if (sec.pt3d_count != 0) {
            if (sec.L_um <= 0.0 && sec.pt3d_count >= 2) {
                const double last = morph.pt3d.arc[sec.pt3d_offset + sec.pt3d_count - 1];
                sec.L_um = (last <= 1e-9) ? 1e-9 : last;
            }
            continue;
        }

        // Allow "implicit cylinder" sections (NEURON n3d==0): keep the caller-provided L_um
        // and do not synthesize pt3d from the dummy SWC point stream.
        if (sec.count == 1 && sec.L_um > 0.0) {
            continue;
        }

        sec.pt3d_offset = morph.pt3d.size();
        sec.pt3d_count = 0;
        sec.L_um = 0.0;

        if (sec.count == 0) {
            continue;
        }

        const bool fix_first_diam =
            (!sec.wire_first && sec.parent_sec_id != invalid_section_id &&
             mind_micro_labels::is_soma_label(morph.sections[sec.parent_sec_id].label) &&
             !mind_micro_labels::is_soma_label(sec.label));

        double arc = 0.0;

        // Sphere replacement: a single point becomes 3 points, with x shifted by ±d/2.
        // Note: NEURON stores pt3d coordinates as float and accumulates arclength from
        // float-coordinate deltas in double.
        if (sec.count == 1) {
            const auto swc_id = morph.datas[sec.offset];
            const auto& s = morph.swc[static_cast<std::size_t>(swc_id)];

            const double d_um = s.d_um;
            const double xd = s.xyz[0];
            const double yd = s.xyz[1];
            const double zd = s.xyz[2];
            const double half = d_um * 0.5;

            const float x0 = static_cast<float>(xd - half);
            const float x1 = static_cast<float>(xd);
            const float x2 = static_cast<float>(xd + half);
            const float y = static_cast<float>(yd);
            const float z = static_cast<float>(zd);

            morph.pt3d.x.push_back(x0);
            morph.pt3d.y.push_back(y);
            morph.pt3d.z.push_back(z);
            morph.pt3d.d.push_back(static_cast<float>(d_um));
            morph.pt3d.arc.push_back(arc);  // 0

            auto step = [&](float x, float y, float z, float& prev_x, float& prev_y, float& prev_z) {
                const double dx = static_cast<double>(x - prev_x);
                const double dy = static_cast<double>(y - prev_y);
                const double dz = static_cast<double>(z - prev_z);
                arc += std::sqrt(dx * dx + dy * dy + dz * dz);
                prev_x = x;
                prev_y = y;
                prev_z = z;
                morph.pt3d.x.push_back(x);
                morph.pt3d.y.push_back(y);
                morph.pt3d.z.push_back(z);
                morph.pt3d.d.push_back(static_cast<float>(d_um));
                morph.pt3d.arc.push_back(arc);
            };

            float prev_x = x0;
            float prev_y = y;
            float prev_z = z;
            step(x1, y, z, prev_x, prev_y, prev_z);
            step(x2, y, z, prev_x, prev_y, prev_z);

            sec.pt3d_count = 3;
            sec.L_um = (arc <= 1e-9) ? 1e-9 : arc;
            continue;
        }

        // sec.count >= 2
        const auto swc_id0 = morph.datas[sec.offset];
        const auto swc_id1 = morph.datas[sec.offset + 1];
        const auto& s0 = morph.swc[static_cast<std::size_t>(swc_id0)];
        const auto& s1 = morph.swc[static_cast<std::size_t>(swc_id1)];

        float d0 = static_cast<float>(s0.d_um);
        if (fix_first_diam) {
            d0 = static_cast<float>(s1.d_um);
        }

        float prev_x = static_cast<float>(s0.xyz[0]);
        float prev_y = static_cast<float>(s0.xyz[1]);
        float prev_z = static_cast<float>(s0.xyz[2]);

        morph.pt3d.x.push_back(prev_x);
        morph.pt3d.y.push_back(prev_y);
        morph.pt3d.z.push_back(prev_z);
        morph.pt3d.d.push_back(d0);
        morph.pt3d.arc.push_back(arc);  // 0

        for (std::size_t i = 1; i < sec.count; ++i) {
            const auto swc_id = morph.datas[sec.offset + i];
            const auto& s = morph.swc[static_cast<std::size_t>(swc_id)];

            const float x = static_cast<float>(s.xyz[0]);
            const float y = static_cast<float>(s.xyz[1]);
            const float z = static_cast<float>(s.xyz[2]);

            const double dx = static_cast<double>(x - prev_x);
            const double dy = static_cast<double>(y - prev_y);
            const double dz = static_cast<double>(z - prev_z);
            arc += std::sqrt(dx * dx + dy * dy + dz * dz);

            prev_x = x;
            prev_y = y;
            prev_z = z;
            morph.pt3d.x.push_back(x);
            morph.pt3d.y.push_back(y);
            morph.pt3d.z.push_back(z);
            morph.pt3d.d.push_back(static_cast<float>(s.d_um));
            morph.pt3d.arc.push_back(arc);
        }

        sec.pt3d_count = sec.count;
        sec.L_um = (arc <= 1e-9) ? 1e-9 : arc;
    }
}

namespace {

NodeCoreSoA build_nodes_neuron_compatible_impl(Morph morph,
                                               const NodeBuildConfig& cfg,
                                               NodeBuildLayout* out_layout) {
    bool need_pt3d = morph.pt3d.empty();
    for (const auto& sec : morph.sections) {
        if (sec.pt3d_count == 0) {
            need_pt3d = true;
            break;
        }
    }
    if (need_pt3d) {
        build_pt3d_geometry(morph);
    }

    for (auto& sec : morph.sections) {
        sec.rallbranch = 1.0;
        if (sec.nseg <= 0) {
            throw std::runtime_error(
                "build_nodes_neuron_compatible_with_layout requires positive section.nseg, "
                "got nseg=" +
                std::to_string(sec.nseg) + " for section '" + sec.name + "'");
        }
    }

    std::vector<section_id> roots;
    for (section_id current_section_id = 0; current_section_id < morph.sections.size(); ++current_section_id) {
        if (morph.sections[current_section_id].parent_sec_id == invalid_section_id) {
            roots.push_back(current_section_id);
        }
    }

    std::vector<section_id> sections_bfs;
    std::deque<section_id> q;
    for (const auto root_section_id : roots) {
        q.push_back(root_section_id);
        while (!q.empty()) {
            const auto current_section_id = q.front();
            q.pop_front();
            sections_bfs.push_back(current_section_id);
            for (const auto child_section_id : morph.sections[current_section_id].children) {
                q.push_back(child_section_id);
            }
        }
    }

    const bool force_single_root = cfg.force_single_root && roots.size() > 1;

    std::size_t total_nodes = roots.size();
    if (force_single_root) {
        // +1 for the synthetic super-root node.
        total_nodes += 1;
    }
    for (const auto& sec : morph.sections) {
        total_nodes += static_cast<std::size_t>(sec.nseg) + 1;
    }

    NodeCoreSoA out{};

    const auto nsec = static_cast<std::int32_t>(morph.sections.size());
    std::vector<std::int32_t> root_node_id(static_cast<std::size_t>(nsec), -1);
    std::vector<std::int32_t> sec_parentnode_id(static_cast<std::size_t>(nsec), -1);
    std::vector<std::int32_t> sec_node_base_id(static_cast<std::size_t>(nsec), -1);

    auto push_node = [&](std::int32_t parent, double area, double a, double b, double ri, double a_scale) {
        out.parent_id.push_back(parent);
        out.area.push_back(area);
        out.a.push_back(a);
        out.b.push_back(b);
        out.ri.push_back(ri);
        out.a_scale.push_back(a_scale);
    };

    // Root nodes / parentnodes:
    //
    // - Normal case (single tree): one root node per root section (sec->parentnode).
    // - force_single_root case (forest): create a single synthetic super-root node, then
    //   create one per-root-section parentnode under it with zero coupling (a=b=0). Each
    //   root section then attaches to its own parentnode, preserving electrical
    //   independence between disconnected trees while producing a single-root node graph.
    if (!force_single_root) {
        // Root nodes: one per root section (sec->parentnode) in root creation order.
        for (const auto root_section_id : roots) {
            const auto node_id = static_cast<std::int32_t>(out.parent_id.size());
            root_node_id[static_cast<std::size_t>(root_section_id)] = node_id;
            sec_parentnode_id[static_cast<std::size_t>(root_section_id)] = node_id;
            // Internal representation: roots have parent_id < 0.
            // Exporters that target NEURON/CoreNEURON file formats may post-process this
            // into the on-disk convention (self-parent root ids).
            push_node(-1, 100.0, 0.0, 0.0, huge_resistance_Mohm, 0.0);
        }
    } else {
        // Synthetic super-root node (the only root in the node graph).
        const auto super_root_node_id = static_cast<std::int32_t>(out.parent_id.size());
        push_node(-1, 100.0, 0.0, 0.0, huge_resistance_Mohm, 0.0);

        // One per-root-section parentnode under the super-root.
        for (const auto root_section_id : roots) {
            const auto node_id = static_cast<std::int32_t>(out.parent_id.size());
            root_node_id[static_cast<std::size_t>(root_section_id)] = node_id;
            sec_parentnode_id[static_cast<std::size_t>(root_section_id)] = node_id;
            // Zero coupling to the super-root: this keeps disconnected trees independent.
            push_node(super_root_node_id, 100.0, 0.0, 0.0, huge_resistance_Mohm, 0.0);
        }
    }

    auto node_id_in_section = [](const Section& sec, double x) -> std::int32_t {
        const auto nseg = static_cast<std::int32_t>(sec.nseg);
        auto i = static_cast<std::int32_t>(static_cast<double>(nseg) * x);
        i = std::clamp(i, std::int32_t{0}, nseg);
        return i;
    };

    // Section nodes use NRN-compatible BFS section traversal. CPU mode does not apply a
    // later CoreNEURON cell/node permutation.
    for (const auto current_section_id : sections_bfs) {
        auto& sec = morph.sections[current_section_id];

        const auto base = static_cast<std::int32_t>(out.parent_id.size());
        sec_node_base_id[static_cast<std::size_t>(current_section_id)] = base;

        std::int32_t parentnode{};
        if (sec.parent_sec_id == invalid_section_id) {
            parentnode = sec_parentnode_id[static_cast<std::size_t>(current_section_id)];
        } else {
            const auto parent_section_id = sec.parent_sec_id;
            const auto& parent_sec = morph.sections[parent_section_id];
            const auto parentx = sec.parentx;

            const auto parent_section_index = static_cast<std::size_t>(parent_section_id);

            if (parentx <= 0.0) {
                parentnode = sec_parentnode_id[parent_section_index];
            } else if (parentx >= 1.0) {
                parentnode = sec_node_base_id[parent_section_index] + parent_sec.nseg;
            } else {
                const auto parent_node_index = node_id_in_section(parent_sec, parentx);
                parentnode = sec_node_base_id[parent_section_index] + parent_node_index;
            }


            sec_parentnode_id[static_cast<std::size_t>(current_section_id)] = parentnode;
        }

        const auto cable = compute_section_cable_data(morph, sec);

        for (std::int32_t j = 0; j <= sec.nseg; ++j) {
            const std::int32_t parent = (j == 0) ? parentnode : (base + (j - 1));
            const auto idx = static_cast<std::size_t>(j);
            const double area = cable.area[idx];
            const double rinv = cable.rinv[idx];
            const double ri = cable.ri[idx];
            const double a_scale = (j == 0) ? sec.rallbranch : 1.0;

            const double parent_area = out.area[static_cast<std::size_t>(parent)];
            const double a = -1.0e2 * a_scale * rinv / parent_area;
            const double b = -1.0e2 * rinv / area;

            push_node(parent, area, a, b, ri, a_scale);
        }
    }

    if (out_layout) {
        out_layout->root_sections = std::move(roots);
        out_layout->root_node_id = std::move(root_node_id);
        out_layout->section_node_base_id = std::move(sec_node_base_id);
        out_layout->section_nseg = std::vector<std::int32_t>(morph.sections.size());
        out_layout->section_parent_node_id = std::vector<std::int32_t>(morph.sections.size());
        for (std::size_t section_index = 0; section_index < morph.sections.size(); ++section_index) {
            out_layout->section_nseg[section_index] = morph.sections[section_index].nseg;
            out_layout->section_parent_node_id[section_index] = sec_parentnode_id[section_index];
        }
    }

    return out;
}

}  // namespace

NodeCoreSoA build_nodes_neuron_compatible(Morph morph, const NodeBuildConfig& cfg) {
    return build_nodes_neuron_compatible_impl(std::move(morph), cfg, /*out_layout=*/nullptr);
}

NodeBuildResult build_nodes_neuron_compatible_with_layout(Morph morph, const NodeBuildConfig& cfg) {
    NodeBuildResult out{};
    out.nodes = build_nodes_neuron_compatible_impl(std::move(morph), cfg, &out.layout);
    return out;
}

}  // namespace mind_micro_morph
