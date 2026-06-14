#pragma once

#include "dat_to_section.hpp"

#include <cstdint>
#include <vector>

namespace mind_micro_morph {

// Minimal node output needed by the current solver/mechanism interface.
struct NodeCoreSoA {
    std::vector<std::int32_t> parent_id{};
    std::vector<double> area{};
    std::vector<double> diam{};
    std::vector<double> a{};
    std::vector<double> b{};
    std::vector<double> ri{};
    std::vector<double> a_scale{};
};

struct NodeBuildConfig {
    // If true and the morphology has multiple root sections (a forest), collapse it into a
    // single-root tree by introducing a synthetic "super-root" node and attaching each original
    // root section's parentnode under it with zero coupling (a=b=0). This preserves electrical
    // independence between disconnected trees while satisfying 1-root-per-cell assumptions in
    // downstream builders/solvers.
    bool force_single_root{false};
    double min_diam_um{1e-6};
};

void build_pt3d_geometry(Morph& morph);

// Build a minimal node representation suitable for passing to the simulator core.
NodeCoreSoA build_nodes_neuron_compatible(Morph morph, const NodeBuildConfig& cfg);

// Additional build metadata for direct section->node mapping.
//
// For each section_id:
//  - section_node_base_id[sec] is the global node id of the first segment node (j=0).
//  - section_nseg[sec] is the nseg value used by the build.
//
// Segment nodes for a section are the contiguous range:
//   node_id = section_node_base_id[sec] + j, for j in [0, section_nseg[sec]-1]
// The final boundary node is:
//   boundary_node_id = section_node_base_id[sec] + section_nseg[sec]
struct NodeBuildLayout {
    std::vector<section_id> root_sections{};
    std::vector<std::int32_t> root_node_id{};
    std::vector<std::int32_t> section_node_base_id{};
    std::vector<std::int32_t> section_nseg{};
    // Template-node id of section's parent connection node (NEURON node_exact x==0 path).
    std::vector<std::int32_t> section_parent_node_id{};
};

struct NodeBuildResult {
    NodeCoreSoA nodes{};
    NodeBuildLayout layout{};
};

// Build nodes and expose section->node layout so downstream code (e.g. channel insertion)
// can expand per-section/per-label data into per-node SoA without re-traversing the tree.
NodeBuildResult build_nodes_neuron_compatible_with_layout(Morph morph, const NodeBuildConfig& cfg);

}  // namespace mind_micro_morph
