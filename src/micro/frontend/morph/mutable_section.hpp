#pragma once

#include "dat_to_section.hpp"

#include <array>
#include <span>
#include <string_view>
#include <vector>

namespace mind_micro_morph {

// Rebuild per-section `children` lists after mutating section topology.
void rebuild_section_children(Morph& morph);

// Append a new section and point-id stream to the Morph, keeping the datas+offset/count layout.
section_id append_section(Morph& morph, Section sec, std::span<const swcdata_id> point_ids);

// Assign pt3d (xyz + diameter + cumulative arc) for a section by appending into Morph::pt3d.
// The section must not already have pt3d points assigned.
void set_section_pt3d(Morph& morph,
                      section_id target_section_id,
                      std::span<const float> x_um,
                      std::span<const float> y_um,
                      std::span<const float> z_um,
                      std::span<const double> arc_um,
                      std::span<const float> diam_um);

// Delete a section subtree (root section + all descendants), compacting
// Morph::sections, Morph::datas, and Morph::pt3d, then rebuilding children lists.
void delete_section_subtree(Morph& morph, section_id subtree_root_section_id);

// Delete all sections with the provided label and every descendant below them.
void delete_label(Morph& morph, std::string_view label);

struct Pt3dPoint {
    // Intermediate morphology points are double precision.
    // They are quantized to float only when written to Morph::pt3d.
    double x_um{};
    double y_um{};
    double z_um{};
    double diam_um{};
};

// A small helper to generate NEURON define_shape-compatible "stylized" 3D:
//  - pt3d x/y/z/diam are float-like (we take a float start x and float diam).
//  - arc is double with arc[0]=0 and arc[2]=L (forced), matching NEURON's behavior.
//  - end_x_um is the rounded float endpoint (used to place subsequent sections).
struct NeuronStylizedPt3d3 {
    std::array<double, 3> arc_um{};
    std::array<float, 3> diam_um{};
    float end_x_um{};
};

NeuronStylizedPt3d3 neuron_stylized_pts3(float start_x_um, double L_um, float diam_um);

// Append a new section whose pt3d (arc+diam) matches NEURON's stylized define_shape.
// The geometry is stored directly in Morph::pt3d; a single dummy SWC point is used
// to satisfy internal datas invariants.
section_id append_stylized_section(Morph& morph,
                                  const std::string& name,
                                  section_id parent_section_id,
                                  double parentx,
                                  std::string label,
                                  float start_x_um,
                                  double L_um,
                                  float diam_um);

// Append a NEURON-compatible "n3d==0" cylinder section:
//  - no pt3d points (Section::pt3d_count stays 0)
//  - length stored in Section::L_um
//  - diameter stored on the dummy SWC head point (for debugging only)
section_id append_cylindrical_section_no_pt3d(Morph& morph,
                                             const std::string& name,
                                             section_id parent_section_id,
                                             double parentx,
                                             std::string label,
                                             double L_um,
                                             double diam_um);

// Arbor-like builder primitive: append a new section by providing pt3d points.
//
// - If `parent_section_id==invalid_section_id`, a new root section is created.
// - Otherwise the new section is connected to `parent_section_id` at `parentx` in [0,1].
// - Geometry is stored directly in `Morph::pt3d` (arc+diam); SWC/datas are filled
//   with a single dummy point to satisfy internal invariants.
section_id append_section_with_pt3d(Morph& morph,
                                    const std::string& name,
                                    section_id parent_section_id,
                                    double parentx,
                                    std::string label,
                                    std::span<const Pt3dPoint> points);

}  // namespace mind_micro_morph
