#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <string_view>

namespace mind_micro_model {
struct CellTemplateMorphLayout;
}

namespace mind_micro_morph {

struct CellLocationSpec {
    int gid{-1};
    int section_index{-1};
    double loc{0.5};
};

inline bool operator==(const CellLocationSpec& lhs, const CellLocationSpec& rhs) noexcept {
    return lhs.gid == rhs.gid && lhs.section_index == rhs.section_index && lhs.loc == rhs.loc;
}

enum class CellLocationResolveMode {
    Segment = 0,
    ExactNode,
};

struct CellLocationResolveInfo {
    int gid{-1};
    int cell_index{-1};
    int template_index{-1};
    int section_index{-1};
    int segment_index{-1};
    int nseg{-1};
    int template_node_index{-1};
    int original_node_index{-1};
    double requested_loc{std::numeric_limits<double>::quiet_NaN()};
    double segment_center_loc{std::numeric_limits<double>::quiet_NaN()};
};

[[nodiscard]] bool resolve_cell_location(
    const mind_micro_model::CellTemplateMorphLayout& morph,
    int gid,
    int section_index,
    double loc,
    CellLocationResolveMode mode,
    CellLocationResolveInfo* out,
    std::string* error = nullptr);

[[nodiscard]] bool resolve_cell_location(
    const mind_micro_model::CellTemplateMorphLayout& morph,
    const CellLocationSpec& location,
    CellLocationResolveMode mode,
    CellLocationResolveInfo* out,
    std::string* error = nullptr);

}  // namespace mind_micro_morph
