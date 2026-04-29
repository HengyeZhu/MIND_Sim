#include "morph/cell_location.hpp"

#include "morph/cell_template_layout.hpp"
#include "morph/section_distance.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace mind_micro_morph {
namespace {

bool set_error(std::string* error, std::string message) {
    if (error != nullptr) {
        *error = std::move(message);
    }
    return false;
}

struct CellSectionContext {
    int gid{-1};
    int cell_index{-1};
    int template_index{-1};
    int section_index{-1};
    const mind_micro_model::CellTemplateInfo* tpl{nullptr};
};

bool resolve_cell_template_context(const mind_micro_model::CellTemplateMorphLayout& morph,
                                   int gid,
                                   int* out_cell_index,
                                   int* out_template_index,
                                   const mind_micro_model::CellTemplateInfo** out_tpl,
                                   std::string* error) {
    if (gid < 0 || gid >= morph.num_cells_total) {
        return set_error(
            error,
            "invalid gid=" + std::to_string(gid) +
                " (num_cells_total=" + std::to_string(morph.num_cells_total) + ")");
    }

    const auto cell_index = static_cast<std::size_t>(gid);
    if (cell_index >= morph.cell_template_id.size() || cell_index >= morph.cell_nonroot_base.size()) {
        return set_error(error, "internal cell layout mismatch for gid=" + std::to_string(gid));
    }

    const auto template_index = morph.cell_template_id[cell_index];
    if (template_index >= morph.templates.size()) {
        return set_error(
            error,
            "invalid template id=" + std::to_string(template_index) + " for gid=" + std::to_string(gid));
    }
    if (out_cell_index != nullptr) {
        *out_cell_index = static_cast<int>(cell_index);
    }
    if (out_template_index != nullptr) {
        *out_template_index = static_cast<int>(template_index);
    }
    if (out_tpl != nullptr) {
        *out_tpl = &morph.templates[template_index];
    }
    return true;
}

bool validate_section_index(const mind_micro_model::CellTemplateInfo& tpl,
                            int section_index,
                            std::string* error) {
    if (section_index < 0) {
        return set_error(error, "invalid section index=" + std::to_string(section_index));
    }
    const auto section_u = static_cast<std::size_t>(section_index);
    if (section_u >= tpl.section_nseg.size() ||
        section_u >= tpl.section_node_base_id.size() ||
        section_u >= tpl.section_parent_node_id.size()) {
        return set_error(error,
                         "section index=" + std::to_string(section_index) +
                             " out of range for template '" + tpl.name + "'");
    }
    return true;
}

bool resolve_cell_section_context(const mind_micro_model::CellTemplateMorphLayout& morph,
                                  int gid,
                                  int section_index,
                                  CellSectionContext* out,
                                  std::string* error) {
    int cell_index = -1;
    int template_index = -1;
    const mind_micro_model::CellTemplateInfo* tpl = nullptr;
    if (!resolve_cell_template_context(morph, gid, &cell_index, &template_index, &tpl, error)) {
        return false;
    }
    if (!validate_section_index(*tpl, section_index, error)) {
        return false;
    }
    if (out == nullptr) {
        return set_error(error, "internal error: missing output context");
    }
    out->gid = gid;
    out->cell_index = cell_index;
    out->template_index = template_index;
    out->section_index = section_index;
    out->tpl = tpl;
    return true;
}

bool resolve_cell_location_info(const mind_micro_model::CellTemplateMorphLayout& morph,
                                const CellSectionContext& ctx,
                                double loc,
                                CellLocationResolveMode mode,
                                CellLocationResolveInfo* out,
                                std::string* error) {
    if (out == nullptr) {
        return set_error(error,
                         mode == CellLocationResolveMode::Segment ? "internal error: missing output segment info"
                                                                  : "internal error: missing output node info");
    }
    *out = CellLocationResolveInfo{};
    out->gid = ctx.gid;
    out->requested_loc = loc;

    if (!std::isfinite(loc)) {
        return set_error(error, "loc must be finite (got=" + std::to_string(loc) + ")");
    }

    const auto section_index = static_cast<std::size_t>(ctx.section_index);
    const int nseg = ctx.tpl->section_nseg[section_index];
    if (nseg <= 0) {
        return set_error(
            error,
            "invalid nseg=" + std::to_string(nseg) + " for section_index=" + std::to_string(ctx.section_index));
    }

    std::int32_t segment_index = -1;
    int template_node_index = -1;
    if (mode == CellLocationResolveMode::Segment) {
        if (!resolve_segment_index_for_nseg(static_cast<std::int32_t>(nseg), loc, &segment_index)) {
            return set_error(
                error,
                "failed to resolve segment for section_index=" + std::to_string(ctx.section_index) +
                    " loc=" + std::to_string(loc) +
                    " nseg=" + std::to_string(nseg));
        }
        template_node_index = ctx.tpl->section_node_base_id[section_index] + static_cast<int>(segment_index);
    } else {
        std::int32_t resolved_template_node = -1;
        if (!resolve_template_node_by_section_loc(ctx.tpl->section_node_base_id,
                                                  ctx.tpl->section_nseg,
                                                  ctx.tpl->section_parent_node_id,
                                                  section_index,
                                                  loc,
                                                  &resolved_template_node)) {
            return set_error(
                error,
                "failed to resolve exact node for section_index=" + std::to_string(ctx.section_index) +
                    " loc=" + std::to_string(loc));
        }
        template_node_index = static_cast<int>(resolved_template_node);
        (void)resolve_segment_index_for_nseg(static_cast<std::int32_t>(nseg), loc, &segment_index);
    }

    *out = CellLocationResolveInfo{
        .gid = ctx.gid,
        .cell_index = ctx.cell_index,
        .template_index = ctx.template_index,
        .section_index = ctx.section_index,
        .segment_index = static_cast<int>(segment_index),
        .nseg = nseg,
        .template_node_index = template_node_index,
        .original_node_index =
            mind_micro_model::map_template_node_to_original(morph, ctx.gid, template_node_index),
        .requested_loc = loc,
        .segment_center_loc =
            segment_index >= 0 ? (2.0 * static_cast<double>(segment_index) + 1.0) / (2.0 * static_cast<double>(nseg))
                               : std::numeric_limits<double>::quiet_NaN(),
    };
    return true;
}

}  // namespace

bool resolve_cell_location(const mind_micro_model::CellTemplateMorphLayout& morph,
                           int gid,
                           int section_index,
                           double loc,
                           CellLocationResolveMode mode,
                           CellLocationResolveInfo* out,
                           std::string* error) {
    CellSectionContext ctx{};
    if (!resolve_cell_section_context(morph, gid, section_index, &ctx, error)) {
        return false;
    }
    return resolve_cell_location_info(morph, ctx, loc, mode, out, error);
}

bool resolve_cell_location(const mind_micro_model::CellTemplateMorphLayout& morph,
                           const CellLocationSpec& location,
                           CellLocationResolveMode mode,
                           CellLocationResolveInfo* out,
                           std::string* error) {
    return resolve_cell_location(
        morph,
        location.gid,
        location.section_index,
        location.loc,
        mode,
        out,
        error);
}


}  // namespace mind_micro_morph
