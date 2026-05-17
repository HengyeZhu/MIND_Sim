#include "micro/frontend/model.hpp"

#include "coreneuron/coreneuron.hpp"
#include "coreneuron/apps/corenrn_parameters.hpp"
#include "coreneuron/io/mem_layout_util.hpp"
#include "coreneuron/nrniv/nrniv_decl.h"
#include "coreneuron/permute/cellorder.hpp"
#include "coreneuron/sim/multicore.hpp"
#include "morph/section_to_node.hpp"
#include "micro/sim/mechanism_runtime.hpp"
#include "micro/sim/micro_runtime.hpp"

#if defined(MIND_SIM_ENABLE_GPU) && defined(CORENEURON_ENABLE_GPU)
#include "coreneuron/gpu/nrn_acc_manager.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>

namespace mind_sim::micro::frontend {

namespace {

struct TemplateBuildScratch {
    std::string name{};
    int num_cells{0};
    std::vector<std::string> label_names{};
    std::unordered_map<std::string, std::size_t> label_index{};
    std::vector<std::vector<mind_micro_morph::section_id>> sections_by_label{};
    std::vector<std::int32_t> section_label_u_by_sec{};
    mind_micro_morph::NodeBuildResult tpl{};
    std::int32_t tpl_soma_node{-1};
};

struct PopulationLayout {
    int num_cells_total{0};
    std::vector<std::size_t> cell_template_id{};
    std::vector<std::size_t> cell_nonroot_base{};
};

constexpr double infinite_resistance_cutoff_Mohm = 1.0e29;

[[nodiscard]] std::size_t checked_node_count(const mind_micro_morph::NodeCoreSoA& nodes) {
    if (nodes.area.size() != nodes.parent_id.size() || nodes.a.size() != nodes.parent_id.size() ||
        nodes.b.size() != nodes.parent_id.size() || nodes.ri.size() != nodes.parent_id.size() ||
        nodes.a_scale.size() != nodes.parent_id.size()) {
        throw std::runtime_error("inconsistent morphology node data");
    }
    return nodes.parent_id.size();
}

template <typename Fn>
void for_each_population_node(const std::vector<TemplateBuildScratch>& built,
                              const mind_micro_model::CellTemplateMorphLayout& morph_layout,
                              Fn&& fn) {
    const auto num_cells_total = static_cast<std::size_t>(morph_layout.num_cells_total);
    for (std::size_t cell = 0; cell < num_cells_total; ++cell) {
        const auto tpl_id = morph_layout.cell_template_id[cell];
        const auto& tpl_nodes = built[tpl_id].tpl.nodes;
        const std::size_t tpl_n = checked_node_count(tpl_nodes);
        const auto nonroot_base = morph_layout.cell_nonroot_base[cell];

        for (std::size_t tnode = 0; tnode < tpl_n; ++tnode) {
            const auto original_index = (tnode == 0) ? cell : (nonroot_base + (tnode - 1));
            const auto tpl_parent = tpl_nodes.parent_id[tnode];
            std::int32_t parent_index = -1;
            if (tpl_parent >= 0) {
                parent_index = (tpl_parent == 0)
                                   ? static_cast<std::int32_t>(cell)
                                   : static_cast<std::int32_t>(
                                         nonroot_base + (static_cast<std::size_t>(tpl_parent) - 1));
            }
            fn(original_index,
               parent_index,
               tpl_nodes.a[tnode],
               tpl_nodes.b[tnode],
               tpl_nodes.area[tnode],
               tpl_nodes.ri[tnode],
               tpl_nodes.a_scale[tnode]);
        }
    }
}

[[nodiscard]] std::int32_t pick_soma_node_index(
    const std::vector<std::vector<mind_micro_morph::section_id>>& sections_by_label,
    const std::unordered_map<std::string, std::size_t>& label_index,
    const mind_micro_morph::NodeBuildLayout& layout) {
    auto it = label_index.find("soma");
    if (it == label_index.end()) {
        throw std::runtime_error("missing 'soma' label for morphology template");
    }
    const auto soma_label = it->second;
    if (soma_label >= sections_by_label.size() || sections_by_label[soma_label].empty()) {
        throw std::runtime_error("empty 'soma' label for morphology template");
    }
    const auto soma_sec = static_cast<std::size_t>(sections_by_label[soma_label].front());
    const auto base = layout.section_node_base_id[soma_sec];
    const auto nseg = layout.section_nseg[soma_sec];
    const auto center = std::min<std::int32_t>(
        nseg - 1,
        static_cast<std::int32_t>(static_cast<double>(nseg) * 0.5));
    return base + center;
}

[[nodiscard]] std::vector<TemplateBuildScratch> build_template_scratch(
    const std::vector<MorphologyTemplateSpec>& templates) {
    if (templates.empty()) {
        throw std::runtime_error("build_morphology requires at least one morphology template");
    }

    std::unordered_set<std::string> unique_names;
    unique_names.reserve(templates.size());
    std::vector<TemplateBuildScratch> built;
    built.resize(templates.size());

    for (std::size_t idx = 0; idx < templates.size(); ++idx) {
        const auto& spec = templates[idx];
        if (spec.name.empty()) {
            throw std::runtime_error("morphology template name must not be empty");
        }
        if (!unique_names.insert(spec.name).second) {
            throw std::runtime_error("duplicate morphology template name: " + spec.name);
        }
        if (!spec.sections || spec.sections->empty()) {
            throw std::runtime_error("morphology template '" + spec.name + "' is missing sections");
        }
        if (spec.num_cells <= 0) {
            throw std::runtime_error("morphology template '" + spec.name + "' has invalid num_cells");
        }

        mind_micro_frontend::SectionNseg section_nseg;
        auto morph = mind_micro_frontend::build_morph_from_sections(*spec.sections, section_nseg);
        mind_micro_frontend::apply_section_nseg(morph, section_nseg);

        TemplateBuildScratch out{};
        out.name = spec.name;
        out.num_cells = spec.num_cells;
        out.sections_by_label = std::move(morph.sections_by_label);
        out.label_names = std::move(morph.label_names);
        out.label_index = std::move(morph.label_index);

        mind_micro_morph::NodeBuildConfig cfg{};
        cfg.force_single_root = true;
        cfg.min_diam_um = 1e-6;
        out.tpl = mind_micro_morph::build_nodes_neuron_compatible_with_layout(std::move(morph), cfg);

        out.section_label_u_by_sec.assign(out.tpl.layout.section_nseg.size(), -1);
        for (std::size_t label = 0; label < out.sections_by_label.size(); ++label) {
            for (const auto sec_id : out.sections_by_label[label]) {
                const auto sec_index = static_cast<std::size_t>(sec_id);
                if (sec_index >= out.section_label_u_by_sec.size()) {
                    throw std::runtime_error("section index out of range in label map");
                }
                if (out.section_label_u_by_sec[sec_index] >= 0) {
                    throw std::runtime_error("section appears in multiple labels");
                }
                out.section_label_u_by_sec[sec_index] = static_cast<std::int32_t>(label);
            }
        }
        out.tpl_soma_node = pick_soma_node_index(out.sections_by_label, out.label_index, out.tpl.layout);
        built[idx] = std::move(out);
    }

    return built;
}

void build_population_layout(const std::vector<TemplateBuildScratch>& built,
                             PopulationLayout& pop_layout,
                             std::vector<std::size_t>& template_cell_base,
                             std::size_t& nnode) {
    pop_layout = PopulationLayout{};
    for (const auto& tpl : built) {
        pop_layout.num_cells_total += tpl.num_cells;
    }

    template_cell_base.assign(built.size(), 0);
    pop_layout.cell_template_id.resize(static_cast<std::size_t>(pop_layout.num_cells_total));

    std::size_t next_cell = 0;
    for (std::size_t tpl_id = 0; tpl_id < built.size(); ++tpl_id) {
        template_cell_base[tpl_id] = next_cell;
        for (int c = 0; c < built[tpl_id].num_cells; ++c) {
            pop_layout.cell_template_id[next_cell++] = tpl_id;
        }
    }

    std::size_t total_nonroots = 0;
    for (const auto& tpl : built) {
        const std::size_t tpl_nodes = checked_node_count(tpl.tpl.nodes);
        if (tpl_nodes == 0) {
            throw std::runtime_error("morphology template generated no nodes");
        }
        total_nonroots += static_cast<std::size_t>(tpl.num_cells) * (tpl_nodes - 1);
    }

    nnode = static_cast<std::size_t>(pop_layout.num_cells_total) + total_nonroots;
    pop_layout.cell_nonroot_base.resize(static_cast<std::size_t>(pop_layout.num_cells_total));

    std::size_t next_nonroot = static_cast<std::size_t>(pop_layout.num_cells_total);
    for (std::size_t cell = 0; cell < static_cast<std::size_t>(pop_layout.num_cells_total); ++cell) {
        const auto tpl_id = pop_layout.cell_template_id[cell];
        const std::size_t tpl_nodes = checked_node_count(built[tpl_id].tpl.nodes);
        pop_layout.cell_nonroot_base[cell] = next_nonroot;
        next_nonroot += tpl_nodes - 1;
    }
}

[[nodiscard]] mind_micro_model::CellTemplateMorphLayout build_morph_layout(
    std::vector<TemplateBuildScratch>& built,
    const std::vector<std::size_t>& template_cell_base,
    const PopulationLayout& pop_layout,
    std::size_t nnode) {
    mind_micro_model::CellTemplateMorphLayout morph{};
    morph.num_cells_total = pop_layout.num_cells_total;
    morph.nnode = nnode;
    morph.cell_template_id = pop_layout.cell_template_id;
    morph.cell_nonroot_base = pop_layout.cell_nonroot_base;
    morph.templates.reserve(built.size());

    for (std::size_t tpl_id = 0; tpl_id < built.size(); ++tpl_id) {
        auto& src = built[tpl_id];
        mind_micro_model::CellTemplateInfo dst{};
        dst.name = std::move(src.name);
        dst.cell_base = static_cast<int>(template_cell_base[tpl_id]);
        dst.num_cells = src.num_cells;
        dst.tpl_nodes_per_cell = checked_node_count(src.tpl.nodes);
        dst.tpl_soma_node = src.tpl_soma_node;
        dst.label_names = std::move(src.label_names);
        dst.label_index = std::move(src.label_index);
        dst.sections_by_label = std::move(src.sections_by_label);
        dst.section_label_u_by_sec = std::move(src.section_label_u_by_sec);
        dst.section_node_base_id = std::move(src.tpl.layout.section_node_base_id);
        dst.section_nseg = std::move(src.tpl.layout.section_nseg);
        dst.section_parent_node_id = std::move(src.tpl.layout.section_parent_node_id);
        morph.templates.push_back(std::move(dst));
    }

    return morph;
}

[[nodiscard]] std::vector<PopulationRange> build_population_ranges(
    const mind_micro_model::CellTemplateMorphLayout& morph) {
    std::vector<PopulationRange> ranges;
    ranges.reserve(morph.templates.size());
    for (const auto& tpl : morph.templates) {
        ranges.push_back(PopulationRange{
            .name = tpl.name,
            .gid_begin = tpl.cell_base,
            .gid_end = tpl.cell_base + tpl.num_cells,
        });
    }
    return ranges;
}

const mind_micro_biophysical::ParamValue* find_param(const mind_micro_biophysical::ParamList& params,
                                                     const std::string& key) {
    for (const auto& [name, value] : params) {
        if (name == key) {
            return &value;
        }
    }
    return nullptr;
}

struct ParamSpan {
    const CoreMechanismBuildState* builder{};
    std::size_t begin{0};
    std::size_t end{0};
};

[[nodiscard]] ParamSpan param_span_for_insert_index(const CoreMechanismBuildState& builder,
                                                    std::size_t insert_index) {
    const auto& insert = builder.inserts[insert_index];
    return ParamSpan{
        .builder = &builder,
        .begin = insert.param_begin,
        .end = insert.param_end,
    };
}

const mind_micro_biophysical::ParamValue* find_param(const ParamSpan& params,
                                                     int field_index) {
    for (std::size_t i = params.begin; i < params.end; ++i) {
        if (params.builder->param_overrides[i].field_index == field_index) {
            return &params.builder->param_overrides[i].value;
        }
    }
    return nullptr;
}

double segment_batch_value(const mind_micro_biophysical::SegmentParamBatch& batch,
                           int section_index,
                           int segment_index) {
    for (std::size_t row = 0; row < batch.section_indices.size(); ++row) {
        if (batch.section_indices[row] != section_index) {
            continue;
        }
        const auto begin = batch.value_offsets[row];
        const auto end = batch.value_offsets[row + 1];
        if (begin >= end) {
            return 0.0;
        }
        const auto offset = begin + static_cast<std::size_t>(
            std::min<int>(segment_index, static_cast<int>(end - begin) - 1));
        return batch.values[offset];
    }
    return 0.0;
}

double supplied_param_value_for_segment(const mind_micro_biophysical::ParamValue& value,
                                        int section_index,
                                        int segment_index) {
    if (std::holds_alternative<double>(value)) {
        return std::get<double>(value);
    }
    if (std::holds_alternative<mind_micro_biophysical::SegmentParamBatch>(value)) {
        return segment_batch_value(std::get<mind_micro_biophysical::SegmentParamBatch>(value),
                                   section_index,
                                   segment_index);
    }
    throw std::runtime_error("non-numeric mechanism parameter passed validation unexpectedly");
}

void append_local_param_overrides(const mind_micro_biophysical::MechanismMetadata& metadata,
                                  mind_micro_biophysical::ParamList& params,
                                  std::vector<CoreParamOverride>& out) {
    for (auto& [name, value] : params) {
        const auto field_index_it = metadata.field_index_by_name.find(name);
        if (field_index_it == metadata.field_index_by_name.end()) {
            throw std::runtime_error(
                "parameter '" + name + "' is not a RANGE/PARAMETER/STATE/ASSIGNED field of mechanism '" +
                metadata.name + "'");
        }
        if (!std::holds_alternative<double>(value) &&
            !std::holds_alternative<mind_micro_biophysical::SegmentParamBatch>(value)) {
            throw std::runtime_error(
                "parameter '" + name + "' for mechanism '" + metadata.name +
                "' must be numeric or segment_values");
        }
        const int field_index = field_index_it->second;
        const auto& field = metadata.fields[static_cast<std::size_t>(field_index)];
        if (field.is_global) {
            if (!std::holds_alternative<double>(value)) {
                throw std::runtime_error(
                    "global parameter '" + name + "' for mechanism '" + metadata.name +
                    "' must be scalar");
            }
            mind_sim::micro::sim::core_set_global_parameter(
                metadata.name,
                field.name,
                std::get<double>(value));
            continue;
        }
        out.push_back(CoreParamOverride{
            .field_index = field_index,
            .data_offset = metadata.field_data_offsets[static_cast<std::size_t>(field_index)],
            .value = std::move(value),
        });
    }
}

int padded_node_count(int count, int mechanism_type) {
    return coreneuron::nrn_soa_padded_size(
        count,
        coreneuron::corenrn.get_mech_data_layout()[static_cast<std::size_t>(mechanism_type)]);
}

std::size_t mechanism_data_size(int mechanism_type, int padded_count) {
    return static_cast<std::size_t>(
               coreneuron::corenrn.get_prop_param_size()[static_cast<std::size_t>(mechanism_type)]) *
           static_cast<std::size_t>(padded_count);
}

std::size_t mechanism_data_size(const mind_sim::micro::sim::CoreMembList& ml) {
    return mechanism_data_size(ml.type, ml.ml._nodecount_padded);
}

struct ThreadDataAppender {
    mind_sim::micro::sim::CoreNeuronThread& thread;
    std::size_t next_offset{0};

    explicit ThreadDataAppender(mind_sim::micro::sim::CoreNeuronThread& target,
                                std::size_t mechanism_data_size)
        : thread(target),
          next_offset(target.mechanism_data_begin()) {
        thread.data_storage.resize(next_offset + mechanism_data_size, 0.0);
    }

    double* allocate(mind_sim::micro::sim::CoreMembList& ml, std::size_t size) {
        ml.thread_data_offset = next_offset;
        next_offset += size;
        return thread.data_storage.data() + ml.thread_data_offset;
    }

    void append(mind_sim::micro::sim::CoreMembList ml) {
        thread.tml_storage.push_back(coreneuron::NrnThreadMembList{
            .next = nullptr,
            .ml = nullptr,
            .index = ml.type,
            .dependencies = nullptr,
            .ndependencies = 0,
        });
        thread.memb_lists.push_back(std::move(ml));
    }
};

double* thread_memb_data(mind_sim::micro::sim::CoreNeuronThread& thread,
                         const mind_sim::micro::sim::CoreMembList& ml) {
    return thread.data_storage.data() + ml.thread_data_offset;
}

coreneuron::Datum append_vdata(mind_sim::micro::sim::CoreNeuronThread& thread, void* value) {
    const auto index = static_cast<coreneuron::Datum>(thread.vdata_storage.size());
    thread.vdata_storage.push_back(value);
    return index;
}

coreneuron::Datum append_random123(mind_sim::micro::sim::CoreNeuronThread& thread) {
    auto* state = coreneuron::nrnran123_newstream3(0, 0, 0);
    thread.random123_storage.emplace_back(state);
    return append_vdata(thread, state);
}

struct IonThreadStorage {
    int type{-1};
    std::size_t offset{0};
    int padded_count{0};
    const std::vector<int>* row_by_node{};
};

struct PendingIonMembList {
    mind_sim::micro::sim::CoreMembList ml{};
    std::unordered_map<std::string, int> field_index_by_name{};
    std::vector<int> row_by_node{};
};

[[nodiscard]] bool contains_name(const std::vector<std::string>& names,
                                 const std::string& name) {
    return std::find(names.begin(), names.end(), name) != names.end();
}

constexpr int nrn_ion_conc_mask = 03;
constexpr int nrn_ion_conc_init = 04;
constexpr int nrn_ion_rev_mask = 030;
constexpr int nrn_ion_rev_shift = 3;
constexpr int nrn_ion_rev_init = 040;
constexpr int nrn_ion_rev_advance = 0100;
constexpr int nrn_ion_write_interior = 0200;
constexpr int nrn_ion_write_exterior = 0400;
constexpr int nrn_ion_write_mask = nrn_ion_write_interior | nrn_ion_write_exterior;
constexpr int nrn_ion_style_low_mask = 0177;

[[nodiscard]] int promoted_ion_style(int current_style, int conc, int rev) {
    int old_conc = current_style & nrn_ion_conc_mask;
    int old_rev = (current_style & nrn_ion_rev_mask) >> nrn_ion_rev_shift;
    if (old_conc < conc) {
        old_conc = conc;
    }
    if (old_rev < rev) {
        old_rev = rev;
    }
    if (old_conc > 0 && old_rev < 2) {
        old_rev = 2;
    }
    int style = current_style & ~nrn_ion_style_low_mask;
    style += old_conc + (1 << nrn_ion_rev_shift) * old_rev;
    if (old_conc == 3) {
        style += nrn_ion_conc_init;
        if (old_rev == 2) {
            style += nrn_ion_rev_advance;
        }
    }
    if (old_conc > 0 && old_rev == 2) {
        style += nrn_ion_rev_init;
    }
    return style;
}

void build_tml_from_registered_execution_order(mind_sim::micro::sim::CoreNeuronThread& thread) {
    struct OrderedType {
        int type{-1};
        int rank{0};
        std::size_t original_index{0};
    };

    std::vector<OrderedType> order;
    order.reserve(thread.memb_lists.size());
    for (std::size_t index = 0; index < thread.memb_lists.size(); ++index) {
        const auto& ml = thread.memb_lists[index];
        order.push_back(OrderedType{
            .type = ml.type,
            .rank = mind_sim::micro::sim::core_mechanism_order_rank(ml.type),
            .original_index = index,
        });
    }

    std::stable_sort(order.begin(), order.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.rank != rhs.rank) {
            return lhs.rank < rhs.rank;
        }
        if (lhs.type != rhs.type) {
            return lhs.type < rhs.type;
        }
        return lhs.original_index < rhs.original_index;
    });

    thread.tml_storage.clear();
    thread.tml_storage.reserve(order.size());
    for (const auto& item : order) {
        thread.tml_storage.push_back(coreneuron::NrnThreadMembList{
            .next = nullptr,
            .ml = nullptr,
            .index = item.type,
            .dependencies = nullptr,
            .ndependencies = 0,
        });
    }
}

int resolve_original_node(const mind_micro_model::CellTemplateMorphLayout& morph,
                          int gid,
                          int section_index,
                          double loc) {
    mind_micro_morph::CellLocationResolveInfo info{};
    std::string error;
    if (!mind_micro_morph::resolve_cell_location(
            morph,
            gid,
            section_index,
            loc,
            mind_micro_morph::CellLocationResolveMode::ExactNode,
            &info,
            &error)) {
        throw std::runtime_error(error);
    }
    return info.original_node_index;
}

}  // namespace

MicroFrontendModel::MicroFrontendModel() {
    load_default_mechanism_metadata();
}

void MicroFrontendModel::set_dt(double dt) {
    if (!std::isfinite(dt) || dt <= 0.0) {
        throw std::runtime_error("dt must be positive and finite");
    }
    dt_ = dt;
    core_neuron_data_->dt = dt_;
    if (!core_neuron_data_->threads.empty()) {
        core_neuron_data_->threads.front()._dt = dt_;
        core_neuron_data_->bind();
    }
    runtime_backend_.reset();
}

void MicroFrontendModel::set_celsius(double celsius) {
    if (!std::isfinite(celsius)) {
        throw std::runtime_error("celsius must be finite");
    }
    section_properties_.celsius = celsius;
    core_neuron_data_->celsius = section_properties_.celsius;
    runtime_backend_.reset();
}

void MicroFrontendModel::set_device(const std::string& device) {
    auto config = mind_sim::micro::sim::parse_micro_device(device);
#ifndef MIND_SIM_ENABLE_GPU
    if (config.kind == mind_sim::micro::sim::MicroDeviceKind::Gpu) {
        throw std::runtime_error(
            "MIND_Sim was built without GPU support; rebuild with -DMIND_SIM_ENABLE_GPU=ON");
    }
#endif
    if (microcircuit_built_) {
        throw std::runtime_error("set_device must be called before build_microcircuit");
    }
    device_config_ = config;
    core_neuron_data_->device_config = device_config_;
    if (has_morphology_ && !core_neuron_data_->threads.empty()) {
        reset_core_node_storage();
        core_neuron_data_->bind();
    }
    runtime_backend_.reset();
}

void MicroFrontendModel::load_mech_metadata(std::string path) {
    mechanism_catalog_.load_metadata_path(path);
    refresh_mechanism_runtime_cache();
    loaded_mech_metadata_paths_.push_back(std::move(path));
}

int MicroFrontendModel::ion_register(std::string ion, double charge) {
    const int type = mind_sim::micro::sim::core_ion_register(ion, charge);
    mechanism_catalog_.sync_registered_mechanisms();
    refresh_mechanism_runtime_cache();
    runtime_backend_.reset();
    return type;
}

double MicroFrontendModel::ion_charge(const std::string& ion_mechanism) const {
    return mind_sim::micro::sim::core_ion_charge(ion_mechanism);
}

void MicroFrontendModel::set_global_scalar(const std::string& name, double value) {
    mind_sim::micro::sim::core_set_global_scalar(name, value);
    runtime_backend_.reset();
}

double MicroFrontendModel::global_scalar(const std::string& name) const {
    return mind_sim::micro::sim::core_global_scalar(name);
}

void MicroFrontendModel::load_default_mechanism_metadata() {
    mind_sim::micro::sim::ensure_core_mechanisms_registered();
    mechanism_catalog_.sync_registered_mechanisms();
    refresh_mechanism_runtime_cache();
    default_mechanism_metadata_loaded_ = true;
}

void MicroFrontendModel::build_morphology(std::vector<MorphologyTemplateSpec> templates) {
    std::vector<TemplateBuildScratch> built;
    {
        built = build_template_scratch(templates);
    }
    PopulationLayout pop_layout{};
    std::vector<std::size_t> template_cell_base;
    std::size_t nnode = 0;
    {
        build_population_layout(built, pop_layout, template_cell_base, nnode);
    }
    {
        morph_layout_ = build_morph_layout(built, template_cell_base, pop_layout, nnode);
    }
    core_neuron_data_ = std::make_shared<mind_sim::micro::sim::CoreNeuronData>();
    runtime_backend_.reset();
    core_neuron_data_->device_config = device_config_;
    core_neuron_data_->dt = dt_;
    core_neuron_data_->celsius = section_properties_.celsius;
    core_neuron_data_->threads.resize(1);
    auto& nt = core_neuron_data_->threads.front();
    nt.id = 0;
    nt._dt = dt_;
    nt.ncell = morph_layout_.num_cells_total;
    nt.end = static_cast<int>(morph_layout_.nnode);
    nt.shadow_rhs.clear();
    nt.shadow_d.clear();
    morph_parent_index_.assign(morph_layout_.nnode, -1);
    morph_area_.assign(morph_layout_.nnode, 0.0);
    axial_a_ra1_.assign(morph_layout_.nnode, 0.0);
    axial_b_ra1_.assign(morph_layout_.nnode, 0.0);
    axial_ri_ra1_.assign(morph_layout_.nnode, 0.0);
    axial_a_scale_.assign(morph_layout_.nnode, 0.0);
    runtime_node_by_original_.resize(morph_layout_.nnode);
    std::iota(runtime_node_by_original_.begin(), runtime_node_by_original_.end(), 0);
    {
        for_each_population_node(
            built,
            morph_layout_,
            [&](std::size_t original_index,
                std::int32_t parent_index,
                double a,
                double b,
                double area,
                double ri,
                double a_scale) {
                morph_parent_index_[original_index] = parent_index;
                axial_a_ra1_[original_index] = a;
                axial_b_ra1_[original_index] = b;
                morph_area_[original_index] = area;
                axial_ri_ra1_[original_index] = ri;
                axial_a_scale_[original_index] = a_scale;
            });
    }
    reset_core_node_storage();
    core_neuron_data_->bind();
    populations_ = build_population_ranges(morph_layout_);
    population_index_by_name_.clear();
    for (std::size_t i = 0; i < populations_.size(); ++i) {
        population_index_by_name_.emplace(populations_[i].name, i);
    }
    section_properties_.clear();
    {
        init_section_properties();
    }
    core_mechanism_builder_.clear();
    network_registry_.clear();
    has_morphology_ = true;
    microcircuit_built_ = false;
    core_initialized_ = false;
}

int MicroFrontendModel::population_count() const {
    require_morphology();
    return static_cast<int>(populations_.size());
}

const PopulationRange& MicroFrontendModel::population(std::size_t index) const {
    require_morphology();
    if (index >= populations_.size()) {
        throw std::runtime_error("population index out of range");
    }
    return populations_[index];
}

const PopulationRange& MicroFrontendModel::population(const std::string& name) const {
    require_morphology();
    const auto it = population_index_by_name_.find(name);
    if (it == population_index_by_name_.end()) {
        throw std::runtime_error("unknown population: " + name);
    }
    return populations_[it->second];
}

int MicroFrontendModel::section_count(int gid) const {
    require_morphology();
    return static_cast<int>(mind_micro_model::section_count(morph_layout_, gid));
}

std::vector<std::size_t> MicroFrontendModel::section_indices_for_label(int gid, const std::string& label) const {
    require_morphology();
    return mind_micro_model::section_indices_for_label(morph_layout_, gid, label);
}

std::string MicroFrontendModel::section_label(int gid, std::size_t section_index) const {
    require_morphology();
    return mind_micro_model::section_label(morph_layout_, gid, section_index);
}

void MicroFrontendModel::set_cell_v_init(int gid, double value) {
    require_morphology();
    (void) mind_micro_model::require_cell_template(morph_layout_, gid);
    mind_micro_biophysical::validate_section_property_value(
        mind_micro_biophysical::ObjectOpKind::VInit,
        value);
    const auto cell = static_cast<std::size_t>(gid);
    const auto begin = section_properties_.cell_section_offsets[cell];
    const auto end = section_properties_.cell_section_offsets[cell + 1];
    std::fill(section_properties_.v_init.begin() + static_cast<std::ptrdiff_t>(begin),
              section_properties_.v_init.begin() + static_cast<std::ptrdiff_t>(end),
              value);
}

double MicroFrontendModel::cell_v_init(int gid) const {
    require_morphology();
    (void) mind_micro_model::require_cell_template(morph_layout_, gid);
    const auto cell = static_cast<std::size_t>(gid);
    const auto begin = section_properties_.cell_section_offsets[cell];
    const auto end = section_properties_.cell_section_offsets[cell + 1];
    if (begin == end) {
        throw std::runtime_error("v_init is not set for gid=" + std::to_string(gid));
    }
    const double first = section_properties_.v_init[begin];
    if (!std::isfinite(first)) {
        throw std::runtime_error("v_init is not set for gid=" + std::to_string(gid));
    }
    for (std::size_t offset = begin + 1; offset < end; ++offset) {
        if (section_properties_.v_init[offset] != first) {
            throw std::runtime_error("cell v_init is not uniform for gid=" + std::to_string(gid));
        }
    }
    return first;
}

void MicroFrontendModel::set_section_group_property(
    int gid,
    const std::vector<std::size_t>& section_indices,
    mind_micro_biophysical::ObjectOpKind kind,
    double value) {
    require_morphology();
    mind_micro_biophysical::validate_section_property_value(kind, value);
    auto* values = kind == mind_micro_biophysical::ObjectOpKind::VInit ? &section_properties_.v_init
                 : kind == mind_micro_biophysical::ObjectOpKind::Cm    ? &section_properties_.cm
                                                                        : &section_properties_.ra;
    for (const auto section_index : section_indices) {
        (*values)[section_property_offset(gid, section_index)] = value;
    }
    microcircuit_built_ = false;
    core_initialized_ = false;
}

double MicroFrontendModel::section_group_property(
    int gid,
    const std::vector<std::size_t>& section_indices,
    mind_micro_biophysical::ObjectOpKind kind) const {
    require_morphology();
    if (section_indices.empty()) {
        throw std::runtime_error("section group resolved to no sections");
    }
    const auto* values = kind == mind_micro_biophysical::ObjectOpKind::VInit ? &section_properties_.v_init
                       : kind == mind_micro_biophysical::ObjectOpKind::Cm    ? &section_properties_.cm
                                                                              : &section_properties_.ra;
    const double first = (*values)[section_property_offset(gid, section_indices.front())];
    if (!std::isfinite(first)) {
        throw std::runtime_error(std::string("section ") +
                                 mind_micro_biophysical::section_property_name(kind) +
                                 " is not set for gid=" + std::to_string(gid));
    }
    for (const auto section_index : section_indices) {
        const double value = (*values)[section_property_offset(gid, section_index)];
        if (value != first) {
            throw std::runtime_error(std::string("section group ") +
                                     mind_micro_biophysical::section_property_name(kind) +
                                     " is not uniform");
        }
    }
    return first;
}

std::pair<std::string, std::string> MicroFrontendModel::resolve_ion_field(
    const std::string& field) const {
    for (const auto& metadata : mechanism_catalog_.mechanisms()) {
        if (!metadata.name.ends_with("_ion")) {
            continue;
        }
        const auto field_it = metadata.field_index_by_name.find(field);
        if (field_it == metadata.field_index_by_name.end()) {
            continue;
        }
        const auto field_index = static_cast<std::size_t>(field_it->second);
        if (field_index >= metadata.field_data_offsets.size() ||
            metadata.field_data_offsets[field_index] < 0) {
            continue;
        }
        return {metadata.name.substr(0, metadata.name.size() - 4), metadata.name};
    }
    throw std::runtime_error("ion range variable is not registered: " + field);
}

std::string MicroFrontendModel::resolve_ion_mechanism(const std::string& ion_mechanism) const {
    const auto& metadata = mechanism_catalog_.require(ion_mechanism);
    if (!metadata.name.ends_with("_ion")) {
        throw std::runtime_error(ion_mechanism + " is not an ion mechanism");
    }
    return metadata.name.substr(0, metadata.name.size() - 4);
}

void MicroFrontendModel::set_section_group_ion_range(
    int gid,
    const std::vector<std::size_t>& section_indices,
    std::string field,
    mind_micro_biophysical::ParamValue value) {
    require_morphology();
    const auto normalized = mind_micro_biophysical::normalize_section_group_indices(
        morph_layout_,
        gid,
        section_indices);
    const auto [ion, ion_mechanism] = resolve_ion_field(field);
    ion_range_overrides_.push_back(IonRangeOverride{
        .gid = gid,
        .section_indices = normalized,
        .ion = ion,
        .ion_mechanism = ion_mechanism,
        .field = std::move(field),
        .value = std::move(value),
    });
    runtime_backend_.reset();
}

double MicroFrontendModel::section_group_ion_range(
    int gid,
    const std::vector<std::size_t>& section_indices,
    const std::string& field) const {
    require_morphology();
    const auto normalized = mind_micro_biophysical::normalize_section_group_indices(
        morph_layout_,
        gid,
        section_indices);
    for (auto it = ion_range_overrides_.rbegin(); it != ion_range_overrides_.rend(); ++it) {
        if (it->gid == gid && it->field == field && it->section_indices == normalized) {
            return supplied_param_value_for_segment(it->value, static_cast<int>(normalized.front()), 0);
        }
    }
    const auto [_, ion_mechanism] = resolve_ion_field(field);
    const auto& metadata = mechanism_catalog_.require(ion_mechanism);
    const auto field_it = metadata.field_index_by_name.find(field);
    return metadata.fields[static_cast<std::size_t>(field_it->second)].default_value;
}

int MicroFrontendModel::section_group_ion_style(
    int gid,
    const std::vector<std::size_t>& section_indices,
    const std::string& ion_mechanism) const {
    require_morphology();
    const auto ion = resolve_ion_mechanism(ion_mechanism);
    const auto normalized = mind_micro_biophysical::normalize_section_group_indices(
        morph_layout_,
        gid,
        section_indices);
    for (auto it = ion_style_overrides_.rbegin(); it != ion_style_overrides_.rend(); ++it) {
        if (it->gid == gid && it->ion == ion && it->section_indices == normalized) {
            return it->style;
        }
    }
    return -1;
}

int MicroFrontendModel::set_section_group_ion_style(
    int gid,
    const std::vector<std::size_t>& section_indices,
    const std::string& ion_mechanism,
    int c_style,
    int e_style,
    int einit,
    int eadvance,
    int cinit) {
    require_morphology();
    const int old_style = section_group_ion_style(gid, section_indices, ion_mechanism);
    const auto ion = resolve_ion_mechanism(ion_mechanism);
    const auto normalized = mind_micro_biophysical::normalize_section_group_indices(
        morph_layout_,
        gid,
        section_indices);
    const int style = c_style +
                      (1 << nrn_ion_rev_shift) * e_style +
                      nrn_ion_rev_init * einit +
                      nrn_ion_rev_advance * eadvance +
                      nrn_ion_conc_init * cinit;
    ion_style_overrides_.push_back(IonStyleOverride{
        .gid = gid,
        .section_indices = normalized,
        .ion = ion,
        .ion_mechanism = ion_mechanism,
        .style = style,
    });
    runtime_backend_.reset();
    return old_style;
}

int MicroFrontendModel::insert_mechanism(int gid,
                                         std::vector<std::size_t> section_indices,
                                         std::optional<double> loc,
                                         const std::string& mech,
                                         mind_micro_biophysical::ParamList params) {
    require_morphology();
    if (mech.empty()) {
        throw std::runtime_error("mechanism name must not be empty");
    }
    if (!mechanism_catalog_.contains(mech) && !default_mechanism_metadata_loaded_) {
        load_default_mechanism_metadata();
    }
    const auto& metadata = mechanism_catalog_.require(mech);
    const bool is_density = metadata.kind == mind_micro_biophysical::MechanismKind::Density;
    const bool is_event_target = metadata.kind == mind_micro_biophysical::MechanismKind::EventTarget;
    if (metadata.kind == mind_micro_biophysical::MechanismKind::ArtificialCell) {
        throw std::runtime_error("artificial-cell mechanisms must be registered without a section location");
    }
    if (section_indices.empty()) {
        throw std::runtime_error("mechanism insert target resolved to no sections");
    }
    std::vector<int> normalized_sections;
    normalized_sections.reserve(section_indices.size());
    for (const auto section_index : section_indices) {
        normalized_sections.push_back(static_cast<int>(
            mind_micro_model::require_section_index(morph_layout_, gid, section_index)));
    }
    if (loc.has_value() && (!std::isfinite(*loc) || *loc < 0.0 || *loc > 1.0)) {
        throw std::runtime_error("located mechanism insert loc must be finite and in [0, 1]");
    }
    if (is_density && loc.has_value()) {
        throw std::runtime_error("density mechanism '" + mech + "' must be inserted on a section set, not a point location");
    }
    if (is_event_target && !loc.has_value()) {
        throw std::runtime_error("event target mechanism '" + mech + "' requires a located section, e.g. sec(0.5).insert(...)");
    }
    if (is_event_target && normalized_sections.size() != 1) {
        throw std::runtime_error("event target mechanism '" + mech + "' requires exactly one section location");
    }

    const int insert_id = static_cast<int>(core_mechanism_builder_.inserts.size());
    int target_id = -1;
    if (is_event_target) {
        target_id = network_registry_.register_event_target(
            gid,
            normalized_sections.front(),
            *loc,
            mech,
            metadata.has_net_receive ? metadata.net_receive_weight_count : 0);
    }

    auto& block = block_for_metadata(metadata.id, metadata.runtime_type);
    const std::size_t instance_begin = block.instances.size();
    if (is_density) {
        for (const int section_index : normalized_sections) {
            int segment_index = 0;
            mind_micro_biophysical::for_each_section_segment_original_node(
                morph_layout_,
                gid,
                static_cast<std::size_t>(section_index),
                [&](std::size_t node_index) {
                    block.instances.push_back(CoreMechanismInstance{
                        .insert_id = insert_id,
                        .insert_index = static_cast<std::size_t>(insert_id),
                        .node_index = static_cast<int>(node_index),
                        .gid = gid,
                        .section_index = section_index,
                        .segment_index = segment_index,
                        .event_target_id = -1,
                    });
                    ++segment_index;
                });
        }
    } else if (is_event_target) {
        const int section_index = normalized_sections.front();
        block.instances.push_back(CoreMechanismInstance{
            .insert_id = insert_id,
            .insert_index = static_cast<std::size_t>(insert_id),
            .node_index = resolve_cached_original_node(gid, section_index, *loc),
            .gid = gid,
            .section_index = section_index,
            .segment_index = 0,
            .event_target_id = target_id,
        });
    }
    const std::size_t instance_end = block.instances.size();

    const std::size_t param_begin = core_mechanism_builder_.param_overrides.size();
    append_local_param_overrides(metadata, params, core_mechanism_builder_.param_overrides);
    const std::size_t param_end = core_mechanism_builder_.param_overrides.size();
    core_mechanism_builder_.inserts.push_back(CoreMechanismInsert{
        .id = insert_id,
        .metadata_id = metadata.id,
        .runtime_type = metadata.runtime_type,
        .placement = is_event_target ? MechanismPlacementKind::Location
                                     : MechanismPlacementKind::SectionSet,
        .gid = gid,
        .loc = loc.value_or(std::numeric_limits<double>::quiet_NaN()),
        .section_index = normalized_sections.front(),
        .param_begin = param_begin,
        .param_end = param_end,
        .target_id = target_id,
        .instance_begin = instance_begin,
        .instance_end = instance_end,
    });
    microcircuit_built_ = false;
    core_initialized_ = false;
    return insert_id;
}

int MicroFrontendModel::register_artificial_cell(const std::string& mech,
                                                 mind_micro_biophysical::ParamList params) {
    if (mech.empty()) {
        throw std::runtime_error("artificial-cell mechanism name must not be empty");
    }
    if (!mechanism_catalog_.contains(mech) && !default_mechanism_metadata_loaded_) {
        load_default_mechanism_metadata();
    }
    const auto& metadata = mechanism_catalog_.require(mech);
    if (metadata.kind != mind_micro_biophysical::MechanismKind::ArtificialCell) {
        throw std::runtime_error("mechanism '" + mech + "' is not an ARTIFICIAL_CELL");
    }
    const int insert_id = static_cast<int>(core_mechanism_builder_.inserts.size());
    const int target_id = network_registry_.register_artificial_cell(
        mech,
        metadata.has_net_receive ? metadata.net_receive_weight_count : 0);
    auto& block = block_for_metadata(metadata.id, metadata.runtime_type);
    const std::size_t instance_begin = block.instances.size();
    block.instances.push_back(CoreMechanismInstance{
        .insert_id = insert_id,
        .insert_index = static_cast<std::size_t>(insert_id),
        .node_index = -1,
        .gid = -1,
        .section_index = -1,
        .segment_index = 0,
        .event_target_id = target_id,
    });
    const std::size_t instance_end = block.instances.size();
    const std::size_t param_begin = core_mechanism_builder_.param_overrides.size();
    append_local_param_overrides(metadata, params, core_mechanism_builder_.param_overrides);
    const std::size_t param_end = core_mechanism_builder_.param_overrides.size();
    core_mechanism_builder_.inserts.push_back(CoreMechanismInsert{
        .id = insert_id,
        .metadata_id = metadata.id,
        .runtime_type = metadata.runtime_type,
        .placement = MechanismPlacementKind::Artificial,
        .gid = -1,
        .loc = std::numeric_limits<double>::quiet_NaN(),
        .section_index = -1,
        .param_begin = param_begin,
        .param_end = param_end,
        .target_id = target_id,
        .instance_begin = instance_begin,
        .instance_end = instance_end,
    });
    microcircuit_built_ = false;
    core_initialized_ = false;
    return insert_id;
}

std::string MicroFrontendModel::mechanism_mech(int insert_id) const {
    const auto idx = require_insert_index(insert_id);
    return mechanism_catalog_.require(core_mechanism_builder_.inserts[idx].metadata_id).name;
}

int MicroFrontendModel::mechanism_gid(int insert_id) const {
    return core_mechanism_builder_.inserts[require_insert_index(insert_id)].gid;
}

int MicroFrontendModel::mechanism_section_index(int insert_id) const {
    return core_mechanism_builder_.inserts[require_insert_index(insert_id)].section_index;
}

double MicroFrontendModel::mechanism_loc(int insert_id) const {
    return core_mechanism_builder_.inserts[require_insert_index(insert_id)].loc;
}

MechanismPlacementKind MicroFrontendModel::mechanism_placement(int insert_id) const {
    return core_mechanism_builder_.inserts[require_insert_index(insert_id)].placement;
}

double MicroFrontendModel::mechanism_scalar(int insert_id, const std::string& key) const {
    const auto params = params_for_insert(insert_id);
    const auto* value = find_param(params, key);
    if (value == nullptr || !std::holds_alternative<double>(*value)) {
        throw std::runtime_error("mechanism scalar parameter is not available: " + key);
    }
    return std::get<double>(*value);
}

void MicroFrontendModel::set_mechanism_scalar(int insert_id, const std::string& key, double value) {
    if (!std::isfinite(value)) {
        throw std::runtime_error("mechanism scalar parameter must be finite");
    }
    const auto idx = require_insert_index(insert_id);
    const auto& insert = core_mechanism_builder_.inserts[idx];
    const auto& metadata = mechanism_catalog_.require(insert.metadata_id);
    const auto field_index_it = metadata.field_index_by_name.find(key);
    if (field_index_it == metadata.field_index_by_name.end()) {
        throw std::runtime_error(
            "parameter '" + key + "' is not a RANGE/PARAMETER/STATE/ASSIGNED field of mechanism '" +
            metadata.name + "'");
    }
    const int field_index = field_index_it->second;
    const auto& field = metadata.fields[static_cast<std::size_t>(field_index)];
    if (field.is_global) {
        mind_sim::micro::sim::core_set_global_parameter(metadata.name, field.name, value);
        runtime_backend_.reset();
        return;
    }
    const auto begin = insert.param_begin;
    const auto end = insert.param_end;
    for (std::size_t i = begin; i < end; ++i) {
        if (core_mechanism_builder_.param_overrides[i].field_index == field_index) {
            core_mechanism_builder_.param_overrides[i].value = value;
            microcircuit_built_ = false;
            core_initialized_ = false;
            return;
        }
    }
    core_mechanism_builder_.param_overrides.insert(
        core_mechanism_builder_.param_overrides.begin() + static_cast<std::ptrdiff_t>(end),
        CoreParamOverride{
            .field_index = field_index,
            .data_offset = metadata.field_data_offsets.at(static_cast<std::size_t>(field_index)),
            .value = value,
        });
    for (std::size_t insert_index = idx; insert_index < core_mechanism_builder_.inserts.size(); ++insert_index) {
        auto& item = core_mechanism_builder_.inserts[insert_index];
        if (insert_index == idx) {
            item.param_end += 1;
        } else {
            item.param_begin += 1;
            item.param_end += 1;
        }
    }
    microcircuit_built_ = false;
    core_initialized_ = false;
}

int MicroFrontendModel::register_gid_source(int gid,
                                            const VariableRef& source,
                                            std::optional<double> threshold) {
    if (source.kind != VariableRef::Kind::LocationVoltage && source.kind != VariableRef::Kind::Location) {
        throw std::runtime_error("register_gid_source requires a section location variable reference");
    }
    return network_registry_.register_gid_source(
        gid,
        source.gid,
        source.section_index,
        source.x,
        threshold);
}

int MicroFrontendModel::gid_connect(int gid, int post_insert_id, double weight, double delay) {
    const auto idx = require_insert_index(post_insert_id);
    const int target_id = core_mechanism_builder_.inserts[idx].target_id;
    if (target_id < 0) {
        throw std::runtime_error("gid_connect target must be a located mechanism insert");
    }
    return network_registry_.register_gid_connect(gid, target_id, weight, delay);
}

int MicroFrontendModel::register_spike_input_source() {
    return network_registry_.register_spike_input_source();
}

int MicroFrontendModel::spike_input_connect(int spike_input_id,
                                            int post_insert_id,
                                            double weight,
                                            double delay) {
    const auto idx = require_insert_index(post_insert_id);
    const int target_id = core_mechanism_builder_.inserts[idx].target_id;
    if (target_id < 0) {
        throw std::runtime_error("spike input target must be a located mechanism insert");
    }
    return network_registry_.register_spike_input_netcon(spike_input_id, target_id, weight, delay);
}

int MicroFrontendModel::spike_input_runtime_index(int spike_input_id) const {
    return network_registry_.spike_input_runtime_index(spike_input_id);
}

std::size_t MicroFrontendModel::netcon_weight_count(int connection_id) const {
    return network_registry_.netcon_weight_count(connection_id);
}

double MicroFrontendModel::netcon_weight(int connection_id, int array_index) const {
    return network_registry_.get_netcon_weight(connection_id, array_index);
}

void MicroFrontendModel::set_netcon_weight(int connection_id, int array_index, double value) {
    network_registry_.set_netcon_weight(connection_id, array_index, value);
}

double MicroFrontendModel::netcon_delay(int connection_id) const {
    return network_registry_.get_netcon_delay(connection_id);
}

void MicroFrontendModel::set_netcon_delay(int connection_id, double value) {
    network_registry_.set_netcon_delay(connection_id, value);
}

double MicroFrontendModel::netcon_threshold(int connection_id) const {
    return network_registry_.get_netcon_threshold(connection_id);
}

void MicroFrontendModel::set_netcon_threshold(int connection_id, double value) {
    network_registry_.set_netcon_threshold(connection_id, value);
}

int MicroFrontendModel::netcon_runtime_index(int connection_id) const {
    return network_registry_.get_netcon_runtime_index(connection_id);
}

int MicroFrontendModel::netcon_target_event_target_id(int connection_id) const {
    return network_registry_.get_netcon_target_event_target_id(connection_id);
}

int MicroFrontendModel::netcon_source_event_target_id(int connection_id) const {
    return network_registry_.get_netcon_source_event_target_id(connection_id);
}

double* MicroFrontendModel::resolve_variable_pointer(const VariableRef& ref, const char* action) const {
    require_morphology();
    if (core_neuron_data_->threads.empty()) {
        throw std::runtime_error(std::string("micro variable ") + action +
                                 " requires built CoreNEURON thread data");
    }
    auto& thread = core_neuron_data_->threads.front();

    auto resolve_voltage_node = [&]() {
        const int original_node = resolve_original_node(morph_layout_, ref.gid, ref.section_index, ref.x);
        const int node_index = runtime_node_for_original(original_node);
        const auto values = thread.actual_v();
        if (node_index < 0 || static_cast<std::size_t>(node_index) >= values.size()) {
            throw std::runtime_error(std::string("variable ") + action +
                                     " resolved an invalid node index");
        }
        return node_index;
    };

    auto resolve_field_offset = [](const mind_micro_biophysical::MechanismMetadata& metadata,
                                   const std::string& var,
                                   int array_index) {
        const auto field_it = metadata.field_index_by_name.find(var);
        if (field_it == metadata.field_index_by_name.end()) {
            throw std::runtime_error("mechanism variable is not a CoreNEURON data field: " +
                                     metadata.name + "." + var);
        }
        const auto& field = metadata.fields[static_cast<std::size_t>(field_it->second)];
        if (array_index >= field.array_size) {
            throw std::runtime_error("mechanism variable array index is out of range for: " +
                                     metadata.name + "." + var);
        }
        const int field_offset =
            metadata.field_data_offsets[static_cast<std::size_t>(field_it->second)] +
            std::max(array_index, 0);
        if (field_offset < 0) {
            throw std::runtime_error("mechanism variable is not stored in CoreNEURON data: " +
                                     metadata.name + "." + var);
        }
        return field_offset;
    };

    if (ref.kind == VariableRef::Kind::LocationVoltage ||
        (ref.kind == VariableRef::Kind::Location && ref.mech == "global" && ref.var == "v")) {
        const int node_index = resolve_voltage_node();
        auto values = thread.actual_v();
        return values.data() + static_cast<std::size_t>(node_index);
    }
    if (ref.kind == VariableRef::Kind::Location) {
        if (ref.mech.empty() || ref.var.empty()) {
            throw std::runtime_error(std::string("mechanism variable ") + action +
                                     " requires non-empty mechanism and variable names");
        }
        const int node_index = resolve_voltage_node();
        const auto type_it = core_neuron_data_->mechanism_type.find(ref.mech);
        if (type_it == core_neuron_data_->mechanism_type.end()) {
            throw std::runtime_error(std::string("mechanism variable ") + action +
                                     " requested an unbuilt mechanism: " + ref.mech);
        }
        const int type = type_it->second;
        const auto ml_it = std::find_if(
            thread.memb_lists.begin(),
            thread.memb_lists.end(),
            [type](const auto& ml) { return ml.type == type; });
        if (ml_it == thread.memb_lists.end() || ml_it->ml.data == nullptr) {
            throw std::runtime_error(std::string("mechanism variable ") + action +
                                     " found no Memb_list for: " + ref.mech);
        }
        const auto& metadata = mechanism_catalog_.require(ref.mech);
        const int field_offset = resolve_field_offset(metadata, ref.var, ref.array_index);
        const auto instance_it = std::find(
            ml_it->nodeindices.begin(),
            ml_it->nodeindices.end(),
            node_index);
        if (instance_it == ml_it->nodeindices.end()) {
            throw std::runtime_error(std::string("mechanism variable ") + action +
                                     " found no instance for: " +
                                     ref.mech + "." + ref.var);
        }
        const auto instance_index = static_cast<std::size_t>(
            std::distance(ml_it->nodeindices.begin(), instance_it));
        const auto padded_count = static_cast<std::size_t>(ml_it->ml._nodecount_padded);
        const auto data_index = static_cast<std::size_t>(field_offset) * padded_count + instance_index;
        return ml_it->ml.data + data_index;
    }
    if (ref.kind == VariableRef::Kind::Mechanism) {
        const auto insert_index = require_insert_index(ref.insert_id);
        const auto& insert = core_mechanism_builder_.inserts[insert_index];
        if (insert.placement == MechanismPlacementKind::SectionSet) {
            throw std::runtime_error(std::string("object variable ") + action +
                                     " requires a point process or artificial cell");
        }
        const auto& metadata = mechanism_catalog_.require(insert.metadata_id);
        const int field_offset = resolve_field_offset(metadata, ref.var, ref.array_index);
        const auto ml_it = std::find_if(
            thread.memb_lists.begin(),
            thread.memb_lists.end(),
            [&](const auto& ml) { return ml.type == insert.runtime_type; });
        if (ml_it == thread.memb_lists.end() || ml_it->ml.data == nullptr) {
            throw std::runtime_error(std::string("object variable ") + action +
                                     " found no Memb_list for: " + metadata.name);
        }
        const auto instance_index = insert.instance_begin;
        const auto data_index = static_cast<std::size_t>(field_offset) *
                                    static_cast<std::size_t>(ml_it->ml._nodecount_padded) +
                                instance_index;
        return ml_it->ml.data + data_index;
    }
    throw std::runtime_error(std::string(action) +
                             " currently supports voltage, density variables, and object variables");
}

double MicroFrontendModel::read_variable(const VariableRef& ref) const {
    return *resolve_variable_pointer(ref, "read");
}

double* MicroFrontendModel::variable_pointer(const VariableRef& ref) {
    return resolve_variable_pointer(ref, "record");
}

const mind_micro_model::CellTemplateMorphLayout& MicroFrontendModel::morph_layout() const {
    require_morphology();
    return morph_layout_;
}

void MicroFrontendModel::require_morphology() const {
    if (!has_morphology_) {
        throw std::runtime_error("micro morphology has not been built");
    }
}

std::size_t MicroFrontendModel::require_insert_index(int insert_id) const {
    if (insert_id < 0 || static_cast<std::size_t>(insert_id) >= core_mechanism_builder_.inserts.size()) {
        throw std::runtime_error("unknown mechanism insert id=" + std::to_string(insert_id));
    }
    return static_cast<std::size_t>(insert_id);
}

mind_micro_biophysical::ParamList MicroFrontendModel::params_for_insert(int insert_id) const {
    const auto idx = require_insert_index(insert_id);
    const auto& insert = core_mechanism_builder_.inserts[idx];
    const auto begin = insert.param_begin;
    const auto end = insert.param_end;
    mind_micro_biophysical::ParamList out;
    out.reserve(end - begin);
    const auto& metadata = mechanism_catalog_.require(insert.metadata_id);
    for (std::size_t i = begin; i < end; ++i) {
        const int field_index = core_mechanism_builder_.param_overrides[i].field_index;
        out.emplace_back(metadata.fields[static_cast<std::size_t>(field_index)].name,
                         core_mechanism_builder_.param_overrides[i].value);
    }
    return out;
}

void MicroFrontendModel::init_section_properties() {
    section_properties_.cell_section_offsets.clear();
    section_properties_.cell_section_offsets.reserve(static_cast<std::size_t>(morph_layout_.num_cells_total) + 1);
    section_properties_.cell_section_offsets.push_back(0);
    std::size_t total_sections = 0;
    for (int gid = 0; gid < morph_layout_.num_cells_total; ++gid) {
        total_sections += mind_micro_model::section_count(morph_layout_, gid);
        section_properties_.cell_section_offsets.push_back(total_sections);
    }
    section_properties_.v_init.assign(total_sections, std::numeric_limits<double>::quiet_NaN());
    section_properties_.cm.assign(total_sections, std::numeric_limits<double>::quiet_NaN());
    section_properties_.ra.assign(total_sections, std::numeric_limits<double>::quiet_NaN());
    section_location_cache_locs_.assign(total_sections, std::numeric_limits<double>::quiet_NaN());
    section_location_cache_nodes_.assign(total_sections, -1);
}

std::size_t MicroFrontendModel::section_property_offset(int gid, std::size_t section_index) const {
    const auto section = mind_micro_model::require_section_index(morph_layout_, gid, section_index);
    const auto cell = static_cast<std::size_t>(gid);
    return section_properties_.cell_section_offsets[cell] + section;
}

void MicroFrontendModel::refresh_mechanism_runtime_cache() {
    dparam_bindings_by_metadata_id_.clear();
    for (const auto& binding : mind_sim::micro::sim::core_dparam_bindings()) {
        if (binding.mechanism.empty() || !mechanism_catalog_.contains(binding.mechanism)) {
            continue;
        }
        const int metadata_id = mechanism_catalog_.metadata_id(binding.mechanism);
        dparam_bindings_by_metadata_id_[metadata_id].push_back(binding);
    }
}

const std::vector<mind_sim::micro::sim::CoreDParamBinding>&
MicroFrontendModel::dparam_bindings_for_metadata(int metadata_id) const {
    static const std::vector<mind_sim::micro::sim::CoreDParamBinding> empty;
    const auto it = dparam_bindings_by_metadata_id_.find(metadata_id);
    return it == dparam_bindings_by_metadata_id_.end() ? empty : it->second;
}

int MicroFrontendModel::resolve_cached_original_node(int gid, int section_index, double loc) {
    const auto offset = section_property_offset(gid, static_cast<std::size_t>(section_index));
    if (section_location_cache_nodes_[offset] >= 0 &&
        section_location_cache_locs_[offset] == loc) {
        return section_location_cache_nodes_[offset];
    }
    const int node = resolve_original_node(morph_layout_, gid, section_index, loc);
    section_location_cache_locs_[offset] = loc;
    section_location_cache_nodes_[offset] = node;
    return node;
}

int MicroFrontendModel::runtime_node_for_original(int original_node) const {
    if (original_node < 0) {
        return original_node;
    }
    return runtime_node_by_original_[static_cast<std::size_t>(original_node)];
}

void MicroFrontendModel::reset_core_node_storage() {
    auto& nt = core_neuron_data_->threads.front();
    coreneuron::nrn_nthread = 1;
    const int cell_permute =
        mind_sim::micro::sim::cell_permute_for_device(core_neuron_data_->device_config.kind);
    coreneuron::corenrn_param.cell_interleave_permute =
        static_cast<unsigned>(cell_permute);
    coreneuron::interleave_permute_type = cell_permute;
    coreneuron::cellorder_nwarp = static_cast<int>(coreneuron::corenrn_param.nwarp);
    coreneuron::use_solve_interleave = coreneuron::interleave_permute_type != 0;
    coreneuron::destroy_interleave_info();
    coreneuron::create_interleave_info();

    auto parent_for_permute = morph_parent_index_;
    int* const runtime_order = coreneuron::interleave_order(
        nt.id,
        nt.ncell,
        nt.end,
        parent_for_permute.data());
    runtime_node_by_original_.assign(runtime_order, runtime_order + nt.end);
    delete[] runtime_order;

    nt.allocate_node_data(morph_layout_.nnode);
    nt.node_permutation = runtime_node_by_original_;
    nt.v_parent_index.assign(morph_layout_.nnode, -1);
    auto actual_a = nt.actual_a();
    auto actual_b = nt.actual_b();
    auto area = nt.actual_area();
    for (std::size_t original = 0; original < morph_layout_.nnode; ++original) {
        const auto runtime = static_cast<std::size_t>(runtime_node_by_original_[original]);
        const int parent = morph_parent_index_[original];
        nt.v_parent_index[runtime] = parent >= 0 ? runtime_node_by_original_[static_cast<std::size_t>(parent)] : -1;
        actual_a[runtime] = axial_a_ra1_[original];
        actual_b[runtime] = axial_b_ra1_[original];
        area[runtime] = morph_area_[original];
    }
}

int MicroFrontendModel::intern_mechanism_type(const std::string& name) {
    auto it = core_neuron_data_->mechanism_type.find(name);
    if (it != core_neuron_data_->mechanism_type.end()) {
        return it->second;
    }
    const int type = mind_sim::micro::sim::core_mechanism_type(name);
    core_neuron_data_->mechanism_type.emplace(name, type);
    if (core_neuron_data_->mechanisms.size() <= static_cast<std::size_t>(type)) {
        core_neuron_data_->mechanisms.resize(static_cast<std::size_t>(type) + 1);
    }
    core_neuron_data_->mechanisms[static_cast<std::size_t>(type)] =
        mind_sim::micro::sim::MechanismRuntimeInfo{
            .type = type,
            .metadata_id = -1,
            .name = name,
            .is_event_target = false,
        };
    return type;
}

CoreMechanismBlockBuilder& MicroFrontendModel::block_for_metadata(int metadata_id, int runtime_type) {
    auto it = core_mechanism_builder_.block_index_by_metadata_id.find(metadata_id);
    if (it != core_mechanism_builder_.block_index_by_metadata_id.end()) {
        return core_mechanism_builder_.blocks[it->second];
    }
    const auto index = core_mechanism_builder_.blocks.size();
    core_mechanism_builder_.block_index_by_metadata_id.emplace(metadata_id, index);
    core_mechanism_builder_.blocks.push_back(CoreMechanismBlockBuilder{
        .metadata_id = metadata_id,
        .runtime_type = runtime_type,
        .instances = {},
    });
    return core_mechanism_builder_.blocks.back();
}

void MicroFrontendModel::apply_section_axial_resistance() {
    if (core_neuron_data_->threads.empty()) {
        throw std::runtime_error("cannot apply axial resistance before morphology core data exists");
    }
    auto& nt = core_neuron_data_->threads.front();
    if (axial_a_ra1_.size() != morph_layout_.nnode || axial_b_ra1_.size() != morph_layout_.nnode ||
        axial_ri_ra1_.size() != morph_layout_.nnode || axial_a_scale_.size() != morph_layout_.nnode) {
        throw std::runtime_error("morphology axial coefficient cache is inconsistent with core node count");
    }
    auto actual_a = nt.actual_a();
    auto actual_b = nt.actual_b();

    for (int gid = 0; gid < morph_layout_.num_cells_total; ++gid) {
        const auto& tpl = mind_micro_model::require_cell_template(morph_layout_, gid);
        const auto section_count_value = mind_micro_model::section_count(morph_layout_, gid);
        for (std::size_t section_index = 0; section_index < section_count_value; ++section_index) {
            const double ra = section_properties_.ra[section_property_offset(gid, section_index)];
            if (!std::isfinite(ra) || ra <= 0.0) {
                throw std::runtime_error("section Ra must be positive before building CoreNEURON data");
            }
            const auto base = tpl.section_node_base_id[section_index];
            const auto nseg = tpl.section_nseg[section_index];
            for (std::int32_t local_node = 0; local_node <= nseg; ++local_node) {
                const auto original = mind_micro_model::map_template_node_to_original(
                    morph_layout_,
                    gid,
                    base + local_node);
                const auto node = static_cast<std::size_t>(runtime_node_for_original(original));
                const double ri_ra1 = axial_ri_ra1_[static_cast<std::size_t>(original)];
                const double rinv =
                    (!std::isfinite(ri_ra1) || ri_ra1 >= infinite_resistance_cutoff_Mohm)
                        ? 0.0
                        : (1.0 / (ri_ra1 * ra));
                const int parent = morph_parent_index_[static_cast<std::size_t>(original)];
                if (parent >= 0) {
                    const double parent_area = morph_area_[static_cast<std::size_t>(parent)];
                    actual_a[node] =
                        -1.0e2 * axial_a_scale_[static_cast<std::size_t>(original)] * rinv /
                        parent_area;
                } else {
                    actual_a[node] = 0.0;
                }
                actual_b[node] = -1.0e2 * rinv / morph_area_[static_cast<std::size_t>(original)];
            }
        }
    }
}

void MicroFrontendModel::rebuild_core_mechanisms() {
    {
        mind_sim::micro::sim::ensure_core_mechanisms_registered();
        mind_sim::micro::sim::core_verify_ion_charges_defined();
    }

    auto& nt = core_neuron_data_->threads[0];
    {
        apply_section_axial_resistance();
    }
    {
        nt.memb_lists.clear();
        nt.tml_storage.clear();
        nt.pntproc_storage.clear();
        nt.pntproc_event_target_ids.clear();
        nt.presyn_storage.clear();
        nt.presyn_helpers.clear();
        nt.input_presyns.clear();
        nt.netcon_storage.clear();
        nt.netcon_presyn_order.clear();
        nt.netcon_weight_counts.clear();
        nt.weight_storage.clear();
        nt.pnt2presyn_ix_storage.clear();
        nt.pnt2presyn_ix_offsets.clear();
        nt.pnt2presyn_ix_ptrs.clear();
        nt.shadow_rhs.clear();
        nt.shadow_d.clear();
        nt.truncate_mechanism_data();
        nt.vdata_storage.clear();
        nt.random123_storage.clear();
        nt.n_real_output = 0;
        nt.shadow_rhs_cnt = 0;
        core_neuron_data_->mechanism_type.clear();
        core_neuron_data_->mechanisms.clear();
    }

    std::size_t point_process_count = 0;
    std::size_t vdata_count = 0;
    std::size_t random123_count = 0;
    for (const auto& block : core_mechanism_builder_.blocks) {
        const auto& metadata = mechanism_catalog_.require(block.metadata_id);
        if (metadata.kind != mind_micro_biophysical::MechanismKind::Density) {
            point_process_count += block.instances.size();
        }
        for (const auto& binding : dparam_bindings_for_metadata(metadata.id)) {
            if (binding.kind == mind_sim::micro::sim::CoreDParamKind::PointProcess ||
                binding.kind == mind_sim::micro::sim::CoreDParamKind::NetSend ||
                binding.kind == mind_sim::micro::sim::CoreDParamKind::BbcorePointer) {
                vdata_count += block.instances.size();
            } else if (binding.kind == mind_sim::micro::sim::CoreDParamKind::Random) {
                vdata_count += block.instances.size();
                random123_count += block.instances.size();
            }
        }
    }
    nt.pntproc_storage.reserve(point_process_count);
    nt.pntproc_event_target_ids.reserve(point_process_count);
    nt.vdata_storage.reserve(vdata_count);
    nt.random123_storage.reserve(random123_count);

    const auto make_node_memb_list = [&](const std::string& name, const std::vector<int>& nodes) {
        mind_sim::micro::sim::CoreMembList ml;
        ml.type = intern_mechanism_type(name);
        ml.name = name;
        ml.nodeindices = nodes;
        ml.ml.nodecount = static_cast<int>(ml.nodeindices.size());
        ml.ml._nodecount_padded = padded_node_count(ml.ml.nodecount, ml.type);
        const auto dparam_size =
            coreneuron::corenrn.get_prop_dparam_size()[static_cast<std::size_t>(ml.type)];
        ml.pdata.assign(static_cast<std::size_t>(dparam_size) *
                            static_cast<std::size_t>(ml.ml._nodecount_padded),
                        0);
        return ml;
    };

    mind_sim::micro::sim::CoreMembList cap;
    std::vector<double> cap_by_runtime_node(morph_layout_.nnode, 0.0);
    std::size_t cap_data_size = 0;
    {
        cap.type = intern_mechanism_type("capacitance");
        cap.name = "capacitance";
        cap.metadata_id = -1;
        std::vector<unsigned char> cap_node_present(morph_layout_.nnode, 0);
        std::size_t cap_node_count = 0;
        for (int gid = 0; gid < morph_layout_.num_cells_total; ++gid) {
            const auto section_count_value = mind_micro_model::section_count(morph_layout_, gid);
            for (std::size_t section_index = 0; section_index < section_count_value; ++section_index) {
                const double cm = section_properties_.cm[section_property_offset(gid, section_index)];
                if (!std::isfinite(cm) || cm <= 0.0) {
                    throw std::runtime_error("section cm must be positive before building CoreNEURON data");
                }
                mind_micro_biophysical::for_each_section_segment_original_node(
                    morph_layout_,
                    gid,
                    section_index,
                    [&](std::size_t node_index) {
                        const auto runtime_node =
                            static_cast<std::size_t>(runtime_node_for_original(static_cast<int>(node_index)));
                        cap_node_count += cap_node_present[runtime_node] == 0 ? 1 : 0;
                        cap_node_present[runtime_node] = 1;
                        cap_by_runtime_node[runtime_node] = cm;
                    });
            }
        }
        cap.nodeindices.reserve(cap_node_count);
        for (std::size_t runtime_node = 0; runtime_node < cap_node_present.size(); ++runtime_node) {
            if (cap_node_present[runtime_node] != 0) {
                cap.nodeindices.push_back(static_cast<int>(runtime_node));
            }
        }
        cap.ml.nodecount = static_cast<int>(cap.nodeindices.size());
        cap.ml._nodecount_padded = padded_node_count(cap.ml.nodecount, cap.type);
        cap_data_size = mechanism_data_size(cap);
    }

    std::vector<std::string> required_ions;
    std::unordered_map<std::string, std::vector<unsigned char>> ion_node_marks_by_ion;
    {
        for (const auto& block : core_mechanism_builder_.blocks) {
            const auto& metadata = mechanism_catalog_.require(block.metadata_id);
            const auto& bindings = dparam_bindings_for_metadata(metadata.id);
            std::vector<std::string> block_ions;
            for (const auto& binding : bindings) {
                if ((binding.kind == mind_sim::micro::sim::CoreDParamKind::IonVariable ||
                     binding.kind == mind_sim::micro::sim::CoreDParamKind::IonStyle) &&
                    !binding.ion.empty() &&
                    !contains_name(required_ions, binding.ion)) {
                    required_ions.push_back(binding.ion);
                }
                if ((binding.kind == mind_sim::micro::sim::CoreDParamKind::IonVariable ||
                     binding.kind == mind_sim::micro::sim::CoreDParamKind::IonStyle) &&
                    !binding.ion.empty() &&
                    !contains_name(block_ions, binding.ion)) {
                    block_ions.push_back(binding.ion);
                }
            }
            for (const auto& ion : block_ions) {
                auto& node_marks = ion_node_marks_by_ion[ion];
                if (node_marks.empty()) {
                    node_marks.assign(morph_layout_.nnode, 0);
                }
                for (const auto& instance : block.instances) {
                    if (instance.node_index < 0) {
                        continue;
                    }
                    node_marks[static_cast<std::size_t>(runtime_node_for_original(instance.node_index))] = 1;
                }
            }
        }
    }

    std::unordered_map<std::string, PendingIonMembList> ion_memb_lists;
    std::unordered_map<std::string, std::vector<int>> ion_styles_by_node;
    ion_memb_lists.reserve(required_ions.size());
    ion_styles_by_node.reserve(required_ions.size());
    {
        for (const auto& ion : required_ions) {
            const std::string ion_mech = ion + "_ion";
            const auto& ion_metadata = mechanism_catalog_.require(ion_mech);
            auto node_marks_it = ion_node_marks_by_ion.find(ion);
            if (node_marks_it == ion_node_marks_by_ion.end()) {
                throw std::runtime_error("missing CoreNEURON ion node set for " + ion);
            }
            const auto& node_marks = node_marks_it->second;
            const auto ion_node_count = static_cast<std::size_t>(
                std::count(node_marks.begin(), node_marks.end(), static_cast<unsigned char>(1)));
            std::vector<int> ion_nodes;
            ion_nodes.reserve(ion_node_count);
            for (std::size_t runtime_node = 0; runtime_node < node_marks.size(); ++runtime_node) {
                if (node_marks[runtime_node] != 0) {
                    ion_nodes.push_back(static_cast<int>(runtime_node));
                }
            }
            auto ion_ml = make_node_memb_list(ion_mech, ion_nodes);
            ion_ml.metadata_id = ion_metadata.id;
            PendingIonMembList pending{};
            pending.field_index_by_name.reserve(ion_metadata.fields.size());
            pending.row_by_node.assign(morph_layout_.nnode, -1);
            for (std::size_t row = 0; row < ion_ml.nodeindices.size(); ++row) {
                pending.row_by_node[static_cast<std::size_t>(ion_ml.nodeindices[row])] =
                    static_cast<int>(row);
            }
            for (std::size_t field_index = 0; field_index < ion_metadata.fields.size(); ++field_index) {
                const auto& field = ion_metadata.fields[field_index];
                pending.field_index_by_name.emplace(field.name, static_cast<int>(field_index));
            }
            pending.ml = std::move(ion_ml);
            ion_styles_by_node.emplace(ion, std::vector<int>(morph_layout_.nnode, 0));
            ion_memb_lists.emplace(ion, std::move(pending));
        }
    }

    std::size_t exact_mechanism_data_size = cap_data_size;
    for (const auto& ion : required_ions) {
        exact_mechanism_data_size += mechanism_data_size(ion_memb_lists.at(ion).ml);
    }
    for (const auto& block : core_mechanism_builder_.blocks) {
        const auto& metadata = mechanism_catalog_.require(block.metadata_id);
        const int type = intern_mechanism_type(metadata.name);
        exact_mechanism_data_size += mechanism_data_size(
            type,
            padded_node_count(static_cast<int>(block.instances.size()), type));
    }
    ThreadDataAppender data_appender(nt, exact_mechanism_data_size);

    {
        double* cap_data = data_appender.allocate(cap, cap_data_size);
        for (std::size_t instance_index = 0; instance_index < cap.nodeindices.size(); ++instance_index) {
            cap_data[instance_index] =
                cap_by_runtime_node[static_cast<std::size_t>(cap.nodeindices[instance_index])];
        }
        data_appender.append(std::move(cap));
    }

    {
        for (const auto& ion : required_ions) {
            auto& pending = ion_memb_lists.at(ion);
            auto& ion_ml = pending.ml;
            double* ion_data = data_appender.allocate(ion_ml, mechanism_data_size(ion_ml));
            const auto& ion_metadata = mechanism_catalog_.require(ion_ml.metadata_id);
            for (std::size_t field_index = 0; field_index < ion_metadata.fields.size(); ++field_index) {
                const auto& field = ion_metadata.fields[field_index];
                const auto base = field_index *
                                  static_cast<std::size_t>(ion_ml.ml._nodecount_padded);
                for (std::size_t row = 0; row < ion_ml.nodeindices.size(); ++row) {
                    ion_data[base + row] = field.default_value;
                }
            }
        }
    }

    {
    for (const auto& block : core_mechanism_builder_.blocks) {
        const int metadata_id = block.metadata_id;
        const auto& metadata = mechanism_catalog_.require(metadata_id);
        const auto& bindings = dparam_bindings_for_metadata(metadata.id);
        struct IonPromotionPlan {
            std::vector<int>* styles{};
            int conc{0};
            int rev{0};
            int write_flags{0};
        };
        struct IonPromotionSpec {
            int conc{0};
            int rev{0};
            int write_flags{0};
        };
        struct IonAssignmentPlan {
            int mechanism_field_index{-1};
            double* ion_data{};
            std::size_t ion_data_base{0};
            const std::vector<int>* row_by_node{};
        };
        std::unordered_map<std::string, IonPromotionSpec> promotion_by_ion;
        std::vector<IonAssignmentPlan> ion_assignments;
        for (const auto& binding : bindings) {
            if (binding.kind == mind_sim::micro::sim::CoreDParamKind::IonStyle) {
                auto& promotion = promotion_by_ion[binding.ion];
                promotion.conc = std::max(promotion.conc, binding.ion_conc_style);
                promotion.rev = std::max(promotion.rev, binding.ion_rev_style);
                if (binding.ion_write_interior) {
                    promotion.write_flags |= nrn_ion_write_interior;
                }
                if (binding.ion_write_exterior) {
                    promotion.write_flags |= nrn_ion_write_exterior;
                }
                continue;
            }
            if (binding.kind != mind_sim::micro::sim::CoreDParamKind::IonVariable) {
                continue;
            }
            if (binding.ion_conc_style > 0 || binding.ion_rev_style > 0 ||
                binding.ion_write_interior || binding.ion_write_exterior) {
                auto& promotion = promotion_by_ion[binding.ion];
                promotion.conc = std::max(promotion.conc, binding.ion_conc_style);
                promotion.rev = std::max(promotion.rev, binding.ion_rev_style);
                if (binding.ion_write_interior) {
                    promotion.write_flags |= nrn_ion_write_interior;
                }
                if (binding.ion_write_exterior) {
                    promotion.write_flags |= nrn_ion_write_exterior;
                }
            }
            if (binding.ion_field < 0 || binding.mechanism_field.empty()) {
                continue;
            }
            const auto mechanism_field_it = metadata.field_index_by_name.find(binding.mechanism_field);
            if (mechanism_field_it == metadata.field_index_by_name.end()) {
                continue;
            }
            auto ion_ml_it = ion_memb_lists.find(binding.ion);
            if (ion_ml_it == ion_memb_lists.end()) {
                throw std::runtime_error("missing inferred ion data storage for " + binding.ion);
            }
            const auto& ion_field_index = ion_ml_it->second.field_index_by_name.find(binding.mechanism_field);
            if (ion_field_index == ion_ml_it->second.field_index_by_name.end() ||
                ion_field_index->second != binding.ion_field) {
                throw std::runtime_error("ion field metadata mismatch for '" + binding.ion +
                                         "' field '" + binding.mechanism_field + "'");
            }
            auto& ion_ml = ion_ml_it->second.ml;
            ion_assignments.push_back(IonAssignmentPlan{
                .mechanism_field_index = mechanism_field_it->second,
                .ion_data = thread_memb_data(nt, ion_ml),
                .ion_data_base = static_cast<std::size_t>(binding.ion_field) *
                                 static_cast<std::size_t>(ion_ml.ml._nodecount_padded),
                .row_by_node = &ion_ml_it->second.row_by_node,
            });
        }
        std::vector<IonPromotionPlan> ion_promotions;
        ion_promotions.reserve(promotion_by_ion.size());
        for (const auto& [ion, promotion] : promotion_by_ion) {
            auto style_it = ion_styles_by_node.find(ion);
            if (style_it == ion_styles_by_node.end()) {
                throw std::runtime_error("missing inferred ion style storage for " + ion);
            }
            ion_promotions.push_back(IonPromotionPlan{
                .styles = &style_it->second,
                .conc = promotion.conc,
                .rev = promotion.rev,
                .write_flags = promotion.write_flags,
            });
        }
        if (ion_promotions.empty() && ion_assignments.empty()) {
            continue;
        }

        for (const auto& instance : block.instances) {
            if (instance.node_index < 0) {
                continue;
            }
            const auto node = static_cast<std::size_t>(runtime_node_for_original(instance.node_index));
            for (const auto& promotion : ion_promotions) {
                auto& style = promotion.styles->at(node);
                style = promoted_ion_style(style, promotion.conc, promotion.rev);
                style |= promotion.write_flags;
            }
            if (ion_assignments.empty()) {
                continue;
            }
            const auto params = param_span_for_insert_index(core_mechanism_builder_, instance.insert_index);
            for (const auto& assignment : ion_assignments) {
                const auto* supplied_value = find_param(params, assignment.mechanism_field_index);
                if (supplied_value == nullptr) {
                    continue;
                }
                const int row = assignment.row_by_node->at(node);
                if (row < 0) {
                    throw std::runtime_error("ion parameter assignment resolved no ion row");
                }
                assignment.ion_data[assignment.ion_data_base + static_cast<std::size_t>(row)] = supplied_param_value_for_segment(
                    *supplied_value,
                    instance.section_index,
                    instance.segment_index);
            }
        }
    }
    }

    {
        for (const auto& override : ion_range_overrides_) {
            auto ion_it = ion_memb_lists.find(override.ion);
            if (ion_it == ion_memb_lists.end()) {
                continue;
            }
            const auto& metadata = mechanism_catalog_.require(override.ion_mechanism);
            const auto field_it = metadata.field_index_by_name.find(override.field);
            if (field_it == metadata.field_index_by_name.end()) {
                throw std::runtime_error("ion override field disappeared from metadata: " + override.field);
            }
            const auto field_index = static_cast<std::size_t>(field_it->second);
            const int field_data_offset = metadata.field_data_offsets[field_index];
            if (field_data_offset < 0) {
                throw std::runtime_error("ion override field is not a CoreNEURON data field: " + override.field);
            }
            auto& ion_ml = ion_it->second.ml;
            double* ion_data = thread_memb_data(nt, ion_ml);
            const auto field_base = static_cast<std::size_t>(field_data_offset) *
                                    static_cast<std::size_t>(ion_ml.ml._nodecount_padded);
            const auto& tpl = mind_micro_model::require_cell_template(morph_layout_, override.gid);
            for (const auto section_index : override.section_indices) {
                const auto sec_index = mind_micro_model::require_section_index(
                    morph_layout_,
                    override.gid,
                    section_index);
                const auto base = tpl.section_node_base_id[sec_index];
                const auto nseg = tpl.section_nseg[sec_index];
                for (std::int32_t segment = 0; segment < nseg; ++segment) {
                    const auto original = mind_micro_model::map_template_node_to_original(
                        morph_layout_,
                        override.gid,
                        base + segment);
                    const auto runtime_node = static_cast<std::size_t>(runtime_node_for_original(original));
                    const int row = ion_it->second.row_by_node.at(runtime_node);
                    if (row < 0) {
                        throw std::runtime_error("ion override resolved no ion row for " + override.ion);
                    }
                    ion_data[field_base + static_cast<std::size_t>(row)] = supplied_param_value_for_segment(
                        override.value,
                        static_cast<int>(section_index),
                        segment);
                }
            }
        }
    }

    {
        for (const auto& override : ion_style_overrides_) {
            auto style_it = ion_styles_by_node.find(override.ion);
            if (style_it == ion_styles_by_node.end()) {
                continue;
            }
            const auto& tpl = mind_micro_model::require_cell_template(morph_layout_, override.gid);
            for (const auto section_index : override.section_indices) {
                const auto sec_index = mind_micro_model::require_section_index(
                    morph_layout_,
                    override.gid,
                    section_index);
                const auto base = tpl.section_node_base_id[sec_index];
                const auto nseg = tpl.section_nseg[sec_index];
                for (std::int32_t segment = 0; segment < nseg; ++segment) {
                    const auto original = mind_micro_model::map_template_node_to_original(
                        morph_layout_,
                        override.gid,
                        base + segment);
                    const auto runtime_node = static_cast<std::size_t>(runtime_node_for_original(original));
                    auto& style = style_it->second[runtime_node];
                    style = (style & nrn_ion_write_mask) + override.style;
                }
            }
        }
    }

    std::unordered_map<int, IonThreadStorage> ion_storage_by_type;
    {
        for (const auto& ion : required_ions) {
            auto ion_it = ion_memb_lists.find(ion);
            if (ion_it == ion_memb_lists.end()) {
                throw std::runtime_error("missing inferred ion memb list for " + ion);
            }
            auto& ion_ml = ion_it->second.ml;
            if (!ion_ml.pdata.empty()) {
                auto style_it = ion_styles_by_node.find(ion);
                if (style_it == ion_styles_by_node.end()) {
                    throw std::runtime_error("missing inferred ion style values for " + ion);
                }
                for (std::size_t row = 0; row < ion_ml.nodeindices.size(); ++row) {
                    const auto node = static_cast<std::size_t>(ion_ml.nodeindices[row]);
                    ion_ml.pdata[row] = static_cast<coreneuron::Datum>(style_it->second[node]);
                }
            }
            const int ion_type = ion_ml.type;
            const std::size_t offset = ion_ml.thread_data_offset;
            ion_storage_by_type.emplace(ion_type, IonThreadStorage{
                                                      .type = ion_type,
                                                      .offset = offset,
                                                      .padded_count = ion_ml.ml._nodecount_padded,
                                                      .row_by_node = &ion_it->second.row_by_node,
                                                  });
            data_appender.append(std::move(ion_ml));
        }
    }

    {
    for (const auto& block : core_mechanism_builder_.blocks) {
        const int metadata_id = block.metadata_id;
        const auto& metadata = mechanism_catalog_.require(metadata_id);
        const auto& instances = block.instances;
        mind_sim::micro::sim::CoreMembList ml;
        ml.type = intern_mechanism_type(metadata.name);
        core_neuron_data_->mechanisms[static_cast<std::size_t>(ml.type)].metadata_id = metadata.id;
        core_neuron_data_->mechanisms[static_cast<std::size_t>(ml.type)].is_event_target =
            metadata.kind != mind_micro_biophysical::MechanismKind::Density;
        ml.metadata_id = metadata.id;
        ml.name = metadata.name;
        ml.is_event_target = metadata.kind != mind_micro_biophysical::MechanismKind::Density;

        std::vector<int> runtime_nodes(instances.size(), -1);
        for (std::size_t instance_slot = 0; instance_slot < instances.size(); ++instance_slot) {
            runtime_nodes[instance_slot] = runtime_node_for_original(instances[instance_slot].node_index);
        }
        std::vector<std::size_t> row_order(instances.size());
        std::iota(row_order.begin(), row_order.end(), 0);
        if (metadata.kind != mind_micro_biophysical::MechanismKind::ArtificialCell) {
            std::stable_sort(row_order.begin(), row_order.end(), [&](std::size_t lhs, std::size_t rhs) {
                return runtime_nodes[lhs] < runtime_nodes[rhs];
            });
        }
        std::vector<std::size_t> row_by_instance(instances.size(), 0);
        for (std::size_t row = 0; row < row_order.size(); ++row) {
            row_by_instance[row_order[row]] = row;
        }

        ml.nodeindices.reserve(instances.size());
        for (const auto instance_slot : row_order) {
            ml.nodeindices.push_back(runtime_nodes[instance_slot]);
        }
        ml.ml.nodecount = static_cast<int>(ml.nodeindices.size());
        ml.ml._nodecount_padded = padded_node_count(ml.ml.nodecount, ml.type);
        const auto param_size =
            coreneuron::corenrn.get_prop_param_size()[static_cast<std::size_t>(ml.type)];
        const auto dparam_size =
            coreneuron::corenrn.get_prop_dparam_size()[static_cast<std::size_t>(ml.type)];
        double* ml_data = data_appender.allocate(
            ml,
            static_cast<std::size_t>(param_size) *
                static_cast<std::size_t>(ml.ml._nodecount_padded));
        ml.pdata.assign(static_cast<std::size_t>(dparam_size) *
                            static_cast<std::size_t>(ml.ml._nodecount_padded),
                        0);
        ml.bind();
        std::vector<unsigned char> ion_bound_fields(metadata.fields.size(), 0);
        const auto& metadata_bindings = dparam_bindings_for_metadata(metadata.id);
        for (const auto& binding : metadata_bindings) {
            if (binding.kind == mind_sim::micro::sim::CoreDParamKind::IonVariable &&
                !binding.mechanism_field.empty()) {
                const auto field_it = metadata.field_index_by_name.find(binding.mechanism_field);
                if (field_it != metadata.field_index_by_name.end()) {
                    ion_bound_fields[static_cast<std::size_t>(field_it->second)] = 1;
                }
            }
        }
        const auto& field_data_offsets = metadata.field_data_offsets;
        if (field_data_offsets.size() != metadata.fields.size()) {
            throw std::runtime_error("compiled mechanism field offset count mismatch for " +
                                     metadata.name);
        }
        for (std::size_t field_index = 0; field_index < metadata.fields.size(); ++field_index) {
            const auto& field = metadata.fields[field_index];
            const int field_data_offset = field_data_offsets[field_index];
            if (field_data_offset < 0) {
                continue;
            }
            for (int element = 0; element < field.array_size; ++element) {
                const auto field_data_base =
                    static_cast<std::size_t>(field_data_offset + element) *
                    static_cast<std::size_t>(ml.ml._nodecount_padded);
                std::fill_n(ml_data + field_data_base, instances.size(), field.default_value);
            }
        }
        for (std::size_t row = 0; row < row_order.size(); ++row) {
            const auto& instance = instances[row_order[row]];
            const auto params = param_span_for_insert_index(core_mechanism_builder_, instance.insert_index);
            for (std::size_t param_index = params.begin; param_index < params.end; ++param_index) {
                const auto& override = core_mechanism_builder_.param_overrides[param_index];
                const int field_index = override.field_index;
                if (field_index < 0 ||
                    static_cast<std::size_t>(field_index) >= metadata.fields.size()) {
                    throw std::runtime_error("mechanism parameter field index is out of range for '" +
                                             metadata.name + "'");
                }
                const auto& field = metadata.fields[static_cast<std::size_t>(field_index)];
                const int field_data_offset = override.data_offset;
                const auto& supplied_value = override.value;
                if (field_data_offset >= 0) {
                    const double value = supplied_param_value_for_segment(
                        supplied_value,
                        instance.section_index,
                        instance.segment_index);
                    for (int element = 0; element < field.array_size; ++element) {
                        const auto field_data_base =
                            static_cast<std::size_t>(field_data_offset + element) *
                            static_cast<std::size_t>(ml.ml._nodecount_padded);
                        ml_data[field_data_base + row] = value;
                    }
                    continue;
                }
                if (field.is_global && std::holds_alternative<double>(supplied_value)) {
                    mind_sim::micro::sim::core_set_global_parameter(
                        metadata.name,
                        field.name,
                        std::get<double>(supplied_value));
                    continue;
                }
                const bool supplied_default_scalar =
                    std::holds_alternative<double>(supplied_value) &&
                    std::get<double>(supplied_value) == field.default_value;
                if (ion_bound_fields[static_cast<std::size_t>(field_index)] != 0) {
                    continue;
                }
                if (!supplied_default_scalar) {
                    throw std::runtime_error(
                        "mechanism parameter '" + field.name + "' for '" + metadata.name +
                        "' is not a per-instance CoreNEURON variable");
                }
            }
        }

        int pntproc_base = -1;
        if (metadata.kind != mind_micro_biophysical::MechanismKind::Density) {
            pntproc_base = static_cast<int>(nt.pntproc_storage.size());
            for (std::size_t instance_slot = 0; instance_slot < instances.size(); ++instance_slot) {
                const auto& instance = instances[instance_slot];
                nt.pntproc_storage.push_back(coreneuron::Point_process{
                    ._i_instance = static_cast<int>(row_by_instance[instance_slot]),
                    ._type = static_cast<short>(ml.type),
                    ._tid = 0,
                });
                nt.pntproc_event_target_ids.push_back(instance.event_target_id);
            }
        }

        const auto& bindings = dparam_bindings_for_metadata(metadata.id);
        for (const auto& binding : bindings) {
            if (binding.dparam_index < 0 || binding.dparam_index >= dparam_size) {
                throw std::runtime_error("generated CoreNEURON dparam binding index out of range for " +
                                         metadata.name);
            }
            const auto base = static_cast<std::size_t>(binding.dparam_index) *
                              static_cast<std::size_t>(ml.ml._nodecount_padded);
            for (std::size_t row = 0; row < row_order.size(); ++row) {
                const auto instance_slot = row_order[row];
                const auto& instance = instances[instance_slot];
                const int runtime_node = runtime_nodes[instance_slot];
                auto& slot = ml.pdata[base + row];
                switch (binding.kind) {
                    case mind_sim::micro::sim::CoreDParamKind::IonVariable: {
                        if (binding.ion_field < 0) {
                            throw std::runtime_error(
                                "CoreNEURON ion dparam field is unresolved for mechanism '" +
                                metadata.name + "' ion '" + binding.ion +
                                "'; registered CoreNEURON dparam semantics are insufficient");
                        }
                        const int ion_type =
                            mind_sim::micro::sim::core_mechanism_type(binding.ion + "_ion");
                        const auto ion_it = ion_storage_by_type.find(ion_type);
                        if (ion_it == ion_storage_by_type.end()) {
                            throw std::runtime_error("missing CoreNEURON ion storage for " + binding.ion);
                        }
                        if (instance.node_index < 0) {
                            throw std::runtime_error("ion dparam binding requires a membrane node");
                        }
                        const int ion_row =
                            (*ion_it->second.row_by_node)[static_cast<std::size_t>(runtime_node)];
                        if (ion_row < 0) {
                            throw std::runtime_error("ion dparam binding resolved no ion row for " +
                                                     binding.ion);
                        }
                        slot = static_cast<coreneuron::Datum>(
                            ion_it->second.offset +
                            static_cast<std::size_t>(binding.ion_field) *
                                static_cast<std::size_t>(ion_it->second.padded_count) +
                            static_cast<std::size_t>(ion_row));
                        break;
                    }
                    case mind_sim::micro::sim::CoreDParamKind::IonStyle: {
                        if (instance.node_index < 0) {
                            throw std::runtime_error("ion style dparam binding requires a membrane node");
                        }
                        const auto style_it = ion_styles_by_node.find(binding.ion);
                        if (style_it == ion_styles_by_node.end()) {
                            throw std::runtime_error("missing inferred CoreNEURON ion style for " +
                                                     binding.ion);
                        }
                        const auto style = static_cast<coreneuron::Datum>(
                            style_it->second[static_cast<std::size_t>(runtime_node)]);
                        slot = style;
                        break;
                    }
                    case mind_sim::micro::sim::CoreDParamKind::Area:
                        slot = static_cast<coreneuron::Datum>(
                            nt.node_data_stride * 5 +
                            static_cast<std::size_t>(runtime_node >= 0 ? runtime_node : 0));
                        break;
                    case mind_sim::micro::sim::CoreDParamKind::PointProcess:
                        if (pntproc_base < 0) {
                            throw std::runtime_error("pntproc dparam binding on non-point mechanism");
                        }
                        slot = append_vdata(
                            nt,
                            &nt.pntproc_storage[static_cast<std::size_t>(
                                pntproc_base + static_cast<int>(instance_slot))]);
                        break;
                    case mind_sim::micro::sim::CoreDParamKind::Random:
                        slot = append_random123(nt);
                        break;
                    case mind_sim::micro::sim::CoreDParamKind::NetSend:
                    case mind_sim::micro::sim::CoreDParamKind::BbcorePointer:
                        slot = append_vdata(nt, nullptr);
                        break;
                    case mind_sim::micro::sim::CoreDParamKind::Unsupported:
                        slot = 0;
                        break;
                }
            }
        }

        data_appender.append(std::move(ml));
    }
    }
    nt.n_pntproc = static_cast<int>(nt.pntproc_storage.size());
    {
        int shadow_rhs_count = 0;
        for (const auto& ml : nt.memb_lists) {
            if (!ml.is_event_target) {
                continue;
            }
            const bool has_membrane_node = std::any_of(
                ml.nodeindices.begin(),
                ml.nodeindices.end(),
                [](int node_index) { return node_index >= 0; });
            if (has_membrane_node) {
                shadow_rhs_count = std::max(shadow_rhs_count, ml.ml.nodecount);
            }
        }
        if (shadow_rhs_count > 0) {
            const auto padded_shadow_count = coreneuron::nrn_soa_padded_size(shadow_rhs_count, 0);
            nt.shadow_rhs.assign(static_cast<std::size_t>(padded_shadow_count), 0.0);
            nt.shadow_d.assign(static_cast<std::size_t>(padded_shadow_count), 0.0);
            nt.shadow_rhs_cnt = shadow_rhs_count;
        }
    }
    {
        for (const auto& net_receive_entry : coreneuron::corenrn.get_net_buf_receive()) {
            const int type = net_receive_entry.second;
            for (auto& ml : nt.memb_lists) {
                if (ml.type != type) {
                    continue;
                }
                int pnt_offset = 0;
                bool found_offset = false;
                for (std::size_t pnt_index = 0; pnt_index < nt.pntproc_storage.size(); ++pnt_index) {
                    if (nt.pntproc_storage[pnt_index]._type == type) {
                        pnt_offset = static_cast<int>(pnt_index);
                        found_offset = true;
                        break;
                    }
                }
                ml.allocate_net_receive_buffer(std::max(8, ml.ml.nodecount),
                                               found_offset ? pnt_offset : 0);
            }
        }
    }
    {
        build_tml_from_registered_execution_order(nt);
    }
    {
        core_neuron_data_->bind();
    }
}

void MicroFrontendModel::rebuild_core_network() {
    auto& nt = core_neuron_data_->threads[0];
    {
        nt.presyn_storage.clear();
        nt.presyn_helpers.clear();
        nt.input_presyns.clear();
        nt.netcon_storage.clear();
        nt.netcon_weight_counts.clear();
        nt.net_send_buffer.clear();
        nt.pnt2presyn_ix_storage.clear();
        nt.pnt2presyn_ix_offsets.clear();
        nt.pnt2presyn_ix_ptrs.clear();
        nt.n_real_output = 0;
        nt.weight_storage = network_registry_.event_edge_weights();
    }

    std::vector<int> event_target_to_pntproc(network_registry_.event_targets().size(), -1);
    {
        for (std::size_t i = 0; i < nt.pntproc_storage.size(); ++i) {
            const int event_target_id = nt.pntproc_event_target_ids[i];
            event_target_to_pntproc[static_cast<std::size_t>(event_target_id)] = static_cast<int>(i);
        }
    }

    std::vector<int> source_slot_to_presyn(network_registry_.event_source_slots().size(), -1);
    std::vector<int> source_slot_to_input_presyn(network_registry_.event_source_slots().size(), -1);
    {
        for (std::size_t source_slot = 0; source_slot < network_registry_.event_source_slots().size(); ++source_slot) {
            const auto& source = network_registry_.event_source_slots()[source_slot];
            if (source.source_kind == mind_micro_network::NetConSourceKind::SpikeInput) {
                coreneuron::InputPreSyn input;
                source_slot_to_input_presyn[source_slot] = static_cast<int>(nt.input_presyns.size());
                network_registry_.set_spike_input_runtime_index(
                    source.spike_input_id,
                    static_cast<int>(nt.input_presyns.size()));
                nt.input_presyns.push_back(input);
                continue;
            }

            coreneuron::PreSyn presyn;
            presyn.threshold_ = source.threshold;
            if (source.source_kind == mind_micro_network::NetConSourceKind::RealCell) {
                presyn.gid_ = source.real_source.gid;
                presyn.thvar_index_ = runtime_node_for_original(resolve_cached_original_node(
                    source.real_source.gid,
                    source.real_source.section_index,
                    source.real_source.loc));
                ++nt.n_real_output;
            } else {
                const int pntproc_index =
                    event_target_to_pntproc[static_cast<std::size_t>(source.source_event_target_id)];
                presyn.pntsrc_ = nt.pntproc_storage.data() + pntproc_index;
                presyn.gid_ = network_registry_.event_target_gid(source.source_event_target_id);
            }
            source_slot_to_presyn[source_slot] = static_cast<int>(nt.presyn_storage.size());
            nt.presyn_storage.push_back(presyn);
            nt.presyn_helpers.push_back(coreneuron::PreSynHelper{});
        }
    }

    std::vector<int> target_slot_to_pntproc(network_registry_.event_target_slots().size(), -1);
    {
        for (std::size_t target_slot = 0; target_slot < network_registry_.event_target_slots().size(); ++target_slot) {
            const int event_target_id = network_registry_.event_target_slots()[target_slot].event_target_id;
            const int pntproc_index =
                event_target_to_pntproc[static_cast<std::size_t>(event_target_id)];
            target_slot_to_pntproc[target_slot] = pntproc_index;
        }
    }

    std::vector<int> source_slot_nc_count(network_registry_.event_source_slots().size(), 0);
    std::vector<int> source_slot_nc_index(network_registry_.event_source_slots().size(), -1);
    {
        for (const auto& edge : network_registry_.event_edges()) {
            source_slot_nc_count[static_cast<std::size_t>(edge.source_slot)] += 1;
        }
        int nc_cursor = 0;
        for (std::size_t source_slot = 0; source_slot < source_slot_to_presyn.size(); ++source_slot) {
            const int presyn_index = source_slot_to_presyn[source_slot];
            if (presyn_index < 0) {
                continue;
            }
            auto& presyn = nt.presyn_storage[static_cast<std::size_t>(presyn_index)];
            presyn.nc_index_ = nc_cursor;
            presyn.nc_cnt_ = source_slot_nc_count[source_slot];
            source_slot_nc_index[source_slot] = nc_cursor;
            nc_cursor += source_slot_nc_count[source_slot];
        }
        for (std::size_t source_slot = 0; source_slot < source_slot_to_input_presyn.size(); ++source_slot) {
            const int input_presyn_index = source_slot_to_input_presyn[source_slot];
            if (input_presyn_index < 0) {
                continue;
            }
            auto& input = nt.input_presyns[static_cast<std::size_t>(input_presyn_index)];
            input.nc_index_ = nc_cursor;
            input.nc_cnt_ = source_slot_nc_count[source_slot];
            source_slot_nc_index[source_slot] = nc_cursor;
            nc_cursor += source_slot_nc_count[source_slot];
        }
    }
    std::vector<int> source_edge_seen(network_registry_.event_source_slots().size(), 0);
    {
        nt.netcon_storage.resize(network_registry_.event_edges().size());
        nt.netcon_presyn_order.assign(network_registry_.event_edges().size(), -1);
        nt.netcon_weight_counts.assign(network_registry_.event_edges().size(), 0);
        for (std::size_t edge_index = 0; edge_index < network_registry_.event_edges().size(); ++edge_index) {
            const auto& edge = network_registry_.event_edges()[edge_index];
            const int presyn_order_index =
                source_slot_nc_index[static_cast<std::size_t>(edge.source_slot)] +
                source_edge_seen[static_cast<std::size_t>(edge.source_slot)]++;
            const int netcon_index = static_cast<int>(edge_index);
            auto& netcon = nt.netcon_storage[static_cast<std::size_t>(netcon_index)];
            netcon.active_ = true;
            netcon.delay_ = edge.delay;
            netcon.target_ = nt.pntproc_storage.data() + target_slot_to_pntproc[edge.target_slot];
            netcon.u.weight_index_ = edge.weight_offset;
            nt.netcon_weight_counts[static_cast<std::size_t>(netcon_index)] = edge.weight_count;
            nt.netcon_presyn_order[static_cast<std::size_t>(presyn_order_index)] = netcon_index;
        }
    }

    {
        const auto capacity = static_cast<std::size_t>(core_neuron_data_->mechanism_capacity());
        std::vector<std::size_t> counts(capacity, 0);
        for (const auto& pnt : nt.pntproc_storage) {
            const auto type = static_cast<std::size_t>(pnt._type);
            counts[type] = std::max(counts[type], static_cast<std::size_t>(pnt._i_instance) + 1);
        }
        nt.pnt2presyn_ix_offsets.assign(capacity + 1, 0);
        for (std::size_t type = 0; type < capacity; ++type) {
            nt.pnt2presyn_ix_offsets[type + 1] = nt.pnt2presyn_ix_offsets[type] + counts[type];
        }
        nt.pnt2presyn_ix_storage.assign(nt.pnt2presyn_ix_offsets.back(), -1);
        for (std::size_t presyn_index = 0; presyn_index < nt.presyn_storage.size(); ++presyn_index) {
            const auto& presyn = nt.presyn_storage[presyn_index];
            if (presyn.pntsrc_ == nullptr) {
                continue;
            }
            const auto& pnt = *presyn.pntsrc_;
            const auto row = nt.pnt2presyn_ix_offsets[static_cast<std::size_t>(pnt._type)];
            nt.pnt2presyn_ix_storage[row + static_cast<std::size_t>(pnt._i_instance)] =
                static_cast<int>(presyn_index);
        }
    }

    nt.n_weight = static_cast<int>(nt.weight_storage.size());
    nt.n_netcon = static_cast<int>(nt.netcon_storage.size());
    nt.n_presyn = static_cast<int>(nt.presyn_storage.size());
    nt.n_input_presyn = static_cast<int>(nt.input_presyns.size());
    nt.net_send_buffer.assign(static_cast<std::size_t>(nt.n_real_output), 0);
    {
        core_neuron_data_->bind();
    }
}

}  // namespace mind_sim::micro::frontend
