#pragma once

#include "micro/frontend/model.hpp"
#include "bridge/sim/interfaces.hpp"
#include "coreneuron/io/output_spikes.hpp"
#include "cosim/hybrid_simulator.hpp"
#include "macro/frontend/local_connectivity.hpp"
#include "macro/frontend/network.hpp"
#include "macro/frontend/node_to_roi_map.hpp"
#include "macro/sim/runtime.hpp"
#include "morph/section_distance.hpp"
#include "morph/section_spec.hpp"
#include "io/result_hdf5.hpp"
#include "mind_mod/abi.hpp"
#include "mind_mod/rule_mod.hpp"
#include "utils/dynamic_library.hpp"
#include "utils/rule_source_common.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>

#include <cmath>
#include <cstddef>
#include <optional>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

namespace mind_sim::python_api::bindings {

using mind_sim::micro::frontend::MicroFrontendModel;
using mind_sim::micro::frontend::MechanismPlacementKind;
using mind_sim::micro::frontend::MorphologyTemplateSpec;
using mind_sim::micro::frontend::VariableRef;
using mind_sim::macro::frontend::Connectivity;
using mind_sim::macro::frontend::FieldExposureReducer;
using mind_sim::macro::frontend::GidRange;
using mind_sim::macro::frontend::LocalConnectivity;
using mind_sim::macro::frontend::LocalConnectivityEdge;
using mind_sim::macro::frontend::Network;
using mind_sim::macro::frontend::NodeToRoiMap;
using mind_sim::macro::frontend::ROI;
using mind_sim::macro::sim::CouplingRule;
using mind_sim::macro::sim::MacroRuntime;
using mind_sim::macro::sim::NeuralFieldRule;
using mind_sim::macro::sim::RegionRule;
using mind_sim::macro::sim::ScalarBuffer;
using mind_sim::bridge::sim::RandomStreamBinding;
using mind_sim::bridge::sim::RandomStreamRule;
using mind_micro_biophysical::ObjectOpKind;
using mind_micro_biophysical::ParamList;
using mind_micro_biophysical::ParamValue;
using mind_micro_frontend::SectionPt3d;
using mind_micro_frontend::SectionSpec;

template <typename SizeT>
inline int normalize_py_index(int index, SizeT size, const char* what) {
    const int n = static_cast<int>(size);
    int resolved = index;
    if (resolved < 0) {
        resolved += n;
    }
    if (resolved < 0 || resolved >= n) {
        throw nb::index_error((std::string(what) + " index out of range").c_str());
    }
    return resolved;
}

struct SectionList {
    std::shared_ptr<std::vector<SectionSpec>> sections{std::make_shared<std::vector<SectionSpec>>()};
};

struct SectionListIter {
    std::shared_ptr<std::vector<SectionSpec>> sections{};
    std::size_t next{0};

    SectionSpec& next_item() {
        if (!sections || next >= sections->size()) {
            throw nb::stop_iteration();
        }
        return sections->at(next++);
    }
};

inline std::vector<SectionPt3d> pt3d_from_obj(nb::handle obj) {
    std::vector<SectionPt3d> out;
    if (obj.is_none()) {
        return out;
    }
    for (nb::handle item : obj) {
        auto tuple = nb::cast<nb::tuple>(item);
        if (tuple.size() != 4) {
            throw std::runtime_error("pt3d entries must be (x_um, y_um, z_um, diam_um)");
        }
        out.push_back(SectionPt3d{
            .x_um = static_cast<float>(nb::cast<double>(tuple[0])),
            .y_um = static_cast<float>(nb::cast<double>(tuple[1])),
            .z_um = static_cast<float>(nb::cast<double>(tuple[2])),
            .diam_um = static_cast<float>(nb::cast<double>(tuple[3])),
        });
    }
    return out;
}

inline nb::list pt3d_to_list(const std::vector<SectionPt3d>& pts) {
    nb::list out;
    for (const auto& p : pts) {
        out.append(nb::make_tuple(p.x_um, p.y_um, p.z_um, p.diam_um));
    }
    return out;
}

inline std::vector<SectionSpec> parse_sections(nb::handle obj) {
    std::vector<SectionSpec> out;
    if (nb::isinstance<SectionList>(obj)) {
        const auto& list = nb::cast<const SectionList&>(obj);
        return *list.sections;
    }
    if (nb::isinstance<SectionSpec>(obj)) {
        out.push_back(nb::cast<SectionSpec>(obj));
        return out;
    }
    for (nb::handle item : obj) {
        if (!nb::isinstance<SectionSpec>(item)) {
            throw std::runtime_error("sections must contain mind_sim.section objects");
        }
        out.push_back(nb::cast<SectionSpec>(item));
    }
    return out;
}

inline SectionList make_section_list(std::vector<SectionSpec> sections) {
    SectionList out;
    out.sections = std::make_shared<std::vector<SectionSpec>>(std::move(sections));
    return out;
}

inline std::string section_name_from_arg(nb::handle section) {
    if (nb::isinstance<SectionSpec>(section)) {
        return nb::cast<const SectionSpec&>(section).name;
    }
    return nb::cast<std::string>(section);
}

inline mind_micro_biophysical::SegmentParamBatch make_segment_batch(
    const std::vector<int>& section_indices,
    const std::vector<std::vector<double>>& values) {
    if (section_indices.size() != values.size()) {
        throw std::runtime_error("segment_values: section_indices and values size mismatch");
    }
    mind_micro_biophysical::SegmentParamBatch out;
    out.section_indices = section_indices;
    out.value_offsets.reserve(values.size() + 1);
    out.value_offsets.push_back(0);
    for (const auto& row : values) {
        out.values.insert(out.values.end(), row.begin(), row.end());
        out.value_offsets.push_back(out.values.size());
    }
    return out;
}

struct SegmentValueBatchView {
    mind_micro_biophysical::SegmentParamBatch batch{};
    std::size_t size() const { return batch.section_indices.size(); }
};

inline ParamValue parse_param_value(nb::handle value) {
    if (nb::isinstance<SegmentValueBatchView>(value)) {
        return nb::cast<const SegmentValueBatchView&>(value).batch;
    }
    if (nb::isinstance<nb::str>(value)) {
        throw std::runtime_error("parameter value must be numeric or segment_values");
    }
    if (nb::isinstance<nb::int_>(value) || nb::isinstance<nb::float_>(value)) {
        return nb::cast<double>(value);
    }
    throw std::runtime_error("parameter value must be numeric or segment_values");
}

inline ParamList parse_kwargs(const nb::kwargs& kwargs) {
    ParamList params;
    params.reserve(kwargs.size());
    for (auto item : kwargs) {
        params.push_back({
            nb::cast<std::string>(item.first),
            parse_param_value(item.second),
        });
    }
    return params;
}

inline std::vector<MorphologyTemplateSpec> parse_morphology_templates(nb::handle obj) {
    std::vector<MorphologyTemplateSpec> out;
    for (nb::handle item : obj) {
        if (!nb::isinstance<nb::dict>(item)) {
            throw std::runtime_error("morphology templates must be dictionaries");
        }
        nb::dict d = nb::cast<nb::dict>(item);
        MorphologyTemplateSpec spec;
        spec.name = nb::cast<std::string>(d["name"]);
        spec.num_cells = nb::cast<int>(d["num_cells"]);
        auto sections = parse_sections(d["sections"]);
        spec.sections = std::make_shared<const std::vector<SectionSpec>>(std::move(sections));
        out.push_back(std::move(spec));
    }
    return out;
}

struct Sim;
struct PointProcessView;
struct ArtificialCellView;

struct RecorderBuffer {
    Sim* sim{};
    bool records_time{false};
    bool records_var{false};
    VariableRef ref{};
    std::vector<double> samples{};
};

inline std::vector<int> py_int_vector(nb::handle values, const char* what) {
    std::vector<int> out;
    for (nb::handle value: values) {
        out.push_back(nb::cast<int>(value));
    }
    if (out.empty() && PyObject_Length(values.ptr()) != 0) {
        throw std::runtime_error(std::string(what) + " could not be converted to int vector");
    }
    return out;
}

inline std::vector<double> py_double_vector(nb::handle values, const char* what) {
    std::vector<double> out;
    for (nb::handle value: values) {
        out.push_back(nb::cast<double>(value));
    }
    if (out.empty() && PyObject_Length(values.ptr()) != 0) {
        throw std::runtime_error(std::string(what) + " could not be converted to double vector");
    }
    return out;
}

inline int py_len(nb::handle value, const char* what) {
    const auto length = PyObject_Length(value.ptr());
    if (length < 0) {
        throw std::runtime_error(std::string("could not determine length of ") + what);
    }
    return static_cast<int>(length);
}

inline LocalConnectivity local_connectivity_from_csr(nb::handle matrix) {
    nb::handle csr = matrix;
    nb::object csr_owner;
    if (nb::hasattr(csr, "tocsr")) {
        csr_owner = nb::cast<nb::object>(csr.attr("tocsr")());
        csr = csr_owner;
    }
    if (!nb::hasattr(csr, "indptr") || !nb::hasattr(csr, "indices") ||
        !nb::hasattr(csr, "data")) {
        throw std::runtime_error("LocalConnectivity.from_csr expects a CSR-like matrix");
    }
    if (!nb::hasattr(csr, "shape")) {
        throw std::runtime_error("LocalConnectivity CSR matrix must expose shape");
    }
    auto shape = nb::cast<nb::tuple>(csr.attr("shape"));
    if (shape.size() != 2) {
        throw std::runtime_error("LocalConnectivity CSR matrix must be square");
    }
    const int rows = nb::cast<int>(shape[0]);
    const int cols = nb::cast<int>(shape[1]);
    if (rows != cols) {
        throw std::runtime_error("LocalConnectivity CSR matrix must be square");
    }
    if (nb::hasattr(csr, "copy")) {
        csr_owner = nb::cast<nb::object>(csr.attr("copy")());
        csr = csr_owner;
    }
    if (nb::hasattr(csr, "sum_duplicates")) {
        csr.attr("sum_duplicates")();
    }
    if (nb::hasattr(csr, "sort_indices")) {
        csr.attr("sort_indices")();
    }
    return LocalConnectivity(
        rows,
        py_int_vector(csr.attr("indptr"), "LocalConnectivity indptr"),
        py_int_vector(csr.attr("indices"), "LocalConnectivity indices"),
        py_double_vector(csr.attr("data"), "LocalConnectivity weights"));
}

inline LocalConnectivity local_connectivity_from_edges(int node_count, nb::handle edges) {
    std::vector<LocalConnectivityEdge> parsed;
    for (nb::handle item: edges) {
        auto edge = nb::cast<nb::tuple>(item);
        if (edge.size() != 3) {
            throw std::runtime_error("LocalConnectivity edges must be (target_node, source_node, weight)");
        }
        parsed.push_back(LocalConnectivityEdge{
            .target_node = nb::cast<int>(edge[0]),
            .source_node = nb::cast<int>(edge[1]),
            .weight = nb::cast<double>(edge[2]),
        });
    }
    return LocalConnectivity::from_edges(node_count, parsed);
}

inline LocalConnectivity local_connectivity_from_surface(
    nb::handle surface,
    std::optional<int> node_count) {
    if (!nb::hasattr(surface, "prepare_local_coupling")) {
        throw std::runtime_error("surface must provide prepare_local_coupling(node_count)");
    }
    int resolved_node_count = 0;
    if (node_count.has_value()) {
        resolved_node_count = *node_count;
    } else if (nb::hasattr(surface, "region_mapping")) {
        resolved_node_count = py_len(surface.attr("region_mapping"), "surface.region_mapping");
    } else if (nb::hasattr(surface, "vertices")) {
        resolved_node_count = py_len(surface.attr("vertices"), "surface.vertices");
    } else {
        throw std::runtime_error(
            "LocalConnectivity.from_surface requires node_count when the surface has no "
            "region_mapping or vertices");
    }
    return local_connectivity_from_csr(surface.attr("prepare_local_coupling")(resolved_node_count));
}

inline NodeToRoiMap node_to_roi_map_from_surface(
    nb::handle surface,
    std::optional<std::vector<double>> node_weights) {
    if (!nb::hasattr(surface, "region_mapping")) {
        throw std::runtime_error("NodeToRoiMap.from_surface expects surface.region_mapping");
    }
    return NodeToRoiMap(
        py_int_vector(surface.attr("region_mapping"), "surface.region_mapping"),
        node_weights.value_or(std::vector<double>{}));
}

struct Sim {
    Sim() = default;
    Sim(const Sim&) = delete;
    Sim& operator=(const Sim&) = delete;
    Sim(Sim&&) noexcept = default;
    Sim& operator=(Sim&&) noexcept = default;

    std::string name{"micro"};
    MicroFrontendModel model{};
    std::vector<std::weak_ptr<RecorderBuffer>> recorders{};

    void attach_recorder(const std::shared_ptr<RecorderBuffer>& recorder) {
        recorders.emplace_back(recorder);
    }

    void prune_recorders() {
        std::vector<std::weak_ptr<RecorderBuffer>> live;
        live.reserve(recorders.size());
        for (auto& weak : recorders) {
            if (!weak.expired()) {
                live.emplace_back(weak);
            }
        }
        recorders = std::move(live);
    }

    void sample_recorders() {
        std::vector<std::weak_ptr<RecorderBuffer>> live;
        live.reserve(recorders.size());
        for (auto& weak : recorders) {
            auto recorder = weak.lock();
            if (!recorder) {
                continue;
            }
            if (recorder->records_time) {
                recorder->samples.push_back(model.time());
            } else if (recorder->records_var) {
                recorder->samples.push_back(model.read_variable(recorder->ref));
            }
            live.emplace_back(recorder);
        }
        recorders = std::move(live);
    }

    int run_recorded(double tstop) {
        prune_recorders();
        if (recorders.empty()) {
            return model.run(tstop);
        }

        const double start = model.time();
        if (tstop < start - 1e-12) {
            throw std::runtime_error("run tstop must not be earlier than current simulation time");
        }
        const double runtime = tstop - start;
        if (runtime <= 1e-12) {
            return 0;
        }
        const double dt = model.dt();
        const double exact_steps = runtime / dt;
        const int sample_count = static_cast<int>(std::llround(exact_steps));
        if (sample_count < 0 ||
            std::abs(exact_steps - static_cast<double>(sample_count)) > 1e-9) {
            throw std::runtime_error(
                "recorded run requires tstop-current_time to be an integer multiple of dt");
        }

        std::vector<VariableRef> refs;
        std::vector<double*> sample_buffers;
        std::vector<std::shared_ptr<RecorderBuffer>> live_buffers;
        live_buffers.reserve(recorders.size());
        for (auto& weak : recorders) {
            auto recorder = weak.lock();
            if (!recorder) {
                continue;
            }
            if (recorder->samples.empty()) {
                if (recorder->records_time) {
                    recorder->samples.push_back(start);
                } else if (recorder->records_var) {
                    recorder->samples.push_back(model.read_variable(recorder->ref));
                }
            }
            const auto old_size = recorder->samples.size();
            recorder->samples.resize(old_size + static_cast<std::size_t>(sample_count));
            if (recorder->records_time) {
                for (int step = 0; step < sample_count; ++step) {
                    recorder->samples[old_size + static_cast<std::size_t>(step)] =
                        start + (static_cast<double>(step + 1) * dt);
                }
            } else if (recorder->records_var) {
                refs.push_back(recorder->ref);
                sample_buffers.push_back(recorder->samples.data() + old_size);
            }
            live_buffers.push_back(std::move(recorder));
        }

        const int recorded = model.continue_run_with_recording(
            runtime,
            refs,
            sample_buffers,
            sample_count);
        if (recorded != sample_count) {
            throw std::runtime_error("CoreNEURON trajectory sample count mismatch: expected " +
                                     std::to_string(sample_count) + ", got " +
                                     std::to_string(recorded));
        }
        return 0;
    }

    int set_spike_output_enabled(bool enabled) {
        model.set_spike_output_enabled(enabled);
        return 0;
    }
    bool is_spike_output_enabled() const { return model.spike_output_enabled(); }
    int set_device(const std::string& device) {
        model.set_device(device);
        return 0;
    }
    int set_dt(double dt) {
        model.set_dt(dt);
        return 0;
    }
    double get_dt() const { return model.dt(); }
    int set_num_threads(int num_threads) {
        model.set_num_threads(num_threads);
        return 0;
    }
    int get_num_threads() const { return model.num_threads(); }
    int load_mech_metadata(const std::string& path) {
        model.load_mech_metadata(path);
        return 0;
    }
    int ion_register(const std::string& ion, double charge) {
        return model.ion_register(ion, charge);
    }
    double ion_charge(const std::string& ion_mechanism) const {
        return model.ion_charge(ion_mechanism);
    }
    double get_global(const std::string& name) const {
        return model.global_scalar(name);
    }
    void set_global(const std::string& name, double value) {
        model.set_global_scalar(name, value);
    }
    std::vector<std::string> get_loaded_mech_metadata_paths() const {
        return model.loaded_mech_metadata_paths();
    }
    ArtificialCellView insert(const std::string& mech, ParamList params);
    Sim& build_morphology(nb::handle templates) {
        model.build_morphology(parse_morphology_templates(templates));
        return *this;
    }
    int build_microcircuit() { return model.build_microcircuit(); }
    int finitialize(double v_init) {
        model.finitialize(v_init);
        sample_recorders();
        return 0;
    }
    int run(double tstop) {
        return run_recorded(tstop);
    }
    int continue_run(double runtime) {
        return run_recorded(model.time() + runtime);
    }
    int fadvance() {
        model.fadvance();
        sample_recorders();
        return 0;
    }
    double get_t() const { return model.time(); }
    nb::dict debug_core_thread() const {
        const auto& core = model.core_neuron_data();
        const auto& nt = core.threads.front();
        nb::dict out;
        auto to_list = [](const auto& values) {
            nb::list list;
            for (const auto& value : values) {
                list.append(value);
            }
            return list;
        };
        out["ncell"] = nt.ncell;
        out["end"] = nt.end;
        out["node_permutation"] = to_list(nt.node_permutation);
        out["parent"] = to_list(nt.v_parent_index);
        out["a"] = to_list(nt.actual_a());
        out["b"] = to_list(nt.actual_b());
        out["rhs"] = to_list(nt.actual_rhs());
        out["d"] = to_list(nt.actual_d());
        out["v"] = to_list(nt.actual_v());
        out["area"] = to_list(nt.actual_area());
        out["t"] = nt._t;
        out["dt"] = nt._dt;
        nb::list memb_lists;
        for (const auto& ml : nt.memb_lists) {
            nb::dict item;
            item["name"] = ml.name;
            item["type"] = ml.type;
            item["nodeindices"] = to_list(ml.nodeindices);
            item["data"] = to_list(std::span<const double>(
                ml.ml.data,
                static_cast<std::size_t>(coreneuron::corenrn.get_prop_param_size()[static_cast<std::size_t>(ml.type)]) *
                    static_cast<std::size_t>(ml.ml._nodecount_padded)));
            item["pdata"] = to_list(ml.pdata);
            memb_lists.append(item);
        }
        out["memb_lists"] = memb_lists;
        nb::list tml;
        for (const auto& item : nt.tml_storage) {
            tml.append(coreneuron::nrn_get_mechname(item.index));
        }
        out["tml"] = tml;
        nb::list pnts;
        for (const auto& pnt : nt.pntproc_storage) {
            nb::dict item;
            item["type"] = pnt._type;
            item["i_instance"] = pnt._i_instance;
            item["tid"] = pnt._tid;
            pnts.append(item);
        }
        out["pntprocs"] = pnts;
        nb::list presyns;
        for (const auto& presyn : nt.presyn_storage) {
            nb::dict item;
            item["gid"] = presyn.gid_;
            item["threshold"] = presyn.threshold_;
            item["thvar_index"] = presyn.thvar_index_;
            item["nc_index"] = presyn.nc_index_;
            item["nc_cnt"] = presyn.nc_cnt_;
            presyns.append(item);
        }
        out["presyns"] = presyns;
        nb::list netcons;
        for (const auto& netcon : nt.netcon_storage) {
            nb::dict item;
            item["active"] = netcon.active_;
            item["delay"] = netcon.delay_;
            item["weight_index"] = netcon.u.weight_index_;
            item["target_type"] = netcon.target_ ? netcon.target_->_type : -1;
            item["target_instance"] = netcon.target_ ? netcon.target_->_i_instance : -1;
            netcons.append(item);
        }
        out["netcons"] = netcons;
        out["weights"] = to_list(nt.weight_storage);
        return out;
    }
    std::vector<double> get_spk_by_gid(int gid) const {
        std::vector<double> out;
        for (std::size_t i = 0; i < coreneuron::spikevec_gid.size(); ++i) {
            if (coreneuron::spikevec_gid[i] == gid) {
                out.push_back(coreneuron::spikevec_time[i]);
            }
        }
        return out;
    }
    void set_celsius(double celsius) { model.set_celsius(celsius); }
    double get_celsius() const { return model.celsius(); }
};

struct PopulationView;
struct CellView;
struct SectionGroupView;
struct SectionView;
struct VariableRefView;
struct NetworkView;
struct NetConView;

struct PopulationView {
    Sim* sim{};
    std::string name{};
    int gid_begin{};
    int gid_end{};

    std::size_t size() const { return static_cast<std::size_t>(gid_end - gid_begin); }
    CellView at(std::size_t index) const;
};

struct PopulationIter {
    PopulationView population{};
    std::size_t next{0};
    CellView next_item();
};

struct CellView {
    Sim* sim{};
    int gid{-1};

    void set_v_init(double value) const { sim->model.set_cell_v_init(gid, value); }
    double get_v_init() const { return sim->model.cell_v_init(gid); }
    SectionGroupView group(const std::string& label) const;
};

struct SectionGroupView {
    Sim* sim{};
    int gid{-1};
    std::string label{};
    bool all_sections{false};
    const std::vector<mind_micro_morph::section_id>* section_ids{};
    mutable bool section_cache_valid{false};
    mutable std::vector<std::size_t> section_cache{};

    std::size_t size() const {
        return all_sections ? static_cast<std::size_t>(sim->model.section_count(gid))
                            : section_ids->size();
    }
    std::size_t section_index_at(std::size_t index) const {
        return all_sections ? index : static_cast<std::size_t>((*section_ids)[index]);
    }
    const std::vector<std::size_t>& indices() const {
        if (section_cache_valid) {
            return section_cache;
        }
        const auto n = size();
        section_cache.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            section_cache[i] = section_index_at(i);
        }
        section_cache_valid = true;
        return section_cache;
    }
    SectionView at(std::size_t index) const;
    SegmentValueBatchView segment_values(const std::vector<int>& section_indices_arg,
                                         const std::vector<std::vector<double>>& values) const {
        std::vector<int> absolute_section_indices;
        absolute_section_indices.reserve(section_indices_arg.size());
        for (const int group_index : section_indices_arg) {
            const auto resolved = static_cast<std::size_t>(
                normalize_py_index(group_index, size(), "section_group.segment_values"));
            absolute_section_indices.push_back(static_cast<int>(section_index_at(resolved)));
        }
        return SegmentValueBatchView{make_segment_batch(absolute_section_indices, values)};
    }
    void insert(const std::string& mech, ParamList params) const {
        sim->model.insert_mechanism(gid, indices(), std::nullopt, mech, std::move(params));
    }
    void set_v_init(double value) const {
        sim->model.set_section_group_property(gid, indices(), ObjectOpKind::VInit, value);
    }
    double get_v_init() const {
        return sim->model.section_group_property(gid, indices(), ObjectOpKind::VInit);
    }
    void set_cm(double value) const {
        sim->model.set_section_group_property(gid, indices(), ObjectOpKind::Cm, value);
    }
    double get_cm() const {
        return sim->model.section_group_property(gid, indices(), ObjectOpKind::Cm);
    }
    void set_Ra(double value) const {
        sim->model.set_section_group_property(gid, indices(), ObjectOpKind::Ra, value);
    }
    double get_Ra() const {
        return sim->model.section_group_property(gid, indices(), ObjectOpKind::Ra);
    }
    void set_ion_range(const std::string& field, ParamValue value) const {
        sim->model.set_section_group_ion_range(gid, indices(), field, std::move(value));
    }
    double get_ion_range(const std::string& field) const {
        return sim->model.section_group_ion_range(gid, indices(), field);
    }
    int ion_style(const std::string& ion_mechanism) const {
        return sim->model.section_group_ion_style(gid, indices(), ion_mechanism);
    }
    int ion_style(const std::string& ion_mechanism,
                  int c_style,
                  int e_style,
                  int einit,
                  int eadvance,
                  int cinit) const {
        return sim->model.set_section_group_ion_style(
            gid,
            indices(),
            ion_mechanism,
            c_style,
            e_style,
            einit,
            eadvance,
            cinit);
    }
};

struct SectionGroupIter {
    SectionGroupView group{};
    std::size_t next{0};
    SectionView next_item();
};

struct SectionView {
    Sim* sim{};
    int gid{-1};
    int section_index{-1};
    double x{std::numeric_limits<double>::quiet_NaN()};

    bool has_loc() const { return std::isfinite(x); }
    SectionView loc(double loc_x) const {
        if (!std::isfinite(loc_x) || loc_x < 0.0 || loc_x > 1.0) {
            throw std::runtime_error("section location must be finite and in [0, 1]");
        }
        return SectionView{sim, gid, section_index, loc_x};
    }
    std::string label() const {
        return sim->model.section_label(gid, static_cast<std::size_t>(section_index));
    }
    nb::object insert(const std::string& mech, ParamList params) const;
    VariableRefView ref(const std::string& var, const std::string& mech = "global", int array_index = -1) const;
    VariableRefView ref_v() const;
};

struct VariableRefView {
    Sim* sim{};
    VariableRef ref{};

    double value() const {
        return sim->model.read_variable(ref);
    }
};

struct PointProcessView {
    Sim* sim{};
    int insert_id{-1};

    std::string mech() const { return sim->model.mechanism_mech(insert_id); }
    VariableRefView ref(const std::string& var, int array_index = -1) const {
        VariableRef out;
        out.kind = VariableRef::Kind::Mechanism;
        out.insert_id = insert_id;
        out.mech = mech();
        out.var = var;
        out.array_index = array_index;
        return VariableRefView{sim, std::move(out)};
    }
    double get_var(const std::string& key) const {
        return sim->model.mechanism_scalar(insert_id, key);
    }
    void set_var(const std::string& key, double value) const {
        sim->model.set_mechanism_scalar(insert_id, key, value);
    }
};

struct ArtificialCellView {
    Sim* sim{};
    int insert_id{-1};

    std::string mech() const { return sim->model.mechanism_mech(insert_id); }
    VariableRefView ref(const std::string& var, int array_index = -1) const {
        VariableRef out;
        out.kind = VariableRef::Kind::Mechanism;
        out.insert_id = insert_id;
        out.mech = mech();
        out.var = var;
        out.array_index = array_index;
        return VariableRefView{sim, std::move(out)};
    }
    double get_var(const std::string& key) const {
        return sim->model.mechanism_scalar(insert_id, key);
    }
    void set_var(const std::string& key, double value) const {
        sim->model.set_mechanism_scalar(insert_id, key, value);
    }
};

struct NetConWeightView {
    Sim* sim{};
    int connection_id{-1};
    std::size_t size() const { return sim->model.netcon_weight_count(connection_id); }
    double get(int index) const { return sim->model.netcon_weight(connection_id, index); }
    void set(int index, double value) const { sim->model.set_netcon_weight(connection_id, index, value); }
};

struct NetConView {
    Sim* sim{};
    int connection_id{-1};
    int id() const { return connection_id; }
    int runtime_index() const { return sim->model.netcon_runtime_index(connection_id); }
    int target_event_target_id() const { return sim->model.netcon_target_event_target_id(connection_id); }
    int source_event_target_id() const { return sim->model.netcon_source_event_target_id(connection_id); }
    NetConWeightView weight_view() const { return NetConWeightView{sim, connection_id}; }
    int wcnt() const { return static_cast<int>(sim->model.netcon_weight_count(connection_id)); }
    double get_delay() const { return sim->model.netcon_delay(connection_id); }
    void set_delay(double value) const { sim->model.set_netcon_delay(connection_id, value); }
    double get_threshold() const { return sim->model.netcon_threshold(connection_id); }
    void set_threshold(double value) const { sim->model.set_netcon_threshold(connection_id, value); }
};

struct SpikeInputView {
    Sim* sim{};
    int source_id{-1};

    int id() const { return source_id; }
    int runtime_index() const { return sim->model.spike_input_runtime_index(source_id); }
};

struct SpikeInputGroupView {
    Sim* sim{};
    std::vector<int> source_ids{};

    int size() const { return static_cast<int>(source_ids.size()); }
    SpikeInputView get(int index) const {
        return SpikeInputView{sim, source_ids[static_cast<std::size_t>(index)]};
    }
    int runtime_base() const {
        return sim->model.spike_input_runtime_index(source_ids.front());
    }
};

struct NetworkView {
    Sim* sim{};

    int register_gid_source(int gid, const VariableRefView& source, std::optional<double> threshold) const {
        return sim->model.register_gid_source(gid, source.ref, threshold);
    }
    NetConView gid_connect(int gid, const PointProcessView& post, double weight, double delay) const {
        const int id = sim->model.gid_connect(gid, post.insert_id, weight, delay);
        return NetConView{sim, id};
    }
    NetConView event_connect(const ArtificialCellView& source,
                             const PointProcessView& post,
                             double weight,
                             double delay) const {
        const int id = sim->model.event_target_connect(source.insert_id, post.insert_id, weight, delay);
        return NetConView{sim, id};
    }
    NetConView event_connect(const PointProcessView& source,
                             const PointProcessView& post,
                             double weight,
                             double delay) const {
        const int id = sim->model.event_target_connect(source.insert_id, post.insert_id, weight, delay);
        return NetConView{sim, id};
    }
    SpikeInputView spike_input() const {
        const int id = sim->model.register_spike_input_source();
        return SpikeInputView{sim, id};
    }
    SpikeInputGroupView spike_inputs(int count) const {
        std::vector<int> ids;
        ids.reserve(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i) {
            ids.push_back(sim->model.register_spike_input_source());
        }
        return SpikeInputGroupView{sim, std::move(ids)};
    }
    NetConView spike_connect(const SpikeInputView& source,
                             const PointProcessView& post,
                             double weight,
                             double delay) const {
        const int id = sim->model.spike_input_connect(source.source_id, post.insert_id, weight, delay);
        return NetConView{sim, id};
    }
};

struct TimeRefView {
    Sim* sim{};
};

struct VectorView {
    std::shared_ptr<RecorderBuffer> buffer{std::make_shared<RecorderBuffer>()};

    VectorView& record(const VariableRefView& ref) {
        buffer->sim = ref.sim;
        buffer->records_var = true;
        buffer->records_time = false;
        buffer->ref = ref.ref;
        buffer->samples.clear();
        if (buffer->sim != nullptr) {
            buffer->sim->attach_recorder(buffer);
        }
        return *this;
    }
    VectorView& record(const TimeRefView& ref) {
        buffer->sim = ref.sim;
        buffer->records_time = true;
        buffer->records_var = false;
        buffer->samples.clear();
        if (buffer->sim != nullptr) {
            buffer->sim->attach_recorder(buffer);
        }
        return *this;
    }
    std::vector<double> to_python() const {
        return buffer ? buffer->samples : std::vector<double>{};
    }
    std::size_t size() const { return to_python().size(); }
    double get(int index) const {
        auto values = to_python();
        return values.at(static_cast<std::size_t>(normalize_py_index(index, values.size(), "Vector")));
    }
};

inline CellView PopulationView::at(std::size_t index) const {
    if (index >= size()) {
        throw nb::index_error("population index out of range");
    }
    return CellView{sim, gid_begin + static_cast<int>(index)};
}

inline CellView PopulationIter::next_item() {
    if (next >= population.size()) {
        throw nb::stop_iteration();
    }
    return population.at(next++);
}

inline SectionGroupView CellView::group(const std::string& label) const {
    if (label == mind_micro_model::kAllSectionLabel) {
        return SectionGroupView{sim, gid, label, true, nullptr};
    }
    const auto& tpl = mind_micro_model::require_cell_template(sim->model.morph_layout(), gid);
    const auto it = tpl.label_index.find(label);
    if (it == tpl.label_index.end()) {
        throw std::runtime_error("unknown label '" + label + "' for gid=" + std::to_string(gid));
    }
    return SectionGroupView{sim, gid, label, false, &tpl.sections_by_label[it->second]};
}

inline SectionView SectionGroupView::at(std::size_t index) const {
    if (index >= size()) {
        throw nb::index_error("section_group index out of range");
    }
    return SectionView{sim, gid, static_cast<int>(section_index_at(index))};
}

inline SectionView SectionGroupIter::next_item() {
    if (next >= group.size()) {
        throw nb::stop_iteration();
    }
    return group.at(next++);
}

inline nb::object SectionView::insert(const std::string& mech, ParamList params) const {
    const int id = sim->model.insert_mechanism(
        gid,
        {static_cast<std::size_t>(section_index)},
        has_loc() ? std::optional<double>{x} : std::nullopt,
        mech,
        std::move(params));
    if (sim->model.mechanism_placement(id) == MechanismPlacementKind::Location) {
        return nb::cast(PointProcessView{sim, id});
    }
    return nb::none();
}

inline ArtificialCellView Sim::insert(const std::string& mech, ParamList params) {
    const int id = model.register_artificial_cell(mech, std::move(params));
    return ArtificialCellView{this, id};
}

inline VariableRefView SectionView::ref(const std::string& var, const std::string& mech, int array_index) const {
    if (!has_loc()) {
        throw std::runtime_error("variable reference requires a located section, e.g. sec(0.5)");
    }
    VariableRef out;
    out.kind = (mech == "global" && var == "v") ? VariableRef::Kind::LocationVoltage
                                                  : VariableRef::Kind::Location;
    out.gid = gid;
    out.section_index = section_index;
    out.x = x;
    out.mech = mech;
    out.var = var;
    out.array_index = array_index;
    return VariableRefView{sim, std::move(out)};
}

inline VariableRefView SectionView::ref_v() const {
    return ref("v", "global", -1);
}

inline PopulationView sim_population(Sim& sim, const std::string& name) {
    const auto& range = sim.model.population(name);
    return PopulationView{&sim, range.name, range.gid_begin, range.gid_end};
}

inline NetworkView sim_network(Sim& sim) {
    return NetworkView{&sim};
}

inline TimeRefView sim_time_ref(Sim& sim) {
    return TimeRefView{&sim};
}

inline std::vector<GidRange> make_gid_ranges(const std::vector<int>& begins,
                                             const std::vector<int>& ends) {
    if (begins.size() != ends.size()) {
        throw std::runtime_error("gid range begin/end vectors must have the same size");
    }
    std::vector<GidRange> ranges;
    ranges.reserve(begins.size());
    for (std::size_t i = 0; i < begins.size(); ++i) {
        ranges.push_back(GidRange{.begin = begins[i], .end = ends[i]});
    }
    return ranges;
}

inline int network_use_micro(Network& network, const std::string& name, Sim& micro) {
    if (!micro.model.core_initialized()) {
        throw std::runtime_error("Network.use_micro requires an initialized micro Sim");
    }
    return network.use_micro(name, micro.model.core_neuron_data_shared());
}

inline void network_use_neural_field(Network& network,
                                     const std::string& name,
                                     std::shared_ptr<NeuralFieldRule> rule,
                                     NodeToRoiMap node_map,
                                     LocalConnectivity local_connectivity,
                                     std::vector<double> state_soa,
                                     std::vector<double> params,
                                     std::vector<int> read_input_offsets,
                                     const std::vector<int>& reducer_state_indices,
                                     const std::vector<int>& reducer_exposure_indices) {
    if (reducer_state_indices.size() != reducer_exposure_indices.size()) {
        throw std::runtime_error("field reducer state/exposure vectors must have the same size");
    }
    std::vector<FieldExposureReducer> reducers;
    reducers.reserve(reducer_state_indices.size());
    for (std::size_t index = 0; index < reducer_state_indices.size(); ++index) {
        reducers.push_back(FieldExposureReducer{
            .state_index = reducer_state_indices[index],
            .exposure_index = reducer_exposure_indices[index],
        });
    }
    network.use_neural_field(std::move(name),
                             std::move(rule),
                             std::move(node_map),
                             std::move(local_connectivity),
                             std::move(state_soa),
                             std::move(params),
                             std::move(read_input_offsets),
                             std::move(reducers));
}

inline void network_bind_micro_roi(Network& network,
                                   int micro_circuit_index,
                                   const ROI& roi,
                                   const std::vector<int>& gid_range_begins,
                                   const std::vector<int>& gid_range_ends) {
    network.bind_micro_roi(micro_circuit_index,
                           roi,
                           make_gid_ranges(gid_range_begins, gid_range_ends));
}

inline void network_configure_micro_input_rule(
    Network& network,
    const ROI& roi,
    std::shared_ptr<mind_sim::bridge::sim::MicroInputRule> input_rule,
    std::vector<double> input_state,
    std::vector<double> input_params,
    const std::vector<std::shared_ptr<RandomStreamRule>>& random_rules,
    std::vector<std::vector<double>> random_states,
    std::vector<int> input_port_bases,
    std::vector<int> input_read_offsets) {
    if (random_rules.size() != random_states.size()) {
        throw std::runtime_error("random provider/state vectors must have the same size");
    }
    std::vector<RandomStreamBinding> random_streams;
    random_streams.reserve(random_rules.size());
    for (std::size_t index = 0; index < random_rules.size(); ++index) {
        random_streams.push_back(RandomStreamBinding{
            .rule = random_rules[index],
            .state = std::move(random_states[index]),
        });
    }
    network.configure_micro_input_rule(roi,
                                       std::move(input_rule),
                                       std::move(input_state),
                                       std::move(input_params),
                                       std::move(random_streams),
                                       std::move(input_port_bases),
                                       std::move(input_read_offsets));
}

inline std::string abi_kind_name(int kind) {
    if (kind == static_cast<int>(mind_sim::mind_mod::AbiRuleKind::Coupling)) {
        return "coupling";
    }
    if (kind == static_cast<int>(mind_sim::mind_mod::AbiRuleKind::MicroInput)) {
        return "micro_input";
    }
    if (kind == static_cast<int>(mind_sim::mind_mod::AbiRuleKind::MicroOutput)) {
        return "micro_output";
    }
    if (kind == static_cast<int>(mind_sim::mind_mod::AbiRuleKind::Region)) {
        return "region";
    }
    if (kind == static_cast<int>(mind_sim::mind_mod::AbiRuleKind::NeuralField)) {
        return "neural_field";
    }
    throw std::runtime_error("unknown compiled MindMod kind");
}

inline std::vector<std::string> abi_names(int count, const char* const* names) {
    std::vector<std::string> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        out.emplace_back(names[index]);
    }
    return out;
}

inline std::vector<double> abi_defaults(int count, const double* values) {
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        out.push_back(values[index]);
    }
    return out;
}

inline nb::dict mind_mod_descriptor_to_dict(const mind_sim::mind_mod::AbiRuleDescriptor& descriptor) {
    if (descriptor.abi_version != mind_sim::mind_mod::kMindModAbiVersion) {
        throw std::runtime_error("compiled MindMod ABI version mismatch");
    }
    nb::dict out;
    out["kind"] = abi_kind_name(descriptor.kind);
    out["name"] = std::string(descriptor.name);
    out["read"] = abi_names(descriptor.read_count, descriptor.read_names);
    out["write"] = abi_names(descriptor.write_count, descriptor.write_names);
    out["emit"] = abi_names(descriptor.emit_count, descriptor.emit_names);
    out["random"] = abi_names(descriptor.random_count, descriptor.random_names);
    out["param_names"] = abi_names(descriptor.param_count, descriptor.param_names);
    out["param_defaults"] = abi_defaults(descriptor.param_count, descriptor.param_defaults);
    out["state_names"] = abi_names(descriptor.state_count, descriptor.state_names);
    out["state_defaults"] = abi_defaults(descriptor.state_count, descriptor.state_defaults);
    out["local_state_names"] = abi_names(descriptor.local_state_count, descriptor.local_state_names);
    return out;
}

inline nb::dict inspect_mind_mod_library(const std::string& library_path) {
    auto library = mind_sim::utils::load_dynamic_library(library_path);
    const auto descriptor_fn =
        reinterpret_cast<mind_sim::mind_mod::DescriptorFn>(library->symbol("mind_rule_descriptor"));
    const auto* descriptor = descriptor_fn();
    if (!descriptor) {
        throw std::runtime_error("compiled MindMod returned a null descriptor");
    }
    return mind_mod_descriptor_to_dict(*descriptor);
}

void bind_rules(nb::module_& m);
void bind_io(nb::module_& m);
void bind_micro(nb::module_& m);
void bind_macro(nb::module_& m);

}  // namespace mind_sim::python_api::bindings
