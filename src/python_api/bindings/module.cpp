#include "micro/frontend/model.hpp"
#include "bridge/sim/interfaces.hpp"
#include "coreneuron/io/output_spikes.hpp"
#include "cosim/hybrid_simulator.hpp"
#include "macro/frontend/network.hpp"
#include "macro/sim/runtime.hpp"
#include "morph/section_distance.hpp"
#include "morph/section_spec.hpp"
#include "io/result_hdf5.hpp"
#include "macro/sim/rule_codegen.hpp"
#include "mind_mod/abi.hpp"
#include "mind_mod/rule_mod.hpp"
#include "utils/dynamic_library.hpp"

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

namespace {

using mind_sim::micro::frontend::MicroFrontendModel;
using mind_sim::micro::frontend::MechanismPlacementKind;
using mind_sim::micro::frontend::MorphologyTemplateSpec;
using mind_sim::micro::frontend::VariableRef;
using mind_sim::macro::frontend::Connectivity;
using mind_sim::macro::frontend::GidRange;
using mind_sim::macro::frontend::Network;
using mind_sim::macro::frontend::ROI;
using mind_sim::macro::sim::CouplingRule;
using mind_sim::macro::sim::MacroRuntime;
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
int normalize_py_index(int index, SizeT size, const char* what) {
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

std::vector<SectionPt3d> pt3d_from_obj(nb::handle obj) {
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

nb::list pt3d_to_list(const std::vector<SectionPt3d>& pts) {
    nb::list out;
    for (const auto& p : pts) {
        out.append(nb::make_tuple(p.x_um, p.y_um, p.z_um, p.diam_um));
    }
    return out;
}

std::vector<SectionSpec> parse_sections(nb::handle obj) {
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

SectionList make_section_list(std::vector<SectionSpec> sections) {
    SectionList out;
    out.sections = std::make_shared<std::vector<SectionSpec>>(std::move(sections));
    return out;
}

std::string section_name_from_arg(nb::handle section) {
    if (nb::isinstance<SectionSpec>(section)) {
        return nb::cast<const SectionSpec&>(section).name;
    }
    return nb::cast<std::string>(section);
}

mind_micro_biophysical::SegmentParamBatch make_segment_batch(
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

ParamValue parse_param_value(nb::handle value) {
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

ParamList parse_kwargs(const nb::kwargs& kwargs) {
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

std::vector<MorphologyTemplateSpec> parse_morphology_templates(nb::handle obj) {
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

CellView PopulationView::at(std::size_t index) const {
    if (index >= size()) {
        throw nb::index_error("population index out of range");
    }
    return CellView{sim, gid_begin + static_cast<int>(index)};
}

CellView PopulationIter::next_item() {
    if (next >= population.size()) {
        throw nb::stop_iteration();
    }
    return population.at(next++);
}

SectionGroupView CellView::group(const std::string& label) const {
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

SectionView SectionGroupView::at(std::size_t index) const {
    if (index >= size()) {
        throw nb::index_error("section_group index out of range");
    }
    return SectionView{sim, gid, static_cast<int>(section_index_at(index))};
}

SectionView SectionGroupIter::next_item() {
    if (next >= group.size()) {
        throw nb::stop_iteration();
    }
    return group.at(next++);
}

nb::object SectionView::insert(const std::string& mech, ParamList params) const {
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

ArtificialCellView Sim::insert(const std::string& mech, ParamList params) {
    const int id = model.register_artificial_cell(mech, std::move(params));
    return ArtificialCellView{this, id};
}

VariableRefView SectionView::ref(const std::string& var, const std::string& mech, int array_index) const {
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

VariableRefView SectionView::ref_v() const {
    return ref("v", "global", -1);
}

PopulationView sim_population(Sim& sim, const std::string& name) {
    const auto& range = sim.model.population(name);
    return PopulationView{&sim, range.name, range.gid_begin, range.gid_end};
}

NetworkView sim_network(Sim& sim) {
    return NetworkView{&sim};
}

TimeRefView sim_time_ref(Sim& sim) {
    return TimeRefView{&sim};
}

std::vector<GidRange> make_gid_ranges(const std::vector<int>& begins,
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

int network_use_micro(Network& network, const std::string& name, Sim& micro) {
    if (!micro.model.core_initialized()) {
        throw std::runtime_error("Network.use_micro requires an initialized micro Sim");
    }
    return network.use_micro(name, micro.model.core_neuron_data_shared());
}

void network_bind_micro_roi(Network& network,
                            int micro_circuit_index,
                            const ROI& roi,
                            const std::vector<int>& gid_range_begins,
                            const std::vector<int>& gid_range_ends) {
    network.bind_micro_roi(micro_circuit_index,
                           roi,
                           make_gid_ranges(gid_range_begins, gid_range_ends));
}

void network_configure_micro_input_rule(
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

std::string abi_kind_name(int kind) {
    if (kind == static_cast<int>(mind_sim::mind_mod::AbiRuleKind::Coupling)) {
        return "coupling";
    }
    if (kind == static_cast<int>(mind_sim::mind_mod::AbiRuleKind::MicroInput)) {
        return "micro_input";
    }
    if (kind == static_cast<int>(mind_sim::mind_mod::AbiRuleKind::MicroOutput)) {
        return "micro_output";
    }
    throw std::runtime_error("unknown compiled MindMod kind");
}

std::vector<std::string> abi_names(int count, const char* const* names) {
    std::vector<std::string> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        out.emplace_back(names[index]);
    }
    return out;
}

std::vector<double> abi_defaults(int count, const double* values) {
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        out.push_back(values[index]);
    }
    return out;
}

nb::dict mind_mod_descriptor_to_dict(const mind_sim::mind_mod::AbiRuleDescriptor& descriptor) {
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
    return out;
}

nb::dict inspect_mind_mod_library(const std::string& library_path) {
    auto library = mind_sim::utils::load_dynamic_library(library_path);
    const auto descriptor_fn =
        reinterpret_cast<mind_sim::mind_mod::DescriptorFn>(library->symbol("mind_rule_descriptor"));
    const auto* descriptor = descriptor_fn();
    if (!descriptor) {
        throw std::runtime_error("compiled MindMod returned a null descriptor");
    }
    return mind_mod_descriptor_to_dict(*descriptor);
}

nb::dict region_rule_fields_to_dict(const mind_sim::macro::sim::codegen::RegionRuleFields& fields) {
    nb::dict out;
    out["inputs"] = fields.inputs;
    out["exposures"] = fields.exposures;
    return out;
}

}  // namespace

NB_MODULE(_native, m) {
    m.doc() = "MIND_Sim native C++ frontend and simulator bindings";

    m.def("_codegen_kernel_name",
          &mind_sim::macro::sim::codegen::kernel_name,
          nb::arg("name"),
          nb::arg("what"));
    m.def("_codegen_region_rule_source",
          &mind_sim::macro::sim::codegen::region_rule_source,
          nb::arg("inputs"),
          nb::arg("exposures"),
          nb::arg("states"),
          nb::arg("params"),
          nb::arg("update"));
    m.def("_inspect_region_rule_fields",
          [](const std::vector<std::string>& states,
             const std::vector<std::string>& params,
             const std::string& update) {
              return region_rule_fields_to_dict(
                  mind_sim::macro::sim::codegen::region_rule_fields(states, params, update));
          },
          nb::arg("states"),
          nb::arg("params"),
          nb::arg("update"));
    m.def("_translate_mind_mod_to_cpp",
          &mind_sim::mind_mod::compiled_rule_source,
          nb::arg("source"),
          nb::arg("origin"));
    m.def("_inspect_mind_mod_library",
          &inspect_mind_mod_library,
          nb::arg("library_path"));

    nb::class_<ScalarBuffer>(m, "ScalarBuffer")
        .def(nb::init<>())
        .def(nb::init<std::size_t>(), nb::arg("exposure_count"))
        .def_rw("values", &ScalarBuffer::values)
        .def("get",
             [](const ScalarBuffer& buffer, int exposure_id) {
                 return buffer.get(exposure_id, "ScalarBuffer.get");
             },
             nb::arg("exposure_id"))
        .def("set",
             [](ScalarBuffer& buffer, int exposure_id, double value) {
                 buffer.at(exposure_id, "ScalarBuffer.set") = value;
             },
             nb::arg("exposure_id"),
             nb::arg("value"))
        .def("__len__", &ScalarBuffer::size)
        .def("__repr__",
             [](const ScalarBuffer& buffer) {
                 return "<mind_sim.ScalarBuffer exposures=" + std::to_string(buffer.size()) + ">";
             });

    nb::class_<mind_sim::micro::sim::MicroSpikeTable>(m, "MicroSpikeTable")
        .def(nb::init<>())
        .def_rw("time", &mind_sim::micro::sim::MicroSpikeTable::time)
        .def_rw("gid", &mind_sim::micro::sim::MicroSpikeTable::gid)
        .def("__len__", &mind_sim::micro::sim::MicroSpikeTable::size)
        .def("save_h5",
             &mind_sim::io::save_micro_spikes_h5,
             nb::arg("path"),
             nb::arg("timing_s") = std::vector<double>{},
             nb::arg("metadata") = std::vector<double>{})
        .def("__repr__",
             [](const mind_sim::micro::sim::MicroSpikeTable& spikes) {
                 return "<mind_sim.MicroSpikeTable spikes=" + std::to_string(spikes.size()) + ">";
             });

    nb::class_<ROI>(m, "ROI")
        .def_ro("index", &ROI::index)
        .def_ro("label", &ROI::label)
        .def("__repr__",
             [](const ROI& roi) {
                 return "<mind_sim.ROI index=" + std::to_string(roi.index) +
                        " label='" + roi.label + "'>";
             });

    nb::class_<RegionRule>(m, "_RegionRule")
        .def_prop_ro("name", &RegionRule::name)
        .def_prop_ro("input_count", &RegionRule::input_count)
        .def_prop_ro("exposure_count", &RegionRule::exposure_count)
        .def_prop_ro("state_count", &RegionRule::state_count)
        .def_prop_ro("param_count", &RegionRule::param_count)
        .def_prop_ro("library_path", &RegionRule::library_path);

    m.def("_load_region_rule",
          [](const std::string& name,
             const std::string& library_path,
             int input_count,
             int exposure_count,
             int state_count,
             int param_count) {
              return std::make_shared<RegionRule>(
                  name, library_path, input_count, exposure_count, state_count, param_count);
          },
          nb::arg("name"),
          nb::arg("library_path"),
          nb::arg("input_count"),
          nb::arg("exposure_count"),
          nb::arg("state_count"),
          nb::arg("param_count"));

    nb::class_<CouplingRule>(m, "_CouplingRule")
        .def_prop_ro("name", &CouplingRule::name)
        .def_prop_ro("input_count", &CouplingRule::input_count)
        .def_prop_ro("exposure_count", &CouplingRule::exposure_count)
        .def_prop_ro("param_count", &CouplingRule::param_count)
        .def_prop_ro("library_path", &CouplingRule::library_path);

    m.def("_load_compiled_coupling_rule",
          [](const std::string& library_path) {
              return std::make_shared<CouplingRule>(library_path);
          },
          nb::arg("library_path"));

    nb::class_<mind_sim::bridge::sim::MicroInputRule>(m, "_MicroInputRule")
        .def_prop_ro("name", &mind_sim::bridge::sim::MicroInputRule::name)
        .def_prop_ro("input_count", &mind_sim::bridge::sim::MicroInputRule::input_count)
        .def_prop_ro("state_count", &mind_sim::bridge::sim::MicroInputRule::state_count)
        .def_prop_ro("param_count", &mind_sim::bridge::sim::MicroInputRule::param_count)
        .def_prop_ro("input_port_count", &mind_sim::bridge::sim::MicroInputRule::input_port_count)
        .def_prop_ro("random_count", &mind_sim::bridge::sim::MicroInputRule::random_count)
        .def_prop_ro("library_path", &mind_sim::bridge::sim::MicroInputRule::library_path);

    m.def("_load_compiled_micro_input_rule",
          [](const std::string& library_path) {
              return std::make_shared<mind_sim::bridge::sim::MicroInputRule>(library_path);
          },
          nb::arg("library_path"));

    nb::class_<RandomStreamRule>(m, "_RandomStreamRule")
        .def_prop_ro("state_count", &RandomStreamRule::state_count)
        .def_prop_ro("library_path", &RandomStreamRule::library_path);

    m.def("_load_random_stream_rule",
          [](const std::string& library_path, int state_count) {
              return std::make_shared<RandomStreamRule>(library_path, state_count);
          },
          nb::arg("library_path"),
          nb::arg("state_count"));

    nb::class_<mind_sim::bridge::sim::MicroOutputRule>(m, "_MicroOutputRule")
        .def_prop_ro("name", &mind_sim::bridge::sim::MicroOutputRule::name)
        .def_prop_ro("exposure_count", &mind_sim::bridge::sim::MicroOutputRule::exposure_count)
        .def_prop_ro("state_count", &mind_sim::bridge::sim::MicroOutputRule::state_count)
        .def_prop_ro("param_count", &mind_sim::bridge::sim::MicroOutputRule::param_count)
        .def_prop_ro("library_path", &mind_sim::bridge::sim::MicroOutputRule::library_path);

    m.def("_load_compiled_micro_output_rule",
          [](const std::string& library_path) {
              return std::make_shared<mind_sim::bridge::sim::MicroOutputRule>(library_path);
          },
          nb::arg("library_path"));

    nb::class_<Connectivity>(m, "Connectivity")
        .def(nb::init<std::vector<std::string>,
                      std::vector<std::vector<double>>,
                      std::vector<std::vector<double>>>(),
             nb::arg("labels"),
             nb::arg("weights"),
             nb::arg("delays"))
        .def("rois", &Connectivity::rois)
        .def("roi_count", &Connectivity::roi_count)
        .def("roi_index", &Connectivity::roi_index, nb::arg("label"))
        .def("weight_at", &Connectivity::weight_at, nb::arg("target_roi"), nb::arg("source_roi"))
        .def("delay_at", &Connectivity::delay_at, nb::arg("target_roi"), nb::arg("source_roi"))
        .def("min_positive_delay", &Connectivity::min_positive_delay);

    nb::class_<Network>(m, "_Network")
        .def(nb::init<std::vector<std::string>,
                      std::vector<std::vector<double>>,
                      std::vector<std::vector<double>>,
                      std::vector<std::string>,
                      std::vector<std::string>,
                      std::vector<int>>(),
             nb::arg("roi_labels"),
             nb::arg("weights"),
             nb::arg("delays"),
             nb::arg("inputs"),
             nb::arg("exposures"),
             nb::arg("recorded_rois"))
        .def(nb::init<Connectivity, std::vector<std::string>, std::vector<std::string>, std::vector<int>>(),
             nb::arg("connectivity"),
             nb::arg("inputs"),
             nb::arg("exposures"),
             nb::arg("recorded_rois"))
        .def("roi",
             nb::overload_cast<int>(&Network::roi, nb::const_),
             nb::arg("index"))
        .def("roi",
             nb::overload_cast<const std::string&>(&Network::roi, nb::const_),
             nb::arg("label"))
        .def("rois", &Network::rois)
        .def("inputs", &Network::inputs)
        .def("exposures", &Network::exposures)
        .def("recorded_rois", &Network::recorded_rois)
        .def("set_recorded_rois", &Network::set_recorded_rois, nb::arg("recorded_rois"))
        .def("input_index", &Network::input_index, nb::arg("input"))
        .def("input_count", &Network::input_count)
        .def("exposure_index", &Network::exposure_index, nb::arg("exposure"))
        .def("exposure_count", &Network::exposure_count)
        .def("set_initial_exposure",
             &Network::set_initial_exposure,
             nb::arg("roi"),
             nb::arg("exposure"))
        .def("set_initial_exposure_value",
             &Network::set_initial_exposure_value,
             nb::arg("roi"),
             nb::arg("exposure"),
             nb::arg("value"))
        .def("set_dc_input",
             &Network::set_dc_input,
             nb::arg("roi"),
             nb::arg("input"))
        .def("set_dc_input_value",
             &Network::set_dc_input_value,
             nb::arg("roi"),
             nb::arg("input"),
             nb::arg("value"))
        .def("couple",
             &Network::couple,
             nb::arg("source_roi"),
             nb::arg("target_roi"),
             nb::arg("rule"),
             nb::arg("params"),
             nb::arg("read_exposure_offsets"),
             nb::arg("write_input_offsets"))
        .def("use_region_rule",
             &Network::use_region_rule,
             nb::arg("roi"),
             nb::arg("rule"),
             nb::arg("state"),
             nb::arg("params"))
        .def("use_micro",
             &network_use_micro,
             nb::arg("name"),
             nb::arg("micro"))
        .def("bind_micro_roi",
             &network_bind_micro_roi,
             nb::arg("micro_circuit_index"),
             nb::arg("roi"),
             nb::arg("gid_range_begins"),
             nb::arg("gid_range_ends"))
        .def("configure_micro_input_rule",
             &network_configure_micro_input_rule,
             nb::arg("roi"),
             nb::arg("input_rule"),
             nb::arg("input_state"),
             nb::arg("input_params"),
             nb::arg("random_rules"),
             nb::arg("random_states"),
             nb::arg("input_port_bases"),
             nb::arg("input_read_offsets"))
        .def("configure_micro_output_rule",
             &Network::configure_micro_output_rule,
             nb::arg("roi"),
             nb::arg("output_rule"),
             nb::arg("output_state"),
             nb::arg("output_params"),
             nb::arg("output_write_offsets"))
        .def("roi_index", &Network::roi_index, nb::arg("label"))
        .def("roi_count", &Network::roi_count)
        .def("weight_at", &Network::weight_at, nb::arg("target_roi"), nb::arg("source_roi"))
        .def("delay_at", &Network::delay_at, nb::arg("target_roi"), nb::arg("source_roi"))
        .def("min_positive_delay", &Network::min_positive_delay);

    nb::class_<mind_sim::macro::sim::ExposureRecord>(m, "ExposureRecord")
        .def_ro("roi_count", &mind_sim::macro::sim::ExposureRecord::roi_count)
        .def_ro("exposure_count", &mind_sim::macro::sim::ExposureRecord::exposure_count)
        .def_ro("roi_indices", &mind_sim::macro::sim::ExposureRecord::roi_indices)
        .def_ro("values", &mind_sim::macro::sim::ExposureRecord::values)
        .def_prop_ro("recorded_roi_count",
                     &mind_sim::macro::sim::ExposureRecord::recorded_roi_count)
        .def_prop_ro("sample_count", &mind_sim::macro::sim::ExposureRecord::sample_count);

    nb::class_<mind_sim::macro::sim::MacroSimulationResult>(m, "MacroSimulationResult")
        .def_ro("times", &mind_sim::macro::sim::MacroSimulationResult::times)
        .def_ro("exposures", &mind_sim::macro::sim::MacroSimulationResult::exposures)
        .def("save_h5",
             &mind_sim::io::save_macro_result_h5,
             nb::arg("path"),
             nb::arg("exposure_names"),
             nb::arg("roi_labels"),
             nb::arg("timing_s") = std::vector<double>{},
             nb::arg("metadata") = std::vector<double>{});

    nb::class_<MacroRuntime>(m, "MacroRuntime")
        .def(nb::init<Network>(), nb::arg("network"))
        .def("run",
             &MacroRuntime::run,
             nb::arg("t_stop"),
             nb::arg("dt_macro"),
             nb::call_guard<nb::gil_scoped_release>());

    nb::class_<mind_sim::cosim::SimulationResult>(m, "SimulationResult")
        .def_ro("times", &mind_sim::cosim::SimulationResult::times)
        .def_ro("exposures", &mind_sim::cosim::SimulationResult::exposures)
        .def_ro("micro_spikes_by_roi", &mind_sim::cosim::SimulationResult::micro_spikes_by_roi)
        .def("save_h5",
             &mind_sim::io::save_cosim_result_h5,
             nb::arg("path"),
             nb::arg("exposure_names"),
             nb::arg("roi_labels"),
             nb::arg("spike_roi"),
             nb::arg("timing_s") = std::vector<double>{},
             nb::arg("metadata") = std::vector<double>{});

    nb::class_<mind_sim::cosim::Simulator>(m, "Simulator")
        .def(nb::init<Network, double, double, double, bool>(),
             nb::arg("network"),
             nb::arg("dt_micro"),
             nb::arg("dt_macro"),
             nb::arg("batch_window"),
             nb::arg("record_micro_spikes"))
        .def("run",
             &mind_sim::cosim::Simulator::run,
             nb::arg("t_stop"),
             nb::call_guard<nb::gil_scoped_release>());

    nb::class_<SectionSpec>(m, "section")
        .def(nb::init<const std::string&, const std::string&>(), nb::arg("name"), nb::arg("label"))
        .def_rw("name", &SectionSpec::name)
        .def(
            "connect",
            [](SectionSpec& child, nb::handle parent, double parentx) {
                if (parentx < 0.0 || parentx > 1.0) {
                    throw std::runtime_error("section.connect parentx must be in [0, 1]");
                }
                child.parent_name = parent.is_none() ? std::string{} : section_name_from_arg(parent);
                child.parentx = parentx;
            },
            nb::arg("parent").none(),
            nb::arg("parentx") = 1.0)
        .def_prop_ro("parent",
                     [](const SectionSpec& sec) -> nb::object {
                         return sec.parent_name.empty() ? nb::none() : nb::str(sec.parent_name.c_str());
                     })
        .def_prop_ro("parentx", [](const SectionSpec& sec) { return sec.parentx; })
        .def_prop_ro("label", [](const SectionSpec& sec) { return sec.label; })
        .def_rw("nseg", &SectionSpec::nseg)
        .def_rw("L_um", &SectionSpec::L_um)
        .def_rw("diam_um", &SectionSpec::diam_um)
        .def_prop_rw("pt3d",
                     [](const SectionSpec& sec) { return pt3d_to_list(sec.pt3d); },
                     [](SectionSpec& sec, nb::handle value) { sec.pt3d = pt3d_from_obj(value); })
        .def("__repr__", [](const SectionSpec& sec) { return "<mind_sim.section name='" + sec.name + "'>"; });

    nb::class_<SectionListIter>(m, "_section_list_iterator")
        .def("__iter__", [](SectionListIter& it) -> SectionListIter& { return it; })
        .def("__next__", &SectionListIter::next_item, nb::rv_policy::reference_internal);

    nb::class_<SectionList>(m, "section_list")
        .def(nb::init<>())
        .def("__len__", [](const SectionList& list) { return list.sections->size(); })
        .def("__getitem__",
             [](SectionList& list, int index) -> SectionSpec& {
                 return list.sections->at(
                     static_cast<std::size_t>(normalize_py_index(index, list.sections->size(), "section_list")));
             },
             nb::rv_policy::reference_internal)
        .def("__iter__",
             [](const SectionList& list) {
                 return SectionListIter{list.sections, 0};
             })
        .def("append", [](SectionList& list, const SectionSpec& sec) { list.sections->push_back(sec); })
        .def("extend",
             [](SectionList& list, nb::handle items) {
                 auto parsed = parse_sections(items);
                 list.sections->insert(list.sections->end(), parsed.begin(), parsed.end());
             })
        .def("delete_subtree",
             [](SectionList& list, nb::handle sec) -> SectionList& {
                 *list.sections = mind_micro_frontend::delete_subtree_specs(*list.sections, section_name_from_arg(sec));
                 return list;
             },
             nb::rv_policy::reference_internal)
        .def("delete_label",
             [](SectionList& list, const std::string& label) -> SectionList& {
                 *list.sections = mind_micro_frontend::delete_label_specs(*list.sections, label);
                 return list;
             },
             nb::rv_policy::reference_internal)
        .def("to_list",
             [](const SectionList& list) {
                 nb::list out;
                 for (const auto& sec : *list.sections) {
                     out.append(sec);
                 }
                 return out;
             })
        .def("__repr__",
             [](const SectionList& list) {
                 return "<mind_sim.section_list size=" + std::to_string(list.sections->size()) + ">";
             });

    nb::class_<mind_micro_morph::SectionDistanceLayout>(m, "section_distance_layout")
        .def("distance",
             [](const mind_micro_morph::SectionDistanceLayout& layout,
                const std::string& a_sec,
                double a_x,
                const std::string& b_sec,
                double b_x) {
                 return mind_micro_morph::distance_between_locs_um(layout, a_sec, a_x, b_sec, b_x);
             })
        .def("diam",
             [](const mind_micro_morph::SectionDistanceLayout& layout, const std::string& sec, double x) {
                 mind_micro_morph::SectionLocationResolveInfo info;
                 if (!mind_micro_morph::resolve_section_location(layout, sec, x, &info)) {
                     throw std::runtime_error("section location could not be resolved");
                 }
                 return info.diam_um;
             });

    m.def("load_swc_sections", &mind_micro_frontend::load_swc_sections, nb::arg("swc_file"));
    m.def("load_asc_sections", &mind_micro_frontend::load_asc_sections, nb::arg("asc_file"));
    m.def("load_swc_section_list",
          [](const std::string& swc_file) {
              return make_section_list(mind_micro_frontend::load_swc_sections(swc_file));
          },
          nb::arg("swc_file"));
    m.def("load_asc_section_list",
          [](const std::string& asc_file) {
              return make_section_list(mind_micro_frontend::load_asc_sections(asc_file));
          },
          nb::arg("asc_file"));
    m.def("build_section_distance_layout",
          [](nb::handle sections) {
              mind_micro_frontend::SectionNseg nseg;
              auto morph = mind_micro_frontend::build_morph_from_sections(parse_sections(sections), nseg);
              mind_micro_frontend::apply_section_nseg(morph, nseg);
              return mind_micro_morph::build_section_distance_layout(morph);
          },
          nb::arg("sections"));

    nb::class_<PopulationIter>(m, "_population_iterator")
        .def("__iter__", [](PopulationIter& it) -> PopulationIter& { return it; })
        .def("__next__", &PopulationIter::next_item);

    nb::class_<PopulationView>(m, "population")
        .def_ro("name", &PopulationView::name)
        .def_ro("gid_begin", &PopulationView::gid_begin)
        .def_ro("gid_end", &PopulationView::gid_end)
        .def("__len__", &PopulationView::size)
        .def("__getitem__",
             [](const PopulationView& pop, int index) {
                 return pop.at(static_cast<std::size_t>(normalize_py_index(index, pop.size(), "population")));
             })
        .def("__iter__", [](const PopulationView& pop) { return PopulationIter{pop, 0}; })
        .def("__repr__", [](const PopulationView& pop) { return "<mind_sim.population name='" + pop.name + "'>"; });

    nb::class_<CellView>(m, "cell")
        .def_ro("gid", &CellView::gid)
        .def_prop_rw("v_init", &CellView::get_v_init, &CellView::set_v_init)
        .def("group", &CellView::group, nb::arg("label"))
        .def("__repr__", [](const CellView& cell) { return "<mind_sim.cell gid=" + std::to_string(cell.gid) + ">"; });

    nb::class_<SectionGroupIter>(m, "_section_group_iterator")
        .def("__iter__", [](SectionGroupIter& it) -> SectionGroupIter& { return it; })
        .def("__next__", &SectionGroupIter::next_item);

    nb::class_<SegmentValueBatchView>(m, "segment_values")
        .def("__len__", &SegmentValueBatchView::size)
        .def("__repr__",
             [](const SegmentValueBatchView& batch) {
                 return "<mind_sim.segment_values size=" + std::to_string(batch.size()) + ">";
             });

    nb::class_<SectionGroupView>(m, "section_group")
        .def("__len__", &SectionGroupView::size)
        .def("__getitem__",
             [](const SectionGroupView& group, int index) {
                 return group.at(static_cast<std::size_t>(normalize_py_index(index, group.size(), "section_group")));
             })
        .def("__iter__", [](const SectionGroupView& group) { return SectionGroupIter{group, 0}; })
        .def("segment_values", &SectionGroupView::segment_values, nb::arg("section_indices"), nb::arg("values"))
        .def("insert",
             [](const SectionGroupView& group, const std::string& mech, const nb::kwargs& kwargs) {
                 group.insert(mech, parse_kwargs(kwargs));
             })
        .def("ion_style",
             nb::overload_cast<const std::string&>(&SectionGroupView::ion_style, nb::const_),
             nb::arg("ion_mechanism"))
        .def("ion_style",
             nb::overload_cast<const std::string&, int, int, int, int, int>(
                 &SectionGroupView::ion_style,
                 nb::const_),
             nb::arg("ion_mechanism"),
             nb::arg("c_style"),
             nb::arg("e_style"),
             nb::arg("einit"),
             nb::arg("eadvance"),
             nb::arg("cinit"))
        .def("__getattr__",
             [](const SectionGroupView& group, const std::string& key) {
                 if (key == "v_init") {
                     return group.get_v_init();
                 }
                 if (key == "cm") {
                     return group.get_cm();
                 }
                 if (key == "Ra") {
                     return group.get_Ra();
                 }
                 return group.get_ion_range(key);
             })
        .def("__setattr__",
             [](SectionGroupView& group, const std::string& key, nb::handle value) {
                 if (key == "v_init") {
                     group.set_v_init(nb::cast<double>(value));
                     return;
                 }
                 if (key == "cm") {
                     group.set_cm(nb::cast<double>(value));
                     return;
                 }
                 if (key == "Ra") {
                     group.set_Ra(nb::cast<double>(value));
                     return;
                 }
                 group.set_ion_range(key, parse_param_value(value));
             })
        .def_prop_rw("v_init", &SectionGroupView::get_v_init, &SectionGroupView::set_v_init)
        .def_prop_rw("cm", &SectionGroupView::get_cm, &SectionGroupView::set_cm)
        .def_prop_rw("Ra", &SectionGroupView::get_Ra, &SectionGroupView::set_Ra)
        .def("__repr__",
             [](const SectionGroupView& group) {
                 return "<mind_sim.section_group label='" + group.label + "' size=" +
                        std::to_string(group.size()) + ">";
             });

    nb::class_<SectionView>(m, "section_view")
        .def_prop_ro("gid", [](const SectionView& sec) { return sec.gid; })
        .def_prop_ro("section_index", [](const SectionView& sec) { return sec.section_index; })
        .def_prop_ro("label", &SectionView::label)
        .def("loc", &SectionView::loc, nb::arg("loc"))
        .def("__call__", &SectionView::loc, nb::arg("loc"))
        .def("insert",
             [](const SectionView& sec, const std::string& mech, const nb::kwargs& kwargs) {
                 return sec.insert(mech, parse_kwargs(kwargs));
             })
        .def("ref",
             &SectionView::ref,
             nb::arg("var"),
             nb::arg("mech") = "global",
             nb::arg("array_index") = -1)
        .def_prop_ro("_ref_v", &SectionView::ref_v)
        .def("__repr__",
             [](const SectionView& sec) {
                 return "<mind_sim.section_view gid=" + std::to_string(sec.gid) +
                        " section_index=" + std::to_string(sec.section_index) + ">";
             });

    nb::class_<VariableRefView>(m, "variable_ref")
        .def("value", &VariableRefView::value)
        .def("__repr__", [](const VariableRefView&) { return "<mind_sim.variable_ref>"; });

    nb::class_<PointProcessView>(m, "point_process")
        .def_ro("insert_id", &PointProcessView::insert_id)
        .def("ref", &PointProcessView::ref, nb::arg("var"), nb::arg("array_index") = -1)
        .def("__getattr__",
             [](const PointProcessView& mechanism, const std::string& key) {
                 return mechanism.get_var(key);
             })
        .def("__setattr__",
             [](PointProcessView& mechanism, const std::string& key, nb::handle value) {
                 mechanism.set_var(key, nb::cast<double>(value));
             })
        .def("__repr__",
             [](const PointProcessView& mechanism) {
                 return "<mind_sim.point_process insert_id=" + std::to_string(mechanism.insert_id) + ">";
             });

    nb::class_<ArtificialCellView>(m, "artificial_cell")
        .def_ro("insert_id", &ArtificialCellView::insert_id)
        .def("ref", &ArtificialCellView::ref, nb::arg("var"), nb::arg("array_index") = -1)
        .def("__getattr__",
             [](const ArtificialCellView& mechanism, const std::string& key) {
                 return mechanism.get_var(key);
             })
        .def("__setattr__",
             [](ArtificialCellView& mechanism, const std::string& key, nb::handle value) {
                 mechanism.set_var(key, nb::cast<double>(value));
             })
        .def("__repr__",
             [](const ArtificialCellView& mechanism) {
                 return "<mind_sim.artificial_cell insert_id=" + std::to_string(mechanism.insert_id) + ">";
             });

    nb::class_<NetConWeightView>(m, "netcon_weight")
        .def("__len__", &NetConWeightView::size)
        .def("__getitem__",
             [](const NetConWeightView& w, int index) {
                 return w.get(normalize_py_index(index, w.size(), "netcon_weight"));
             })
        .def("__setitem__",
             [](NetConWeightView& w, int index, double value) {
                 w.set(normalize_py_index(index, w.size(), "netcon_weight"), value);
             });

    nb::class_<NetConView>(m, "netcon")
        .def_prop_ro("id", &NetConView::id)
        .def_prop_ro("runtime_index", &NetConView::runtime_index)
        .def_prop_ro("target_event_target_id", &NetConView::target_event_target_id)
        .def_prop_ro("source_event_target_id", &NetConView::source_event_target_id)
        .def_prop_ro("weight", &NetConView::weight_view)
        .def("wcnt", &NetConView::wcnt)
        .def_prop_rw("delay", &NetConView::get_delay, &NetConView::set_delay)
        .def_prop_rw("threshold", &NetConView::get_threshold, &NetConView::set_threshold)
        .def("__repr__",
             [](const NetConView& nc) {
                 return "<mind_sim.netcon id=" + std::to_string(nc.connection_id) + ">";
             });

    nb::class_<SpikeInputView>(m, "spike_input")
        .def_prop_ro("id", &SpikeInputView::id)
        .def_prop_ro("runtime_index", &SpikeInputView::runtime_index)
        .def("__repr__",
             [](const SpikeInputView& source) {
                 return "<mind_sim.spike_input id=" + std::to_string(source.source_id) + ">";
             });

    nb::class_<SpikeInputGroupView>(m, "spike_input_group")
        .def("__len__", &SpikeInputGroupView::size)
        .def("__getitem__",
             [](const SpikeInputGroupView& group, int index) {
                 return group.get(normalize_py_index(index, group.source_ids.size(), "spike_input_group"));
             })
        .def_prop_ro("runtime_base", &SpikeInputGroupView::runtime_base)
        .def("__repr__",
             [](const SpikeInputGroupView& group) {
                 return "<mind_sim.spike_input_group count=" + std::to_string(group.source_ids.size()) + ">";
             });

    nb::class_<NetworkView>(m, "network")
        .def("register_gid_source",
             &NetworkView::register_gid_source,
             nb::arg("gid"),
             nb::arg("source"),
             nb::arg("threshold") = nb::none())
        .def("gid_connect",
             &NetworkView::gid_connect,
             nb::arg("gid"),
             nb::arg("post"),
             nb::arg("weight"),
             nb::arg("delay"))
        .def("spike_input", &NetworkView::spike_input)
        .def("spike_inputs", &NetworkView::spike_inputs, nb::arg("count"))
        .def("spike_connect",
             &NetworkView::spike_connect,
             nb::arg("source"),
             nb::arg("post"),
             nb::arg("weight"),
             nb::arg("delay"))
        .def("__repr__", [](const NetworkView&) { return "<mind_sim.network>"; });

    nb::class_<TimeRefView>(m, "_time_ref")
        .def("__repr__", [](const TimeRefView&) { return "<mind_sim._time_ref t>"; });

    nb::class_<VectorView>(m, "Vector")
        .def(nb::init<>())
        .def("record", nb::overload_cast<const VariableRefView&>(&VectorView::record), nb::arg("ref"))
        .def("record", nb::overload_cast<const TimeRefView&>(&VectorView::record), nb::arg("ref"))
        .def("to_python", &VectorView::to_python)
        .def("save_h5",
             [](const VectorView& vec, const std::string& path, const std::string& name) {
                 mind_sim::io::save_vector_h5(vec.buffer ? vec.buffer->samples : std::vector<double>{},
                                                path,
                                                name);
             },
             nb::arg("path"),
             nb::arg("name") = "values")
        .def("size", &VectorView::size)
        .def("__len__", &VectorView::size)
        .def("__getitem__",
             [](const VectorView& vec, int index) {
                 return vec.get(index);
             })
        .def("__repr__",
             [](const VectorView& vec) {
                 return "<mind_sim.Vector size=" + std::to_string(vec.size()) + ">";
             });

    nb::class_<Sim>(m, "Sim")
        .def(nb::init<>())
        .def_rw("name", &Sim::name)
        .def("set_spike_output_enabled", &Sim::set_spike_output_enabled)
        .def("is_spike_output_enabled", &Sim::is_spike_output_enabled)
        .def("set_device", &Sim::set_device)
        .def("set_dt", &Sim::set_dt)
        .def("get_dt", &Sim::get_dt)
        .def("load_mech_metadata", &Sim::load_mech_metadata)
        .def("ion_register", &Sim::ion_register, nb::arg("ion"), nb::arg("charge"))
        .def("ion_charge", &Sim::ion_charge, nb::arg("ion_mechanism"))
        .def("get_loaded_mech_metadata_paths", &Sim::get_loaded_mech_metadata_paths)
        .def("__getattr__",
             [](const Sim& sim, const std::string& key) {
                 if (key == "name") {
                     return nb::cast(sim.name);
                 }
                 return nb::cast(sim.get_global(key));
             })
        .def("__setattr__",
             [](Sim& sim, const std::string& key, nb::handle value) {
                 if (key == "name") {
                     sim.name = nb::cast<std::string>(value);
                     return;
                 }
                 if (key == "celsius") {
                     sim.set_celsius(nb::cast<double>(value));
                     return;
                 }
                 sim.set_global(key, nb::cast<double>(value));
             })
        .def("insert",
             [](Sim& sim, const std::string& mech, const nb::kwargs& kwargs) {
                 return sim.insert(mech, parse_kwargs(kwargs));
             })
        .def("build_morphology", &Sim::build_morphology, nb::arg("morph_templates"), nb::rv_policy::reference_internal)
        .def("__len__", [](const Sim& sim) { return sim.model.population_count(); })
        .def("population", &sim_population, nb::arg("name"), nb::keep_alive<0, 1>())
        .def("network", &sim_network, nb::keep_alive<0, 1>())
        .def("build_microcircuit", &Sim::build_microcircuit)
        .def("finitialize", &Sim::finitialize, nb::arg("v_init"))
        .def("run", &Sim::run, nb::arg("tstop"), nb::call_guard<nb::gil_scoped_release>())
        .def("continue_run",
             &Sim::continue_run,
             nb::arg("runtime"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("fadvance", &Sim::fadvance, nb::call_guard<nb::gil_scoped_release>())
        .def("get_t", &Sim::get_t)
        .def("_debug_core_thread", &Sim::debug_core_thread)
        .def("get_spk_by_gid", &Sim::get_spk_by_gid, nb::arg("gid"))
        .def_prop_rw("celsius", &Sim::get_celsius, &Sim::set_celsius)
        .def_prop_ro("_ref_t", &sim_time_ref);
}
