#include "python_api/bindings/bindings.hpp"

#include <nanobind/ndarray.h>

#include <cstdint>

namespace mind_sim::python_api::bindings {

namespace {

using InitialHistoryArray3 =
    nb::ndarray<nb::numpy, const double, nb::ndim<3>, nb::c_contig>;
using InitialHistoryArray4 =
    nb::ndarray<nb::numpy, const double, nb::ndim<4>, nb::c_contig>;

std::vector<std::vector<double>> square_connectivity_matrix(const Connectivity& connectivity,
                                                            const std::vector<double>& flat) {
    const int n = connectivity.roi_count();
    std::vector<std::vector<double>> out(static_cast<std::size_t>(n),
                                         std::vector<double>(static_cast<std::size_t>(n), 0.0));
    for (int target = 0; target < n; ++target) {
        for (int source = 0; source < n; ++source) {
            out[static_cast<std::size_t>(target)][static_cast<std::size_t>(source)] =
                flat[static_cast<std::size_t>(target * n + source)];
        }
    }
    return out;
}

Network::InitialHistoryLayout parse_initial_history_layout(const std::string& layout) {
    if (layout == "time_output_roi" || layout == "tvb") {
        return Network::InitialHistoryLayout::TimeOutputRoi;
    }
    if (layout == "time_roi_output") {
        return Network::InitialHistoryLayout::TimeRoiOutput;
    }
    throw std::runtime_error(
        "initial_history layout must be 'time_output_roi', 'time_roi_output', or 'tvb'");
}

void set_initial_history_from_array(NetworkBuilder& builder,
                                    const double* data,
                                    std::size_t size,
                                    int time_count,
                                    int axis1_count,
                                    int axis2_count,
                                    std::vector<std::string> output_names,
                                    const std::string& layout) {
    std::vector<double> values(data, data + size);
    builder.set_initial_history(std::move(output_names),
                                time_count,
                                axis1_count,
                                axis2_count,
                                std::move(values),
                                parse_initial_history_layout(layout));
}

}  // namespace

void bind_macro(nb::module_& m) {
    nb::class_<ScalarBuffer>(m, "ScalarBuffer")
        .def(nb::init<>())
        .def(nb::init<std::size_t>(), nb::arg("output_count"))
        .def_rw("values", &ScalarBuffer::values)
        .def("get",
             [](const ScalarBuffer& buffer, int output_id) {
                 return buffer.get(output_id, "ScalarBuffer.get");
             },
             nb::arg("output_id"))
        .def("set",
             [](ScalarBuffer& buffer, int output_id, double value) {
                 buffer.at(output_id, "ScalarBuffer.set") = value;
             },
             nb::arg("output_id"),
             nb::arg("value"))
        .def("__len__", &ScalarBuffer::size)
        .def("__repr__",
             [](const ScalarBuffer& buffer) {
                 return "<mind_sim.ScalarBuffer outputs=" + std::to_string(buffer.size()) + ">";
             });

    nb::class_<ROI>(m, "ROI")
        .def_ro("index", &ROI::index)
        .def_ro("label", &ROI::label)
        .def("__repr__",
             [](const ROI& roi) {
                 return "<mind_sim.ROI index=" + std::to_string(roi.index) +
                        " label='" + roi.label + "'>";
             });

    nb::class_<Connectivity>(m, "Connectivity")
        .def(nb::init<std::vector<std::string>,
                      std::vector<std::vector<double>>,
                      std::vector<std::vector<double>>>(),
             nb::arg("labels"),
             nb::arg("weights"),
             nb::arg("delays"))
        .def_static("from_csv", &Connectivity::from_csv, nb::arg("path"))
        .def_prop_ro("labels", [](const Connectivity& connectivity) { return connectivity.labels(); })
        .def_prop_ro("weights",
                     [](const Connectivity& connectivity) {
                         return square_connectivity_matrix(connectivity, connectivity.weights());
                     })
        .def_prop_ro("delays",
                     [](const Connectivity& connectivity) {
                         return square_connectivity_matrix(connectivity, connectivity.delays());
                     })
        .def("rois", &Connectivity::rois)
        .def("roi_count", &Connectivity::roi_count)
        .def("roi_index", &Connectivity::roi_index, nb::arg("label"))
        .def("weight_at", &Connectivity::weight_at, nb::arg("target_roi"), nb::arg("source_roi"))
        .def("delay_at", &Connectivity::delay_at, nb::arg("target_roi"), nb::arg("source_roi"))
        .def("min_positive_delay", &Connectivity::min_positive_delay);

    nb::class_<NodeToRoiMap>(m, "NodeToRoiMap")
        .def(nb::init<std::vector<int>, std::vector<double>>(),
             nb::arg("node_to_roi"),
             nb::arg("node_weights") = std::vector<double>{})
        .def_static("from_surface",
                    &node_to_roi_map_from_surface,
                    nb::arg("surface"),
                    nb::arg("node_weights") = std::optional<std::vector<double>>{})
        .def_static("from_file",
                    &NodeToRoiMap::from_file,
                    nb::arg("node_to_roi_path"),
                    nb::arg("node_weights_path") = std::string{})
        .def_prop_ro("node_count", &NodeToRoiMap::node_count)
        .def_prop_ro("node_to_roi", &NodeToRoiMap::node_to_roi)
        .def_prop_ro("node_weights", &NodeToRoiMap::node_weights)
        .def("__repr__",
             [](const NodeToRoiMap& mapping) {
                 return "<mind_sim.NodeToRoiMap nodes=" +
                        std::to_string(mapping.node_count()) + ">";
             });

    nb::class_<LocalConnectivity>(m, "LocalConnectivity")
        .def(nb::init<int, std::vector<int>, std::vector<int>, std::vector<double>>(),
             nb::arg("node_count"),
             nb::arg("indptr"),
             nb::arg("indices"),
             nb::arg("weights"))
        .def_static("from_arrays",
                    [](int node_count,
                       std::vector<int> indptr,
                       std::vector<int> indices,
                       std::vector<double> weights) {
                        return LocalConnectivity(
                            node_count,
                            std::move(indptr),
                            std::move(indices),
                            std::move(weights));
                    },
                    nb::arg("node_count"),
                    nb::arg("indptr"),
                    nb::arg("indices"),
                    nb::arg("weights"))
        .def_static("from_edges",
                    &local_connectivity_from_edges,
                    nb::arg("node_count"),
                    nb::arg("edges"))
        .def_static("from_csr", &local_connectivity_from_csr, nb::arg("matrix"))
        .def_static("from_surface",
                    &local_connectivity_from_surface,
                    nb::arg("surface"),
                    nb::arg("node_count") = std::optional<int>{})
        .def_prop_ro("node_count", &LocalConnectivity::node_count)
        .def_prop_ro("nnz", &LocalConnectivity::nnz)
        .def_prop_ro("indptr", &LocalConnectivity::indptr)
        .def_prop_ro("indices", &LocalConnectivity::indices)
        .def_prop_ro("weights", &LocalConnectivity::weights)
        .def("__repr__",
             [](const LocalConnectivity& local) {
                 return "<mind_sim.LocalConnectivity nodes=" +
                        std::to_string(local.node_count()) + " edges=" +
                        std::to_string(local.nnz()) + ">";
             });

    nb::class_<Network>(m, "_Network")
        .def(nb::init<Connectivity, std::vector<std::string>, std::vector<std::string>, std::vector<int>, std::vector<int>>(),
             nb::arg("connectivity"),
             nb::arg("inputs"),
             nb::arg("outputs"),
             nb::arg("recorded_rois"),
             nb::arg("recorded_outputs"))
        .def("roi",
             nb::overload_cast<int>(&Network::roi, nb::const_),
             nb::arg("index"))
        .def("roi",
             nb::overload_cast<const std::string&>(&Network::roi, nb::const_),
             nb::arg("label"))
        .def("rois", &Network::rois)
        .def("inputs", &Network::inputs)
        .def("outputs", &Network::outputs)
        .def("recorded_rois", &Network::recorded_rois)
        .def("recorded_outputs", &Network::recorded_outputs)
        .def("set_recorded_rois", &Network::set_recorded_rois, nb::arg("recorded_rois"))
        .def("set_recorded_outputs", &Network::set_recorded_outputs, nb::arg("recorded_outputs"))
        .def("input_index", &Network::input_index, nb::arg("input"))
        .def("input_count", &Network::input_count)
        .def("output_index", &Network::output_index, nb::arg("output"))
        .def("output_count", &Network::output_count)
        .def("set_dc_input",
             &Network::set_dc_input,
             nb::arg("roi"),
             nb::arg("input"))
        .def("set_dc_input_value",
             &Network::set_dc_input_value,
             nb::arg("roi"),
             nb::arg("input"),
             nb::arg("value"))
        .def("macro2macro",
             &Network::macro_to_macro,
             nb::arg("source_roi"),
             nb::arg("target_roi"),
             nb::arg("rule"),
             nb::arg("params"),
             nb::arg("source_exposure_offsets"),
             nb::arg("target_input_offsets"))
        .def("use_region_rule",
             &Network::use_region_rule,
             nb::arg("roi"),
             nb::arg("rule"),
             nb::arg("state"),
             nb::arg("params"),
             nb::arg("target_input_offsets"),
             nb::arg("source_exposure_offsets"))
        .def("use_neural_field",
             &network_use_neural_field,
             nb::arg("name"),
             nb::arg("rule"),
             nb::arg("node_map"),
             nb::arg("local"),
             nb::arg("state"),
             nb::arg("params"),
             nb::arg("target_input_offsets"),
             nb::arg("reducer_state_indices"),
             nb::arg("reducer_output_indices"))
        .def("bind_micro_roi",
             &network_bind_micro_roi,
             nb::arg("micro_circuit_index"),
             nb::arg("roi"),
             nb::arg("gid_range_begins"),
             nb::arg("gid_range_ends"))
        .def("roi_index", &Network::roi_index, nb::arg("label"))
        .def("roi_count", &Network::roi_count);

    nb::class_<NetworkBuilder>(m, "_NetworkBuilder")
        .def(nb::init<Connectivity>(),
             nb::arg("connectivity"))
        .def("roi",
             nb::overload_cast<int>(&NetworkBuilder::roi, nb::const_),
             nb::arg("index"))
        .def("roi",
             nb::overload_cast<const std::string&>(&NetworkBuilder::roi, nb::const_),
             nb::arg("label"))
        .def("rois", &NetworkBuilder::rois)
        .def("roi_count", &NetworkBuilder::roi_count)
        .def("min_positive_delay", &NetworkBuilder::min_positive_delay)
        .def("record",
             &NetworkBuilder::record,
             nb::arg("roi"),
             nb::arg("output"))
        .def("record_rois", &NetworkBuilder::record_rois, nb::arg("roi_indices"))
        .def("record_all_rois", &NetworkBuilder::record_all_rois)
        .def("record_outputs", &NetworkBuilder::record_outputs, nb::arg("output_names"))
        .def("record_all_outputs", &NetworkBuilder::record_all_outputs)
        .def("set_dt", &NetworkBuilder::set_dt, nb::arg("dt"))
        .def("set_exchange_window",
             &NetworkBuilder::set_exchange_window,
             nb::arg("exchange_window"))
        .def("load_mech",
             &NetworkBuilder::load_mech,
             nb::arg("directory"))
        .def("set_initial_history",
             [](NetworkBuilder& builder,
                InitialHistoryArray3 history,
                std::vector<std::string> output_names,
                std::string layout) {
                 set_initial_history_from_array(builder,
                                                history.data(),
                                                history.size(),
                                                static_cast<int>(history.shape(0)),
                                                static_cast<int>(history.shape(1)),
                                                static_cast<int>(history.shape(2)),
                                                std::move(output_names),
                                                layout);
             },
             nb::arg("history"),
             nb::arg("outputs") = std::vector<std::string>{},
             nb::arg("layout") = "time_output_roi")
        .def("set_initial_history",
             [](NetworkBuilder& builder,
                InitialHistoryArray4 history,
                std::vector<std::string> output_names,
                std::string layout) {
                 if (history.shape(3) != 1) {
                     throw std::runtime_error("initial_history mode axis must have length 1");
                 }
                 set_initial_history_from_array(builder,
                                                history.data(),
                                                history.size(),
                                                static_cast<int>(history.shape(0)),
                                                static_cast<int>(history.shape(1)),
                                                static_cast<int>(history.shape(2)),
                                                std::move(output_names),
                                                layout);
             },
             nb::arg("history"),
             nb::arg("outputs") = std::vector<std::string>{},
             nb::arg("layout") = "tvb")
        .def("set_dc_input",
             &NetworkBuilder::set_dc_input,
             nb::arg("roi"),
             nb::arg("values"))
        .def("use_region",
             &NetworkBuilder::use_region,
             nb::arg("roi"),
             nb::arg("library_path"),
             nb::arg("initial_state") = std::unordered_map<std::string, double>{},
             nb::arg("params") = std::unordered_map<std::string, double>{})
        .def("use_neural_field",
             nb::overload_cast<std::string,
                               std::string,
                               NodeToRoiMap,
                               LocalConnectivity,
                               std::unordered_map<std::string, double>,
                               std::unordered_map<std::string, double>>(&NetworkBuilder::use_neural_field),
             nb::arg("name"),
             nb::arg("library_path"),
             nb::arg("node_map"),
             nb::arg("local"),
             nb::arg("initial_state") = std::unordered_map<std::string, double>{},
             nb::arg("params") = std::unordered_map<std::string, double>{})
        .def("use_neural_field",
             nb::overload_cast<std::string,
                               std::string,
                               NodeToRoiMap,
                               std::unordered_map<std::string, double>,
                               std::unordered_map<std::string, double>>(&NetworkBuilder::use_neural_field),
             nb::arg("name"),
             nb::arg("library_path"),
             nb::arg("node_map"),
             nb::arg("initial_state") = std::unordered_map<std::string, double>{},
             nb::arg("params") = std::unordered_map<std::string, double>{})
        .def("macro2macro",
             &NetworkBuilder::macro2macro,
             nb::arg("source_roi"),
             nb::arg("target_roi"),
             nb::arg("library_path"),
             nb::arg("params") = std::unordered_map<std::string, double>{})
        .def("use_micro",
             nb::overload_cast<int>(&NetworkBuilder::use_micro),
             nb::arg("roi"))
        .def("macro2micro",
             &NetworkBuilder::macro2micro,
             nb::arg("roi"),
             nb::arg("library_path"),
             nb::arg("target"),
             nb::arg("weight"),
             nb::arg("delay"),
             nb::arg("state") = std::unordered_map<std::string, double>{},
             nb::arg("params") = std::unordered_map<std::string, double>{})
        .def("micro2macro",
             &NetworkBuilder::micro2macro,
             nb::arg("roi"),
             nb::arg("library_path"),
             nb::arg("sid"),
             nb::arg("state") = std::unordered_map<std::string, double>{},
             nb::arg("params") = std::unordered_map<std::string, double>{})
        .def("build", &NetworkBuilder::build);

    nb::class_<MacroConfig>(m, "_MacroConfig")
        .def("load_mech", &MacroConfig::load_mech, nb::arg("directory"))
        .def("dt", &MacroConfig::set_dt, nb::arg("dt"))
        .def("exchange_window",
             &MacroConfig::set_exchange_window,
             nb::arg("exchange_window"))
        .def("apply", &MacroConfig::apply, nb::arg("builder"));
    static MacroConfig macro_config;
    m.attr("_macro_config") = nb::cast(&macro_config);

    nb::class_<mind_sim::macro::sim::RecordTable>(m, "RecordTable")
        .def_ro("roi_count", &mind_sim::macro::sim::RecordTable::roi_count)
        .def_ro("output_count", &mind_sim::macro::sim::RecordTable::output_count)
        .def_ro("roi_indices", &mind_sim::macro::sim::RecordTable::roi_indices)
        .def_ro("output_indices", &mind_sim::macro::sim::RecordTable::output_indices)
        .def_ro("values", &mind_sim::macro::sim::RecordTable::values)
        .def_prop_ro("recorded_roi_count",
                     &mind_sim::macro::sim::RecordTable::recorded_roi_count)
        .def_prop_ro("recorded_output_count",
                     &mind_sim::macro::sim::RecordTable::recorded_output_count)
        .def_prop_ro("sample_count", &mind_sim::macro::sim::RecordTable::sample_count);

    nb::class_<mind_sim::macro::sim::MacroSimulationResult>(m, "MacroSimulationResult")
        .def_ro("times", &mind_sim::macro::sim::MacroSimulationResult::times)
        .def_ro("records", &mind_sim::macro::sim::MacroSimulationResult::records)
        .def("save_h5",
             &mind_sim::io::save_macro_result_h5,
             nb::arg("path"),
             nb::arg("output_names"),
             nb::arg("roi_labels"),
             nb::arg("timing_s") = std::vector<double>{},
             nb::arg("metadata") = std::vector<double>{});

    nb::class_<MacroRuntime>(m, "MacroRuntime")
        .def(nb::init<Network>(), nb::arg("network"))
        .def("run",
             nb::overload_cast<double>(&MacroRuntime::run),
             nb::arg("t_stop"),
             nb::call_guard<nb::gil_scoped_release>());

    nb::class_<mind_sim::micro::sim::MicroSpikeTable>(m, "MicroSpikeTable")
        .def_ro("time", &mind_sim::micro::sim::MicroSpikeTable::time)
        .def_ro("gid", &mind_sim::micro::sim::MicroSpikeTable::gid)
        .def_prop_ro("size", &mind_sim::micro::sim::MicroSpikeTable::size);

    nb::class_<mind_sim::micro::sim::MicroEventTable>(m, "MicroEventTable")
        .def_ro("time", &mind_sim::micro::sim::MicroEventTable::time)
        .def_ro("index", &mind_sim::micro::sim::MicroEventTable::index)
        .def_prop_ro("size", &mind_sim::micro::sim::MicroEventTable::size);

    nb::class_<mind_sim::cosim::SimulationResult>(m, "SimulationResult")
        .def_ro("times", &mind_sim::cosim::SimulationResult::times)
        .def_ro("records", &mind_sim::cosim::SimulationResult::records)
        .def("save_h5",
             &mind_sim::io::save_cosim_result_h5,
             nb::arg("path"),
             nb::arg("output_names"),
             nb::arg("roi_labels"),
             nb::arg("timing_s") = std::vector<double>{},
             nb::arg("metadata") = std::vector<double>{});

    nb::class_<mind_sim::cosim::Simulator>(m, "Simulator")
        .def(nb::init<Network, std::uint64_t>(),
             nb::arg("network"),
             nb::arg("macro2micro_seed") = 1)
        .def("run",
             &mind_sim::cosim::Simulator::run,
             nb::arg("t_stop"),
             nb::call_guard<nb::gil_scoped_release>());

}

}  // namespace mind_sim::python_api::bindings
