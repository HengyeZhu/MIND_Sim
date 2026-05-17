#include "python_api/bindings/bindings.hpp"

namespace mind_sim::python_api::bindings {

void bind_macro(nb::module_& m) {
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
             nb::arg("params"),
             nb::arg("read_input_offsets"),
             nb::arg("write_exposure_offsets"))
        .def("use_neural_field",
             &network_use_neural_field,
             nb::arg("name"),
             nb::arg("rule"),
             nb::arg("node_map"),
             nb::arg("local"),
             nb::arg("state"),
             nb::arg("params"),
             nb::arg("read_input_offsets"),
             nb::arg("reducer_state_indices"),
             nb::arg("reducer_exposure_indices"))
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
        .def("roi_count", &Network::roi_count);

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

}

}  // namespace mind_sim::python_api::bindings
