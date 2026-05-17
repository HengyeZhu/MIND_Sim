#include "python_api/bindings/bindings.hpp"

namespace mind_sim::python_api::bindings {

void bind_io(nb::module_& m) {
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

}

}  // namespace mind_sim::python_api::bindings
