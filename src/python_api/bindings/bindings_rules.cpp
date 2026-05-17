#include "python_api/bindings/bindings.hpp"

namespace mind_sim::python_api::bindings {

void bind_rules(nb::module_& m) {
    m.def("_codegen_kernel_name",
          [](const std::string& name, const std::string& what) {
              return mind_sim::utils::rule_source::names({name}, what).front();
          },
          nb::arg("name"),
          nb::arg("what"));
    m.def("_translate_mind_mod_to_cpp",
          &mind_sim::mind_mod::compiled_rule_source,
          nb::arg("source"),
          nb::arg("origin"));
    m.def("_inspect_mind_mod_library",
          &inspect_mind_mod_library,
          nb::arg("library_path"));

    nb::class_<RegionRule>(m, "_RegionRule")
        .def_prop_ro("name", &RegionRule::name)
        .def_prop_ro("input_count", &RegionRule::input_count)
        .def_prop_ro("exposure_count", &RegionRule::exposure_count)
        .def_prop_ro("state_count", &RegionRule::state_count)
        .def_prop_ro("param_count", &RegionRule::param_count)
        .def_prop_ro("library_path", &RegionRule::library_path);

    m.def("_load_region_rule",
          [](const std::string& library_path) {
              return std::make_shared<RegionRule>(library_path);
          },
          nb::arg("library_path"));

    nb::class_<NeuralFieldRule>(m, "_NeuralFieldRule")
        .def_prop_ro("name", &NeuralFieldRule::name)
        .def_prop_ro("input_count", &NeuralFieldRule::input_count)
        .def_prop_ro("state_count", &NeuralFieldRule::state_count)
        .def_prop_ro("param_count", &NeuralFieldRule::param_count)
        .def_prop_ro("library_path", &NeuralFieldRule::library_path);

    m.def("_load_neural_field_rule",
          [](const std::string& library_path) {
              return std::make_shared<NeuralFieldRule>(library_path);
          },
          nb::arg("library_path"));

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

}

}  // namespace mind_sim::python_api::bindings
