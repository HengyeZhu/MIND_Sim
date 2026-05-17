include_guard(GLOBAL)

file(
  GLOB_RECURSE
  MIND_SIM_CORENEURON_SOURCES
  CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/src/micro/sim/coreneuron/*.cpp"
)
list(
  REMOVE_ITEM
  MIND_SIM_CORENEURON_SOURCES
  "${PROJECT_SOURCE_DIR}/src/micro/sim/coreneuron/apps/coreneuron.cpp"
  "${PROJECT_SOURCE_DIR}/src/micro/sim/coreneuron/mechanism/mech/enginemech.cpp"
  "${PROJECT_SOURCE_DIR}/src/micro/sim/coreneuron/mpi/lib/mpispike.cpp"
  "${PROJECT_SOURCE_DIR}/src/micro/sim/coreneuron/mpi/lib/nrnmpi.cpp"
)

set(MIND_SIM_BRIDGE_SOURCES
  src/bridge/sim/interfaces.cpp
)

set(MIND_SIM_MACRO_FRONTEND_SOURCES
  src/macro/frontend/connectivity.cpp
  src/macro/frontend/local_connectivity.cpp
  src/macro/frontend/node_to_roi_map.cpp
  src/macro/frontend/network.cpp
  src/macro/frontend/network_field.cpp
  src/macro/frontend/network_micro.cpp
)

set(MIND_SIM_MACRO_RUNTIME_SOURCES
  src/macro/sim/model.cpp
  src/macro/sim/runtime_core.cpp
  src/macro/sim/runtime.cpp
)

set(MIND_SIM_MICRO_FRONTEND_SOURCES
  src/micro/frontend/biophysical/mechanism_catalog.cpp
  src/micro/frontend/morph/cell_location.cpp
  src/micro/frontend/morph/dat_to_section.cpp
  src/micro/frontend/morph/mutable_section.cpp
  src/micro/frontend/morph/section_distance.cpp
  src/micro/frontend/morph/section_spec.cpp
  src/micro/frontend/morph/section_to_node.cpp
  src/micro/frontend/network/network_registry.cpp
  src/micro/frontend/model.cpp
  src/micro/frontend/model_runtime.cpp
)

set(MIND_SIM_MICRO_RUNTIME_SOURCES
  src/micro/sim/coreneuron_host_support.cpp
  src/micro/sim/core_neuron_data.cpp
  src/micro/sim/device.cpp
  src/micro/sim/mechanism_runtime.cpp
  src/micro/sim/micro_runtime.cpp
)

set(MIND_SIM_MIND_MOD_SOURCES
  src/mind_mod/rule_mod_analyzer.cpp
  src/mind_mod/rule_mod_codegen.cpp
  src/mind_mod/rule_mod_parser.cpp
)

set(MIND_SIM_IO_SOURCES
  src/io/result_hdf5.cpp
)

set(MIND_SIM_UTILS_SOURCES
  src/utils/dynamic_library.cpp
)

set(MIND_SIM_COSIM_SOURCES
  src/cosim/hybrid_simulator.cpp
)

set(MIND_SIM_NATIVE_CORE_SOURCES
  ${MIND_SIM_BRIDGE_SOURCES}
  ${MIND_SIM_MACRO_FRONTEND_SOURCES}
  ${MIND_SIM_MACRO_RUNTIME_SOURCES}
  ${MIND_SIM_IO_SOURCES}
  ${MIND_SIM_MIND_MOD_SOURCES}
  ${MIND_SIM_UTILS_SOURCES}
  ${MIND_SIM_MICRO_FRONTEND_SOURCES}
  ${MIND_SIM_MICRO_RUNTIME_SOURCES}
  ${MIND_SIM_COSIM_SOURCES}
)

set(MIND_SIM_PYTHON_BINDING_SOURCES
  src/python_api/bindings/bindings_io.cpp
  src/python_api/bindings/bindings_macro.cpp
  src/python_api/bindings/bindings_micro.cpp
  src/python_api/bindings/bindings_rules.cpp
  src/python_api/bindings/module.cpp
)
