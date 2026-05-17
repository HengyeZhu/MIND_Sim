include_guard(GLOBAL)

include(ProcessorCount)
ProcessorCount(MIND_SIM_HOST_PROCESSOR_COUNT)
if(NOT MIND_SIM_HOST_PROCESSOR_COUNT)
  set(MIND_SIM_HOST_PROCESSOR_COUNT 4)
endif()
set(MIND_SIM_NMODL_BUILD_JOBS "${MIND_SIM_HOST_PROCESSOR_COUNT}" CACHE STRING "Parallel jobs used to build the embedded NMODL executable")

set(MIND_SIM_NMODL_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/micro/mod_compiler/nmodl")
set(MIND_SIM_NMODL_BUILD_DIR "${PROJECT_BINARY_DIR}/nmodl")
set(MIND_SIM_NMODL "${MIND_SIM_NMODL_BUILD_DIR}/bin/nmodl")
set(MIND_SIM_NMODL_BUILDER "${PROJECT_SOURCE_DIR}/src/micro/mod_compiler/build_embedded_nmodl.py")
set(MIND_SIM_NMODL_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
if(NOT MIND_SIM_NMODL_BUILD_TYPE)
  set(MIND_SIM_NMODL_BUILD_TYPE Release)
endif()

set(MIND_SIM_NMODL_STAMP "${MIND_SIM_NMODL_BUILD_DIR}/mind_sim_nmodl.stamp")
add_custom_command(
  OUTPUT "${MIND_SIM_NMODL_STAMP}"
  COMMAND "${Python_EXECUTABLE}" "${MIND_SIM_NMODL_BUILDER}"
          --source-dir "${MIND_SIM_NMODL_SOURCE_DIR}"
          --build-dir "${MIND_SIM_NMODL_BUILD_DIR}"
          --python "${Python_EXECUTABLE}"
          --cxx "${CMAKE_CXX_COMPILER}"
          --build-type "${MIND_SIM_NMODL_BUILD_TYPE}"
          --jobs "${MIND_SIM_NMODL_BUILD_JOBS}"
  COMMAND "${CMAKE_COMMAND}" -E touch "${MIND_SIM_NMODL_STAMP}"
  BYPRODUCTS "${MIND_SIM_NMODL}"
  DEPENDS "${MIND_SIM_NMODL_BUILDER}"
  COMMENT "Building embedded NMODL"
  COMMAND_EXPAND_LISTS
  VERBATIM
)
add_custom_target(mind_sim_nmodl DEPENDS "${MIND_SIM_NMODL_STAMP}")

set(MIND_SIM_CORENEURON_MODFILE_DIR
  "${PROJECT_SOURCE_DIR}/src/micro/sim/coreneuron/mechanism/mech/modfile"
)
set(MIND_SIM_DEFAULT_MECHANISM_NAMES
  passive
  hh
  expsyn
  exp2syn
  netstim
  stim
  svclmp
)

set(MIND_SIM_DEFAULT_MECHANISM_MODFILES)
foreach(name IN LISTS MIND_SIM_DEFAULT_MECHANISM_NAMES)
  list(APPEND MIND_SIM_DEFAULT_MECHANISM_MODFILES
    "${MIND_SIM_CORENEURON_MODFILE_DIR}/${name}.mod"
  )
endforeach()

mind_sim_add_mod_codegen_target(
  TARGET mind_sim_default_mechanisms
  OUTPUT_DIR "${PROJECT_BINARY_DIR}/default_mechanisms"
  MODL_REG_NAME mind_default_modl_reg
  OUT_SOURCES MIND_SIM_DEFAULT_MECHANISM_GENERATED_SOURCES
  MODFILES ${MIND_SIM_DEFAULT_MECHANISM_MODFILES}
  INCLUDES "${PROJECT_SOURCE_DIR}/src/micro/sim"
  DEFINES ${MIND_SIM_CORENEURON_MOD_DEFINES}
  COMMENT "Building default MOD mechanisms"
)
