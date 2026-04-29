cmake_minimum_required(VERSION 3.20)
project(mind_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "Release C++ flags" FORCE)
endif()

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(OpenMP REQUIRED)
set(MIND_SIM_RUNTIME_FP_FLAGS "-O3" "-march=native" "-mtune=native" "-fno-math-errno" "-ffp-contract=off" "-fno-semantic-interposition" "-funroll-loops")
set(MIND_SIM_MODCC_FP_FLAGS "-O3" "-march=native" "-mtune=native" "-fno-math-errno" "-ffp-contract=off" "-fno-semantic-interposition" "-funroll-loops")
string(REPLACE ";" " " MIND_SIM_MODCC_FP_FLAGS_TEXT "${MIND_SIM_MODCC_FP_FLAGS}")
string(STRIP "${OpenMP_CXX_FLAGS} ${MIND_SIM_MODCC_FP_FLAGS_TEXT}" MIND_SIM_MODCC_CXX_FLAGS)
set(MIND_SIM_MODCC_LINK_OPTIONS ${OpenMP_CXX_FLAGS})
if(OpenMP_CXX_LIBRARIES)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND MIND_SIM_MODCC_LINK_OPTIONS "-Wl,--no-as-needed")
  endif()
  list(APPEND MIND_SIM_MODCC_LINK_OPTIONS ${OpenMP_CXX_LIBRARIES})
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND MIND_SIM_MODCC_LINK_OPTIONS "-Wl,--as-needed")
  endif()
endif()
string(REPLACE ";" " " MIND_SIM_MODCC_LINK_FLAGS "${MIND_SIM_MODCC_LINK_OPTIONS}")
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.prefix)"
  OUTPUT_VARIABLE Python_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_path(MIND_SIM_HDF5_INCLUDE_DIR hdf5.h
  HINTS "${Python_PREFIX}/include"
  REQUIRED
)
find_library(MIND_SIM_HDF5_LIBRARY hdf5
  HINTS "${Python_PREFIX}/lib"
  REQUIRED
)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind CONFIG REQUIRED)

set(MIND_SIM_MODCC "${CMAKE_CURRENT_SOURCE_DIR}/src/micro/mod_compiler/mind_modcc.py")
set(MIND_SIM_NMODL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/micro/mod_compiler/nmodl")
set(MIND_SIM_NMODL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/nmodl")
set(MIND_SIM_NMODL "${MIND_SIM_NMODL_BUILD_DIR}/bin/nmodl")
set(MIND_SIM_NMODL_BUILDER "${CMAKE_CURRENT_SOURCE_DIR}/src/micro/mod_compiler/build_embedded_nmodl.py")
set(MIND_SIM_NMODL_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
if(NOT MIND_SIM_NMODL_BUILD_TYPE)
  set(MIND_SIM_NMODL_BUILD_TYPE Release)
endif()
add_custom_target(
  mind_sim_nmodl
  COMMAND "${Python_EXECUTABLE}" "${MIND_SIM_NMODL_BUILDER}"
          --source-dir "${MIND_SIM_NMODL_SOURCE_DIR}"
          --build-dir "${MIND_SIM_NMODL_BUILD_DIR}"
          --python "${Python_EXECUTABLE}"
          --cxx "${CMAKE_CXX_COMPILER}"
          --build-type "${MIND_SIM_NMODL_BUILD_TYPE}"
          --jobs 8
  BYPRODUCTS "${MIND_SIM_NMODL}"
  COMMENT "Building embedded NMODL"
  COMMAND_EXPAND_LISTS
  VERBATIM
)

set(MIND_SIM_CORENEURON_MODFILE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/src/micro/sim/coreneuron/mechanism/mech/modfile"
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
set(MIND_SIM_DEFAULT_MECHANISM_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/default_mechanisms")
set(MIND_SIM_DEFAULT_MECHANISM_OUTPUT_DIR "${MIND_SIM_DEFAULT_MECHANISM_BUILD_DIR}")
set(MIND_SIM_DEFAULT_MECHANISM_GENERATED_DIR
  "${MIND_SIM_DEFAULT_MECHANISM_OUTPUT_DIR}/core"
)
set(MIND_SIM_DEFAULT_MECHANISM_GENERATED_SOURCES)
foreach(name IN LISTS MIND_SIM_DEFAULT_MECHANISM_NAMES)
  list(APPEND MIND_SIM_DEFAULT_MECHANISM_GENERATED_SOURCES
    "${MIND_SIM_DEFAULT_MECHANISM_GENERATED_DIR}/${name}.cpp"
  )
endforeach()
list(APPEND MIND_SIM_DEFAULT_MECHANISM_GENERATED_SOURCES
  "${MIND_SIM_DEFAULT_MECHANISM_GENERATED_DIR}/mind_default_modl_reg.cpp"
)
set(MIND_SIM_DEFAULT_MECHANISM_STAMP
  "${MIND_SIM_DEFAULT_MECHANISM_BUILD_DIR}/default_mechanisms.stamp"
)
file(MAKE_DIRECTORY "${MIND_SIM_DEFAULT_MECHANISM_BUILD_DIR}")

add_custom_command(
  OUTPUT
    "${MIND_SIM_DEFAULT_MECHANISM_STAMP}"
    ${MIND_SIM_DEFAULT_MECHANISM_GENERATED_SOURCES}
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${MIND_SIM_DEFAULT_MECHANISM_BUILD_DIR}"
  COMMAND "${Python_EXECUTABLE}" "${MIND_SIM_MODCC}"
          --nmodl "${MIND_SIM_NMODL}"
          --cxx "${CMAKE_CXX_COMPILER}"
          --output "${MIND_SIM_DEFAULT_MECHANISM_OUTPUT_DIR}"
          --generate-only
          --modl-reg-name mind_default_modl_reg
          "--cxx-flag=${MIND_SIM_MODCC_CXX_FLAGS}"
          "--link-flag=${MIND_SIM_MODCC_LINK_FLAGS}"
          --include "${CMAKE_CURRENT_SOURCE_DIR}/src/micro/sim"
          --define CORENEURON_BUILD
          --define CORENRN_BUILD=1
          --define VECTORIZE=1
          --define HAVE_MALLOC_H
          --define EIGEN_DONT_PARALLELIZE
          --define LAYOUT=0
          --define ENABLE_SPLAYTREE_QUEUING
          --define NRNMPI=0
          --define NRN_MULTISEND=0
          --define DISABLE_HOC_EXP
          --define NET_RECEIVE_BUFFERING=0
          --define NRN_PRCELLSTATE=0
          ${MIND_SIM_DEFAULT_MECHANISM_MODFILES}
  COMMAND "${CMAKE_COMMAND}" -E touch "${MIND_SIM_DEFAULT_MECHANISM_STAMP}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS mind_sim_nmodl "${MIND_SIM_NMODL}" ${MIND_SIM_DEFAULT_MECHANISM_MODFILES} "${MIND_SIM_MODCC}"
  COMMENT "Building default MOD mechanisms"
  COMMAND_EXPAND_LISTS
  VERBATIM
)
add_custom_target(mind_sim_default_mechanisms DEPENDS "${MIND_SIM_DEFAULT_MECHANISM_STAMP}")
set_source_files_properties(${MIND_SIM_DEFAULT_MECHANISM_GENERATED_SOURCES}
  PROPERTIES GENERATED TRUE
)

set(MIND_SIM_BUILD_PYTHON_API_DIR "${CMAKE_CURRENT_BINARY_DIR}/mind_sim")
add_custom_target(
  mind_sim_python_api
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${MIND_SIM_BUILD_PYTHON_API_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
          "${CMAKE_CURRENT_SOURCE_DIR}/src/python_api/mind_sim"
          "${MIND_SIM_BUILD_PYTHON_API_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E rm -rf
          "${MIND_SIM_BUILD_PYTHON_API_DIR}/default_mechanisms"
  DEPENDS mind_sim_default_mechanisms
  COMMENT "Staging Python API"
  VERBATIM
)

file(
  GLOB_RECURSE
  MIND_SIM_CORENEURON_SOURCES
  CONFIGURE_DEPENDS
  src/micro/sim/coreneuron/*.cpp
)
list(
  REMOVE_ITEM
  MIND_SIM_CORENEURON_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/src/micro/sim/coreneuron/apps/coreneuron.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/micro/sim/coreneuron/mechanism/mech/enginemech.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/micro/sim/coreneuron/mpi/lib/mpispike.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/src/micro/sim/coreneuron/mpi/lib/nrnmpi.cpp
)

set(MIND_SIM_MICRO_FRONTEND_SOURCES
  src/bridge/sim/interfaces.cpp
  src/macro/frontend/network.cpp
  src/macro/sim/model.cpp
  src/macro/sim/rule_codegen.cpp
  src/macro/sim/runtime_core.cpp
  src/macro/sim/runtime.cpp
  src/io/result_hdf5.cpp
  src/bridge/sim/rule_codegen.cpp
  src/utils/dynamic_library.cpp
  ${MIND_SIM_DEFAULT_MECHANISM_GENERATED_SOURCES}
  src/micro/frontend/morph/cell_location.cpp
  src/micro/frontend/morph/dat_to_section.cpp
  src/micro/frontend/morph/mutable_section.cpp
  src/micro/frontend/morph/section_distance.cpp
  src/micro/frontend/morph/section_spec.cpp
  src/micro/frontend/morph/section_to_node.cpp
  src/micro/frontend/biophysical/mechanism_catalog.cpp
  src/micro/frontend/network/network_registry.cpp
  src/micro/frontend/model.cpp
  ${MIND_SIM_CORENEURON_SOURCES}
  src/micro/sim/coreneuron_host_support.cpp
  src/micro/sim/mechanism_runtime.cpp
  src/micro/sim/micro_runtime.cpp
  src/cosim/hybrid_simulator.cpp
)

nanobind_add_module(
  _native
  STABLE_ABI
  NOMINSIZE
  NB_STATIC
  ${MIND_SIM_MICRO_FRONTEND_SOURCES}
  src/python_api/bindings/module.cpp
)

add_dependencies(_native mind_sim_default_mechanisms)
add_dependencies(_native mind_sim_python_api)

target_include_directories(_native PRIVATE
  src
  src/micro/frontend
  src/micro/sim
)

set_target_properties(_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${MIND_SIM_BUILD_PYTHON_API_DIR}"
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN OFF
)

target_compile_definitions(_native PRIVATE
  MIND_CORE_MECH_ABI_VERSION=4
  CORENEURON_BUILD
  CORENRN_BUILD=1
  VECTORIZE=1
  HAVE_MALLOC_H
  EIGEN_DONT_PARALLELIZE
  LAYOUT=0
  ENABLE_SPLAYTREE_QUEUING
  NRNMPI=0
  NRN_MULTISEND=0
  DISABLE_HOC_EXP
  NET_RECEIVE_BUFFERING=0
  MIND_SIM_MODCC="${MIND_SIM_MODCC}"
  MIND_SIM_NMODL_EXECUTABLE="${MIND_SIM_NMODL}"
  MIND_SIM_NMODL_SOURCE_DIR="${MIND_SIM_NMODL_SOURCE_DIR}"
  MIND_SIM_PYTHON_EXECUTABLE="${Python_EXECUTABLE}"
  MIND_SIM_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
  MIND_SIM_MODCC_CXX_FLAGS="${MIND_SIM_MODCC_CXX_FLAGS}"
  MIND_SIM_MODCC_LINK_FLAGS="${MIND_SIM_MODCC_LINK_FLAGS}"
  MIND_SIM_MECHANISM_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/micro/sim"
)

target_include_directories(_native PRIVATE ${MIND_SIM_HDF5_INCLUDE_DIR})
target_link_libraries(_native PRIVATE ${CMAKE_DL_LIBS} ${MIND_SIM_HDF5_LIBRARY} OpenMP::OpenMP_CXX)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  target_compile_options(_native PRIVATE -O2 ${MIND_SIM_RUNTIME_FP_FLAGS})
endif()

enable_testing()
add_test(
  NAME mind_sim_ion_channel_check
  COMMAND "${Python_EXECUTABLE}" -S "${CMAKE_CURRENT_SOURCE_DIR}/tests/micro/ion_channel_check.py"
)
set_tests_properties(
  mind_sim_ion_channel_check
  PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
)
add_test(
  NAME mind_sim_custom_ion_check
  COMMAND "${Python_EXECUTABLE}" -S "${CMAKE_CURRENT_SOURCE_DIR}/tests/micro/custom_ion_check.py"
)
set_tests_properties(
  mind_sim_custom_ion_check
  PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
)
add_test(
  NAME mind_sim_ion_register_check
  COMMAND "${Python_EXECUTABLE}" -S "${CMAKE_CURRENT_SOURCE_DIR}/tests/micro/ion_register_check.py"
)
set_tests_properties(
  mind_sim_ion_register_check
  PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
)
add_test(
  NAME mind_sim_mod_global_parameter_check
  COMMAND "${Python_EXECUTABLE}" -S "${CMAKE_CURRENT_SOURCE_DIR}/tests/micro/mod_global_parameter_check.py"
)
set_tests_properties(
  mind_sim_mod_global_parameter_check
  PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
)
add_test(
  NAME mind_sim_arbor_hhplus_check
  COMMAND "${Python_EXECUTABLE}" -S "${CMAKE_CURRENT_SOURCE_DIR}/tests/micro/arbor_hhplus_check.py"
)
set_tests_properties(
  mind_sim_arbor_hhplus_check
  PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
)
add_test(
  NAME mind_sim_arbor_hhplus_recurrent_check
  COMMAND "${Python_EXECUTABLE}" -S "${CMAKE_CURRENT_SOURCE_DIR}/tests/micro/arbor_hhplus_recurrent_check.py"
)
set_tests_properties(
  mind_sim_arbor_hhplus_recurrent_check
  PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
)
add_test(
  NAME mind_sim_external_spike_input_reuse_check
  COMMAND "${Python_EXECUTABLE}" -S "${CMAKE_CURRENT_SOURCE_DIR}/tests/micro/external_spike_input_reuse_check.py"
)
set_tests_properties(
  mind_sim_external_spike_input_reuse_check
  PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
)
add_test(
  NAME mind_sim_coupling_check
  COMMAND "${Python_EXECUTABLE}" -S "${CMAKE_CURRENT_SOURCE_DIR}/tests/macro/coupling_check.py"
)
set_tests_properties(
  mind_sim_coupling_check
  PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
)
add_custom_target(
  mind_sim_check
  COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
  DEPENDS _native
  VERBATIM
)
