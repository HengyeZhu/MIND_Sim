cmake_minimum_required(VERSION 3.20)
project(mind_sim LANGUAGES CXX)

option(MIND_SIM_ENABLE_GPU "Enable CoreNEURON OpenACC GPU backend for the micro runtime" OFF)
set(MIND_SIM_GPU_ARCHS "lineinfo" CACHE STRING "NVHPC -gpu option list for MIND_Sim GPU builds")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "Release C++ flags" FORCE)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC|PGI")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG" CACHE STRING "Release C++ flags" FORCE)
endif()

find_package(OpenMP REQUIRED)
find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

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

set(MIND_SIM_MODCC_BACKEND "cpu")
set(MIND_SIM_GPU_COMPILE_OPTIONS)
set(MIND_SIM_GPU_LINK_OPTIONS)
set(MIND_SIM_MODCC_GPU_DEFINE_ARGS)
set(MIND_SIM_GPU_RUNTIME_LIBRARY)

if(MIND_SIM_ENABLE_GPU)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "NVHPC|PGI")
    message(FATAL_ERROR "MIND_SIM_ENABLE_GPU requires the NVIDIA HPC SDK C++ compiler (nvc++).")
  endif()
  set(MIND_SIM_MODCC_BACKEND "openacc")
  set(MIND_SIM_RUNTIME_FP_FLAGS "-O2")
  set(MIND_SIM_MODCC_FP_FLAGS "-O2" "-Mautoinline")
  set(MIND_SIM_GPU_COMPILE_OPTIONS "-cuda" "-gpu=${MIND_SIM_GPU_ARCHS}" "-acc")
  set(MIND_SIM_GPU_LINK_OPTIONS "-cuda" "-gpu=${MIND_SIM_GPU_ARCHS}" "-acc")
  list(APPEND MIND_SIM_MODCC_FP_FLAGS "-DR123_USE_INTRIN_H=0" "-DEIGEN_DONT_VECTORIZE=1")
  list(APPEND MIND_SIM_MODCC_GPU_DEFINE_ARGS
    --define CORENEURON_ENABLE_GPU
    --define R123_USE_INTRIN_H=0
    --define EIGEN_DONT_VECTORIZE=1
  )
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    set(MIND_SIM_GPU_RUNTIME_LIBRARY CUDA::cudart)
  else()
    get_filename_component(MIND_SIM_NVHPC_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(MIND_SIM_NVHPC_COMPILER_DIR "${MIND_SIM_NVHPC_BIN_DIR}" DIRECTORY)
    get_filename_component(MIND_SIM_NVHPC_VERSION_DIR "${MIND_SIM_NVHPC_COMPILER_DIR}" DIRECTORY)
    file(GLOB MIND_SIM_CUDART_CANDIDATES
      "${MIND_SIM_NVHPC_VERSION_DIR}/cuda/*/targets/x86_64-linux/lib/libcudart.so"
    )
    list(SORT MIND_SIM_CUDART_CANDIDATES)
    list(LENGTH MIND_SIM_CUDART_CANDIDATES MIND_SIM_CUDART_CANDIDATE_COUNT)
    if(MIND_SIM_CUDART_CANDIDATE_COUNT GREATER 0)
      list(GET MIND_SIM_CUDART_CANDIDATES -1 MIND_SIM_GPU_RUNTIME_LIBRARY)
    else()
      message(FATAL_ERROR "Could not locate libcudart for the MIND_Sim GPU build.")
    endif()
  endif()
else()
  set(MIND_SIM_RUNTIME_FP_FLAGS
    "-O3" "-march=native" "-mtune=native" "-fno-math-errno"
    "-ffp-contract=off" "-fno-semantic-interposition" "-funroll-loops"
  )
  set(MIND_SIM_MODCC_FP_FLAGS ${MIND_SIM_RUNTIME_FP_FLAGS})
endif()

list(APPEND MIND_SIM_MODCC_FP_FLAGS ${MIND_SIM_GPU_COMPILE_OPTIONS})
string(REPLACE ";" " " MIND_SIM_MODCC_FP_FLAGS_TEXT "${MIND_SIM_MODCC_FP_FLAGS}")
string(STRIP "${OpenMP_CXX_FLAGS} ${MIND_SIM_MODCC_FP_FLAGS_TEXT}" MIND_SIM_MODCC_CXX_FLAGS)

set(MIND_SIM_MODCC_LINK_OPTIONS ${OpenMP_CXX_FLAGS})
list(APPEND MIND_SIM_MODCC_LINK_OPTIONS ${MIND_SIM_GPU_LINK_OPTIONS})
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

set(MIND_SIM_MODCC "${PROJECT_SOURCE_DIR}/src/micro/mod_compiler/mind_modcc.py")
set(MIND_SIM_CORENEURON_MOD_DEFINES
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
  NRN_PRCELLSTATE=0
)

function(mind_sim_add_mod_codegen_target)
  set(one_value_args
    TARGET
    OUTPUT_DIR
    MODL_REG_NAME
    OUT_SOURCES
    COMMENT
  )
  set(multi_value_args
    MODFILES
    INCLUDES
    DEFINES
    DEPENDS
  )
  cmake_parse_arguments(MODGEN "" "${one_value_args}" "${multi_value_args}" ${ARGN})

  foreach(required_arg TARGET OUTPUT_DIR MODL_REG_NAME OUT_SOURCES)
    if(NOT MODGEN_${required_arg})
      message(FATAL_ERROR "mind_sim_add_mod_codegen_target requires ${required_arg}")
    endif()
  endforeach()
  if(NOT MODGEN_MODFILES)
    message(FATAL_ERROR "mind_sim_add_mod_codegen_target requires at least one MODFILES entry")
  endif()
  if(NOT MODGEN_COMMENT)
    set(MODGEN_COMMENT "Generating MOD mechanisms for ${MODGEN_TARGET}")
  endif()

  set(generated_dir "${MODGEN_OUTPUT_DIR}/core")
  set(generated_sources)
  foreach(modfile IN LISTS MODGEN_MODFILES)
    get_filename_component(mod_name "${modfile}" NAME_WE)
    list(APPEND generated_sources "${generated_dir}/${mod_name}.cpp")
  endforeach()
  set(modl_reg_source "${MODGEN_MODL_REG_NAME}.cpp")
  if(MODGEN_MODL_REG_NAME STREQUAL "modl_reg")
    set(modl_reg_source "mind_modl_reg.cpp")
  endif()
  list(APPEND generated_sources "${generated_dir}/${modl_reg_source}")

  set(include_args)
  foreach(include_dir IN LISTS MODGEN_INCLUDES)
    list(APPEND include_args --include "${include_dir}")
  endforeach()

  set(define_args)
  foreach(define IN LISTS MODGEN_DEFINES)
    list(APPEND define_args --define "${define}")
  endforeach()

  file(MAKE_DIRECTORY "${MODGEN_OUTPUT_DIR}")
  set(stamp "${MODGEN_OUTPUT_DIR}/${MODGEN_TARGET}.stamp")
  set(signature "${MODGEN_OUTPUT_DIR}/${MODGEN_TARGET}.signature")
  file(WRITE "${signature}"
    "backend=${MIND_SIM_MODCC_BACKEND}\n"
    "cxx=${CMAKE_CXX_COMPILER}\n"
    "cxx_flags=${MIND_SIM_MODCC_CXX_FLAGS}\n"
    "link_flags=${MIND_SIM_MODCC_LINK_FLAGS}\n"
    "gpu_defines=${MIND_SIM_MODCC_GPU_DEFINE_ARGS}\n"
    "defines=${MODGEN_DEFINES}\n"
  )
  add_custom_command(
    OUTPUT
      "${stamp}"
      ${generated_sources}
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${MODGEN_OUTPUT_DIR}"
    COMMAND "${Python_EXECUTABLE}" "${MIND_SIM_MODCC}"
            --nmodl "${MIND_SIM_NMODL}"
            --cxx "${CMAKE_CXX_COMPILER}"
            --output "${MODGEN_OUTPUT_DIR}"
            --backend "${MIND_SIM_MODCC_BACKEND}"
            --generate-only
            --modl-reg-name "${MODGEN_MODL_REG_NAME}"
            "--cxx-flag=${MIND_SIM_MODCC_CXX_FLAGS}"
            "--link-flag=${MIND_SIM_MODCC_LINK_FLAGS}"
            ${include_args}
            ${define_args}
            ${MIND_SIM_MODCC_GPU_DEFINE_ARGS}
            ${MODGEN_MODFILES}
    COMMAND "${CMAKE_COMMAND}" -E touch "${stamp}"
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
    DEPENDS mind_sim_nmodl
            "${MIND_SIM_NMODL}"
            "${MIND_SIM_MODCC}"
            "${signature}"
            ${MODGEN_MODFILES}
            ${MODGEN_DEPENDS}
    COMMENT "${MODGEN_COMMENT}"
    COMMAND_EXPAND_LISTS
    VERBATIM
  )
  add_custom_target("${MODGEN_TARGET}" DEPENDS "${stamp}")
  set_source_files_properties(${generated_sources} PROPERTIES GENERATED TRUE)
  set("${MODGEN_OUT_SOURCES}" ${generated_sources} PARENT_SCOPE)
endfunction()

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
  exp2syn
  expsyn
  hh
  netstim
  passive
  pattern
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
  MODL_REG_NAME modl_reg
  OUT_SOURCES MIND_SIM_DEFAULT_MECHANISM_GENERATED_SOURCES
  MODFILES ${MIND_SIM_DEFAULT_MECHANISM_MODFILES}
  INCLUDES "${PROJECT_SOURCE_DIR}/src/micro/sim"
  DEFINES ${MIND_SIM_CORENEURON_MOD_DEFINES}
  COMMENT "Building default MOD mechanisms"
)

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

set(MIND_SIM_MACRO_SOURCES
  src/macro/frontend/connectivity.cpp
  src/macro/frontend/local_connectivity.cpp
  src/macro/frontend/node_to_roi_map.cpp
  src/macro/frontend/network.cpp
  src/macro/frontend/network_field.cpp
  src/macro/frontend/network_micro.cpp
  src/macro/sim/model.cpp
  src/macro/sim/runtime_core.cpp
  src/macro/sim/runtime.cpp
)

set(MIND_SIM_MICRO_SOURCES
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
  ${MIND_SIM_MACRO_SOURCES}
  ${MIND_SIM_MICRO_SOURCES}
  ${MIND_SIM_MIND_MOD_SOURCES}
  ${MIND_SIM_IO_SOURCES}
  ${MIND_SIM_UTILS_SOURCES}
  ${MIND_SIM_COSIM_SOURCES}
)

set(MIND_SIM_PYTHON_BINDING_SOURCES
  src/python_api/bindings/bindings_io.cpp
  src/python_api/bindings/bindings_macro.cpp
  src/python_api/bindings/bindings_micro.cpp
  src/python_api/bindings/bindings_rules.cpp
  src/python_api/bindings/module.cpp
)

set(MIND_SIM_BUILD_PYTHON_API_DIR "${PROJECT_BINARY_DIR}/mind_sim")
set(MIND_SIM_PYTHON_API_SOURCES
  src/python_api/mind_sim/__init__.py
  src/python_api/mind_sim/_codegen.py
  src/python_api/mind_sim/_io.py
  src/python_api/mind_sim/cli.py
  src/python_api/mind_sim/bridge/__init__.py
  src/python_api/mind_sim/cosim/__init__.py
  src/python_api/mind_sim/macro/__init__.py
  src/python_api/mind_sim/micro/__init__.py
)
set(MIND_SIM_PYTHON_API_STAMP "${PROJECT_BINARY_DIR}/mind_sim_python_api.stamp")
add_custom_command(
  OUTPUT "${MIND_SIM_PYTHON_API_STAMP}"
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${MIND_SIM_BUILD_PYTHON_API_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
          "${PROJECT_SOURCE_DIR}/src/python_api/mind_sim"
          "${MIND_SIM_BUILD_PYTHON_API_DIR}"
  COMMAND "${CMAKE_COMMAND}" -E rm -rf
          "${MIND_SIM_BUILD_PYTHON_API_DIR}/default_mechanisms"
          "${MIND_SIM_BUILD_PYTHON_API_DIR}/__pycache__"
          "${MIND_SIM_BUILD_PYTHON_API_DIR}/bridge/__pycache__"
          "${MIND_SIM_BUILD_PYTHON_API_DIR}/cosim/__pycache__"
          "${MIND_SIM_BUILD_PYTHON_API_DIR}/macro/__pycache__"
          "${MIND_SIM_BUILD_PYTHON_API_DIR}/micro/__pycache__"
  COMMAND "${CMAKE_COMMAND}" -E touch "${MIND_SIM_PYTHON_API_STAMP}"
  DEPENDS mind_sim_default_mechanisms
          ${MIND_SIM_PYTHON_API_SOURCES}
  COMMENT "Staging Python API"
  VERBATIM
)
add_custom_target(mind_sim_python_api DEPENDS "${MIND_SIM_PYTHON_API_STAMP}")

function(mind_sim_apply_native_includes target)
  target_include_directories("${target}" PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/src/micro/frontend"
    "${MIND_SIM_HDF5_INCLUDE_DIR}"
  )
  target_include_directories("${target}" SYSTEM PRIVATE
    "${PROJECT_SOURCE_DIR}/src/micro/sim"
  )
endfunction()

function(mind_sim_apply_coreneuron_definitions target)
  target_compile_definitions("${target}" PRIVATE
    ${MIND_SIM_CORENEURON_MOD_DEFINES}
  )
endfunction()

function(mind_sim_apply_mind_runtime_definitions target)
  target_compile_definitions("${target}" PRIVATE
    MIND_SIM_MODCC="${MIND_SIM_MODCC}"
    MIND_SIM_NMODL_EXECUTABLE="${MIND_SIM_NMODL}"
    MIND_SIM_NMODL_SOURCE_DIR="${MIND_SIM_NMODL_SOURCE_DIR}"
    MIND_SIM_PYTHON_EXECUTABLE="${Python_EXECUTABLE}"
    MIND_SIM_CXX_COMPILER="${CMAKE_CXX_COMPILER}"
    MIND_SIM_MODCC_BACKEND="${MIND_SIM_MODCC_BACKEND}"
    MIND_SIM_MODCC_CXX_FLAGS="${MIND_SIM_MODCC_CXX_FLAGS}"
    MIND_SIM_MODCC_LINK_FLAGS="${MIND_SIM_MODCC_LINK_FLAGS}"
    MIND_SIM_MECHANISM_INCLUDE_DIR="${PROJECT_SOURCE_DIR}/src/micro/sim"
  )
endfunction()

function(mind_sim_apply_gpu_definitions target)
  if(MIND_SIM_ENABLE_GPU)
    target_compile_definitions("${target}" PRIVATE
      MIND_SIM_ENABLE_GPU=1
      CORENEURON_ENABLE_GPU
      R123_USE_INTRIN_H=0
      EIGEN_DONT_VECTORIZE=1
    )
  endif()
endfunction()

function(mind_sim_apply_runtime_compile_options target)
  target_link_libraries("${target}" PRIVATE OpenMP::OpenMP_CXX)

  if(MIND_SIM_GPU_COMPILE_OPTIONS)
    target_compile_options("${target}" PRIVATE ${MIND_SIM_GPU_COMPILE_OPTIONS})
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options("${target}" PRIVATE ${MIND_SIM_RUNTIME_FP_FLAGS})
  elseif(MIND_SIM_ENABLE_GPU)
    target_compile_options("${target}" PRIVATE ${MIND_SIM_RUNTIME_FP_FLAGS})
  endif()
endfunction()

function(mind_sim_suppress_vendor_diagnostics target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC|PGI")
    target_compile_options("${target}" PRIVATE
      "--diag_suppress=declared_but_not_referenced"
      "--diag_suppress=set_but_not_used"
      "--diag_suppress=invalid_parallelism_order"
    )
  endif()
endfunction()

function(mind_sim_suppress_binding_diagnostics target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC|PGI")
    target_compile_options("${target}" PRIVATE
      "--diag_suppress=inline_gnu_noinline_conflict"
    )
  endif()
endfunction()

add_library(mind_sim_coreneuron OBJECT ${MIND_SIM_CORENEURON_SOURCES})
mind_sim_apply_native_includes(mind_sim_coreneuron)
mind_sim_apply_coreneuron_definitions(mind_sim_coreneuron)
target_compile_definitions(mind_sim_coreneuron PRIVATE ADDITIONAL_MECHS)
mind_sim_apply_gpu_definitions(mind_sim_coreneuron)
mind_sim_apply_runtime_compile_options(mind_sim_coreneuron)
mind_sim_suppress_vendor_diagnostics(mind_sim_coreneuron)

add_library(mind_sim_default_mechanism_objects OBJECT ${MIND_SIM_DEFAULT_MECHANISM_GENERATED_SOURCES})
add_dependencies(mind_sim_default_mechanism_objects mind_sim_default_mechanisms)
mind_sim_apply_native_includes(mind_sim_default_mechanism_objects)
mind_sim_apply_coreneuron_definitions(mind_sim_default_mechanism_objects)
mind_sim_apply_gpu_definitions(mind_sim_default_mechanism_objects)
mind_sim_apply_runtime_compile_options(mind_sim_default_mechanism_objects)
mind_sim_suppress_vendor_diagnostics(mind_sim_default_mechanism_objects)

add_library(mind_sim_native_core OBJECT ${MIND_SIM_NATIVE_CORE_SOURCES})
add_dependencies(mind_sim_native_core mind_sim_default_mechanisms)
mind_sim_apply_native_includes(mind_sim_native_core)
mind_sim_apply_coreneuron_definitions(mind_sim_native_core)
mind_sim_apply_mind_runtime_definitions(mind_sim_native_core)
mind_sim_apply_gpu_definitions(mind_sim_native_core)
mind_sim_apply_runtime_compile_options(mind_sim_native_core)

set(MIND_SIM_NANOBIND_OPTIONS
  STABLE_ABI
  NOMINSIZE
  NB_STATIC
)
if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC|PGI")
  list(APPEND MIND_SIM_NANOBIND_OPTIONS PROTECT_STACK)
endif()

nanobind_add_module(
  _native
  ${MIND_SIM_NANOBIND_OPTIONS}
  ${MIND_SIM_PYTHON_BINDING_SOURCES}
  $<TARGET_OBJECTS:mind_sim_coreneuron>
  $<TARGET_OBJECTS:mind_sim_default_mechanism_objects>
  $<TARGET_OBJECTS:mind_sim_native_core>
)
add_dependencies(_native
  mind_sim_coreneuron
  mind_sim_default_mechanism_objects
  mind_sim_native_core
  mind_sim_python_api
)
mind_sim_apply_native_includes(_native)
mind_sim_apply_coreneuron_definitions(_native)
mind_sim_apply_gpu_definitions(_native)
mind_sim_suppress_binding_diagnostics(_native)

if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC|PGI" AND TARGET nanobind-static)
  target_compile_definitions(nanobind-static PRIVATE __GXX_ABI_VERSION=1002)
  get_target_property(MIND_SIM_NATIVE_COMPILE_OPTIONS _native COMPILE_OPTIONS)
  if(MIND_SIM_NATIVE_COMPILE_OPTIONS)
    list(REMOVE_ITEM MIND_SIM_NATIVE_COMPILE_OPTIONS "-fno-stack-protector")
    set_target_properties(_native PROPERTIES COMPILE_OPTIONS "${MIND_SIM_NATIVE_COMPILE_OPTIONS}")
  endif()
endif()

set_target_properties(_native PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${MIND_SIM_BUILD_PYTHON_API_DIR}"
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN OFF
)

target_link_libraries(_native PRIVATE
  ${CMAKE_DL_LIBS}
  OpenMP::OpenMP_CXX
  ${MIND_SIM_HDF5_LIBRARY}
  ${MIND_SIM_GPU_RUNTIME_LIBRARY}
)
if(MIND_SIM_GPU_LINK_OPTIONS)
  target_link_options(_native PRIVATE ${MIND_SIM_GPU_LINK_OPTIONS})
endif()

enable_testing()

function(mind_sim_add_python_test name script)
  add_test(
    NAME "${name}"
    COMMAND "${Python_EXECUTABLE}" -S "${PROJECT_SOURCE_DIR}/${script}"
  )
  set_tests_properties(
    "${name}"
    PROPERTIES
      ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR};PYTHONDONTWRITEBYTECODE=1;MIND_SIM_CODEGEN_CACHE=${PROJECT_BINARY_DIR}/mind_sim_codegen_cache"
  )
endfunction()

mind_sim_add_python_test(mind_sim_ion_channel_check tests/micro/ion_channel_check.py)
mind_sim_add_python_test(mind_sim_custom_ion_check tests/micro/custom_ion_check.py)
mind_sim_add_python_test(mind_sim_ion_register_check tests/micro/ion_register_check.py)
mind_sim_add_python_test(mind_sim_mod_global_parameter_check tests/micro/mod_global_parameter_check.py)
mind_sim_add_python_test(mind_sim_arbor_hhplus_check tests/micro/arbor_hhplus_check.py)
mind_sim_add_python_test(mind_sim_arbor_hhplus_recurrent_check tests/micro/arbor_hhplus_recurrent_check.py)
mind_sim_add_python_test(mind_sim_external_spike_input_reuse_check tests/micro/external_spike_input_reuse_check.py)
mind_sim_add_python_test(mind_sim_coupling_check tests/macro/coupling_check.py)
mind_sim_add_python_test(mind_sim_connectivity_loader_check tests/macro/connectivity_loader_check.py)
mind_sim_add_python_test(mind_sim_explicit_rule_fields_check tests/macro/explicit_rule_fields_check.py)
mind_sim_add_python_test(mind_sim_node_to_roi_map_file_check tests/macro/node_to_roi_map_file_check.py)
mind_sim_add_python_test(mind_sim_neural_field_check tests/macro/neural_field_check.py)
mind_sim_add_python_test(mind_sim_neural_field_global_coupling_check tests/macro/neural_field_global_coupling_check.py)

add_custom_target(
  mind_sim_check
  COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
  DEPENDS _native
  VERBATIM
)
