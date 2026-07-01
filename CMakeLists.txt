cmake_minimum_required(VERSION 3.20)

# Vendored NEURON/NMODL still defines a target named "codegen".
if(POLICY CMP0171)
  cmake_policy(SET CMP0171 OLD)
endif()

project(mind_sim LANGUAGES CXX)

if(NOT DEFINED MIND_SIM_CXX_COMPILER_FOR_MODS)
  set(MIND_SIM_CXX_COMPILER_FOR_MODS "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "C++ compiler used by mind-nrnivmodl")
endif()

option(MIND_SIM_ENABLE_GPU "Enable CoreNEURON OpenACC GPU backend for the micro runtime" OFF)
option(MIND_SIM_BINARY_DIST_BUILD "Build portable binary distribution artifacts" OFF)
option(MIND_SIM_ENABLE_NATIVE_OPT "Enable compiler flags tuned for the build host CPU" OFF)
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

set(MIND_SIM_BINARY_DIST_COMPILE_OPTIONS)
set(MIND_SIM_BINARY_DIST_LINK_OPTIONS)
if(MIND_SIM_BINARY_DIST_BUILD AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  foreach(MIND_SIM_PREFIX_MAP_DIR IN ITEMS "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}")
    list(APPEND MIND_SIM_BINARY_DIST_COMPILE_OPTIONS
      "-ffile-prefix-map=${MIND_SIM_PREFIX_MAP_DIR}=."
      "-fmacro-prefix-map=${MIND_SIM_PREFIX_MAP_DIR}=."
    )
  endforeach()
  if(UNIX AND NOT APPLE)
    list(APPEND MIND_SIM_BINARY_DIST_LINK_OPTIONS "-Wl,--strip-debug")
  endif()
endif()
if(MIND_SIM_BINARY_DIST_COMPILE_OPTIONS)
  add_compile_options(${MIND_SIM_BINARY_DIST_COMPILE_OPTIONS})
endif()
if(MIND_SIM_BINARY_DIST_LINK_OPTIONS)
  add_link_options(${MIND_SIM_BINARY_DIST_LINK_OPTIONS})
endif()
if(MIND_SIM_BINARY_DIST_BUILD)
  set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
  set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
  set(CMAKE_INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE)
endif()

find_package(OpenMP REQUIRED)
if(MIND_SIM_BINARY_DIST_BUILD)
  find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
else()
  find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)
endif()
find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)
include(GNUInstallDirs)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import sys; print(sys.prefix)"
  OUTPUT_VARIABLE Python_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_path('platlib'))"
  OUTPUT_VARIABLE Python_PLATLIB
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(MIND_SIM_DEFAULT_INSTALL_PYTHON_PREFIX "${Python_PLATLIB}")
if(IS_ABSOLUTE "${MIND_SIM_DEFAULT_INSTALL_PYTHON_PREFIX}")
  cmake_path(RELATIVE_PATH MIND_SIM_DEFAULT_INSTALL_PYTHON_PREFIX BASE_DIRECTORY "${Python_PREFIX}")
endif()
set(MIND_SIM_DEFAULT_INSTALL_PYTHON_PREFIX "${MIND_SIM_DEFAULT_INSTALL_PYTHON_PREFIX}/mind_sim")

set(MIND_SIM_INSTALL_PYTHON_PREFIX
  "${MIND_SIM_DEFAULT_INSTALL_PYTHON_PREFIX}"
  CACHE STRING
  "Path where MIND_Sim Python components will be installed, relative to CMAKE_INSTALL_PREFIX"
)
if(SKBUILD)
  set(MIND_SIM_INSTALL_PYTHON_PREFIX "mind_sim" CACHE STRING "" FORCE)
  set(MIND_SIM_INSTALL_DATA_PREFIX "mind_sim/.data")
else()
  set(MIND_SIM_INSTALL_DATA_PREFIX "")
endif()
if(IS_ABSOLUTE "${MIND_SIM_INSTALL_PYTHON_PREFIX}")
  cmake_path(RELATIVE_PATH MIND_SIM_INSTALL_PYTHON_PREFIX BASE_DIRECTORY "${CMAKE_INSTALL_PREFIX}")
endif()
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(nanobind CONFIG REQUIRED)

set(MIND_SIM_NMODL_BACKEND "cpu")
set(MIND_SIM_GPU_COMPILE_OPTIONS)
set(MIND_SIM_GPU_LINK_OPTIONS)
set(MIND_SIM_GPU_RUNTIME_LIBRARY)

if(MIND_SIM_ENABLE_GPU)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "NVHPC|PGI")
    message(FATAL_ERROR "MIND_SIM_ENABLE_GPU requires the NVIDIA HPC SDK C++ compiler (nvc++).")
  endif()
  set(MIND_SIM_NMODL_BACKEND "openacc")
  set(MIND_SIM_RUNTIME_FP_FLAGS "-O2")
  set(MIND_SIM_MOD_FP_FLAGS "-O2" "-Mautoinline")
  set(MIND_SIM_GPU_COMPILE_OPTIONS "-cuda" "-gpu=${MIND_SIM_GPU_ARCHS}" "-acc")
  set(MIND_SIM_GPU_LINK_OPTIONS "-cuda" "-gpu=${MIND_SIM_GPU_ARCHS}" "-acc")
  list(APPEND MIND_SIM_MOD_FP_FLAGS "-DR123_USE_INTRIN_H=0" "-DEIGEN_DONT_VECTORIZE=1")
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    set(MIND_SIM_GPU_RUNTIME_LIBRARY CUDA::cudart)
  else()
    get_filename_component(MIND_SIM_GPU_COMPILER_BIN_DIR "${MIND_SIM_CXX_COMPILER_FOR_MODS}" DIRECTORY)
    get_filename_component(MIND_SIM_GPU_COMPILER_DIR "${MIND_SIM_GPU_COMPILER_BIN_DIR}" DIRECTORY)
    get_filename_component(MIND_SIM_GPU_SDK_VERSION_DIR "${MIND_SIM_GPU_COMPILER_DIR}" DIRECTORY)
    file(GLOB MIND_SIM_CUDART_CANDIDATES
      "${MIND_SIM_GPU_SDK_VERSION_DIR}/cuda/*/targets/*/lib/libcudart.so"
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
  if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC|PGI")
    if(MIND_SIM_BINARY_DIST_BUILD)
      set(MIND_SIM_RUNTIME_FP_FLAGS "-O2")
    else()
      set(MIND_SIM_RUNTIME_FP_FLAGS
        "-O3" "-Mautoinline" "-Munroll"
      )
      if(MIND_SIM_ENABLE_NATIVE_OPT)
        list(APPEND MIND_SIM_RUNTIME_FP_FLAGS "-tp=native")
      endif()
    endif()
    set(MIND_SIM_MOD_FP_FLAGS ${MIND_SIM_RUNTIME_FP_FLAGS} "-DR123_USE_INTRIN_H=0")
  else()
    if(MIND_SIM_BINARY_DIST_BUILD)
      set(MIND_SIM_RUNTIME_FP_FLAGS "-O2")
    else()
      set(MIND_SIM_RUNTIME_FP_FLAGS
        "-O3" "-fno-math-errno" "-ffp-contract=off"
        "-fno-semantic-interposition" "-funroll-loops"
      )
      if(MIND_SIM_ENABLE_NATIVE_OPT)
        list(APPEND MIND_SIM_RUNTIME_FP_FLAGS "-march=native" "-mtune=native")
      endif()
    endif()
    set(MIND_SIM_MOD_FP_FLAGS ${MIND_SIM_RUNTIME_FP_FLAGS})
  endif()
endif()

list(APPEND MIND_SIM_MOD_FP_FLAGS ${MIND_SIM_GPU_COMPILE_OPTIONS})
string(REPLACE ";" " " MIND_SIM_MOD_FP_FLAGS_TEXT "${MIND_SIM_MOD_FP_FLAGS}")
if(MIND_SIM_BINARY_DIST_BUILD)
  string(STRIP "${MIND_SIM_MOD_FP_FLAGS_TEXT}" MIND_SIM_MOD_CXX_FLAGS)
else()
  string(STRIP "${OpenMP_CXX_FLAGS} ${MIND_SIM_MOD_FP_FLAGS_TEXT}" MIND_SIM_MOD_CXX_FLAGS)
endif()

set(MIND_SIM_MOD_LINK_OPTIONS)
if(NOT MIND_SIM_BINARY_DIST_BUILD)
  list(APPEND MIND_SIM_MOD_LINK_OPTIONS ${OpenMP_CXX_FLAGS})
endif()
list(APPEND MIND_SIM_MOD_LINK_OPTIONS ${MIND_SIM_GPU_LINK_OPTIONS})
if(OpenMP_CXX_LIBRARIES AND NOT MIND_SIM_BINARY_DIST_BUILD)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND MIND_SIM_MOD_LINK_OPTIONS "-Wl,--no-as-needed")
  endif()
  list(APPEND MIND_SIM_MOD_LINK_OPTIONS ${OpenMP_CXX_LIBRARIES})
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND MIND_SIM_MOD_LINK_OPTIONS "-Wl,--as-needed")
  endif()
endif()
string(REPLACE ";" " " MIND_SIM_MOD_LINK_FLAGS "${MIND_SIM_MOD_LINK_OPTIONS}")

if(MIND_SIM_BINARY_DIST_BUILD)
  set(MIND_SIM_EMBEDDED_CXX_COMPILER "c++")
  set(MIND_SIM_EMBEDDED_SOURCE_DIR ".")
  set(MIND_SIM_EMBEDDED_MECHANISM_INCLUDE_DIR ".")
else()
  set(MIND_SIM_EMBEDDED_CXX_COMPILER "${MIND_SIM_CXX_COMPILER_FOR_MODS}")
  set(MIND_SIM_EMBEDDED_SOURCE_DIR "${PROJECT_SOURCE_DIR}")
  set(MIND_SIM_EMBEDDED_MECHANISM_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/src/micro/sim")
endif()

set(MIND_SIM_NRN_SOURCE_DIR "${PROJECT_SOURCE_DIR}/src/mod/neuron")
if(NOT EXISTS "${MIND_SIM_NRN_SOURCE_DIR}/src/nmodl/CMakeLists.txt")
  message(FATAL_ERROR
    "MIND_Sim builds MIND-aware NMODL from the vendored NEURON/NMODL source "
    "under src/mod/neuron, but src/mod/neuron/src/nmodl/CMakeLists.txt is missing."
  )
endif()
set(MIND_SIM_EXTERNAL_SOURCE_DIR "${PROJECT_SOURCE_DIR}/external")
foreach(MIND_SIM_EXTERNAL_REQUIRED_PATH IN ITEMS
  CLI11/include/CLI/CLI.hpp
  Random123/include/Random123/philox.h
  eigen/Eigen/Core
  fmt/include/fmt/format.h
  json/include/nlohmann/json.hpp
  pybind11/include/pybind11/pybind11.h
  spdlog/include/spdlog/spdlog.h
)
  if(NOT EXISTS "${MIND_SIM_EXTERNAL_SOURCE_DIR}/${MIND_SIM_EXTERNAL_REQUIRED_PATH}")
    message(FATAL_ERROR
      "Missing external/${MIND_SIM_EXTERNAL_REQUIRED_PATH}. "
      "MIND_Sim keeps third-party dependencies in Git submodules under external/. "
      "Run `git submodule update --init --recursive` from the repository root. "
      "If this tree came from GitHub's auto-generated source archive, clone the "
      "repository instead because that archive does not include submodule contents."
    )
  endif()
endforeach()

foreach(module IN ITEMS jinja2 yaml sympy)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import ${module}"
    RESULT_VARIABLE MIND_SIM_NMODL_MODULE_STATUS
  )
  if(NOT MIND_SIM_NMODL_MODULE_STATUS EQUAL 0)
    message(FATAL_ERROR "Python module ${module} is required to build MIND-aware NMODL.")
  endif()
endforeach()

function(mind_sim_add_header_only_target target include_dir)
  if(NOT TARGET "${target}")
    add_library("${target}" INTERFACE IMPORTED GLOBAL)
    set_target_properties("${target}" PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${include_dir}"
    )
  endif()
endfunction()

mind_sim_add_header_only_target(CLI11::CLI11 "${MIND_SIM_EXTERNAL_SOURCE_DIR}/CLI11/include")
mind_sim_add_header_only_target(nlohmann_json::nlohmann_json "${MIND_SIM_EXTERNAL_SOURCE_DIR}/json/include")

if(NOT TARGET fmt::fmt)
  add_library(mind_sim_nmodl_fmt INTERFACE)
  target_include_directories(mind_sim_nmodl_fmt INTERFACE "${MIND_SIM_EXTERNAL_SOURCE_DIR}/fmt/include")
  target_compile_definitions(mind_sim_nmodl_fmt INTERFACE FMT_HEADER_ONLY)
  add_library(fmt::fmt ALIAS mind_sim_nmodl_fmt)
endif()

if(NOT TARGET spdlog::spdlog)
  add_library(mind_sim_nmodl_spdlog INTERFACE)
  target_include_directories(mind_sim_nmodl_spdlog INTERFACE "${MIND_SIM_EXTERNAL_SOURCE_DIR}/spdlog/include")
  target_compile_definitions(mind_sim_nmodl_spdlog INTERFACE SPDLOG_FMT_EXTERNAL)
  target_link_libraries(mind_sim_nmodl_spdlog INTERFACE fmt::fmt)
  add_library(spdlog::spdlog ALIAS mind_sim_nmodl_spdlog)
endif()

if(NOT TARGET pybind11::module)
  add_library(mind_sim_nmodl_pybind11_module INTERFACE)
  target_include_directories(mind_sim_nmodl_pybind11_module INTERFACE
    "${MIND_SIM_EXTERNAL_SOURCE_DIR}/pybind11/include"
    ${Python_INCLUDE_DIRS}
  )
  add_library(pybind11::module ALIAS mind_sim_nmodl_pybind11_module)
endif()
if(NOT TARGET pybind11::embed)
  add_library(mind_sim_nmodl_pybind11_embed INTERFACE)
  target_include_directories(mind_sim_nmodl_pybind11_embed INTERFACE
    "${MIND_SIM_EXTERNAL_SOURCE_DIR}/pybind11/include"
    ${Python_INCLUDE_DIRS}
  )
  add_library(pybind11::embed ALIAS mind_sim_nmodl_pybind11_embed)
endif()

function(cpp_cc_configure_sanitizers)
endfunction()

function(cpp_cc_build_time_copy)
  set(options NO_TARGET)
  set(one_value_args INPUT OUTPUT)
  cmake_parse_arguments(COPY "${options}" "${one_value_args}" "" ${ARGN})
  if(NOT COPY_INPUT OR NOT COPY_OUTPUT)
    message(FATAL_ERROR "cpp_cc_build_time_copy requires INPUT and OUTPUT")
  endif()
  add_custom_command(
    OUTPUT "${COPY_OUTPUT}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${COPY_INPUT}" "${COPY_OUTPUT}"
    DEPENDS "${COPY_INPUT}"
    VERBATIM
  )
  if(NOT COPY_NO_TARGET)
    get_filename_component(copy_target "${COPY_OUTPUT}" NAME_WE)
    add_custom_target("${copy_target}" DEPENDS "${COPY_OUTPUT}")
  endif()
endfunction()

set(PYTHON_EXECUTABLE "${Python_EXECUTABLE}")
set(PYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS})
if(TARGET Python::Python)
  set(PYTHON_LIBRARIES Python::Python)
else()
  set(PYTHON_LIBRARIES ${Python_LIBRARIES})
endif()
set(PYBIND11_INCLUDE_DIR "${MIND_SIM_EXTERNAL_SOURCE_DIR}/pybind11/include")
set(pybind11_INCLUDE_DIRS "${MIND_SIM_EXTERNAL_SOURCE_DIR}/pybind11/include")
set(NMODL_NEURON_SOURCE_DIR "${MIND_SIM_NRN_SOURCE_DIR}")
set(NMODL_ENABLE_PYTHON_BINDINGS OFF CACHE BOOL "Build NMODL Python bindings" FORCE)
set(NMODL_ENABLE_FLEX_BISON_LINES ON CACHE BOOL "Enable line numbers in generated NMODL parser files" FORCE)
set(NMODL_AS_SUBPROJECT ON)
set(MIND_SIM_NMODL_LINK_AGAINST_PYTHON TRUE)
if(MIND_SIM_BINARY_DIST_BUILD)
  set(MIND_SIM_NMODL_LINK_AGAINST_PYTHON FALSE)
endif()
set(NRN_LINK_AGAINST_PYTHON ${MIND_SIM_NMODL_LINK_AGAINST_PYTHON}
  CACHE BOOL "Link MIND-aware NMODL against Python" FORCE)
set(NRN_SANITIZER_COMPILER_FLAGS)
set(NMODL_EXTRA_CXX_FLAGS)
if(MIND_SIM_BINARY_DIST_BUILD)
  set(NMODL_PORTABLE_NRNUNITS_PATHS 1)
else()
  set(NMODL_PORTABLE_NRNUNITS_PATHS 0)
endif()
if(MIND_SIM_INSTALL_DATA_PREFIX)
  set(NRN_INSTALL_DATA_PREFIX "${MIND_SIM_INSTALL_DATA_PREFIX}/")
else()
  set(NRN_INSTALL_DATA_PREFIX "")
endif()
set(NRN_INSTALL_PYTHON_PREFIX "lib/python/neuron/")
set(NMODL_PROJECT_PLATLIB_SOURCE_DIR "${MIND_SIM_NRN_SOURCE_DIR}/src/nmodl")
set(NMODL_PROJECT_PLATLIB_BINARY_DIR "${PROJECT_BINARY_DIR}/src/nmodl")
set(NMODL_PROJECT_PURELIB_SOURCE_DIR "${MIND_SIM_NRN_SOURCE_DIR}/share/lib/python/neuron/nmodl")
set(NMODL_PROJECT_PURELIB_BINARY_DIR "${PROJECT_BINARY_DIR}/lib/python/neuron/nmodl")
if(MIND_SIM_NMODL_LINK_AGAINST_PYTHON)
  set(NMODL_WRAPPER_LIBS pyembed pywrapper ${CMAKE_DL_LIBS})
else()
  set(NMODL_WRAPPER_LIBS pyembed ${CMAKE_DL_LIBS})
endif()

include_directories(
  "${MIND_SIM_EXTERNAL_SOURCE_DIR}/Random123/include"
  "${MIND_SIM_EXTERNAL_SOURCE_DIR}/eigen"
)

set(MIND_SIM_PARENT_CXX_STANDARD "${CMAKE_CXX_STANDARD}")
set(CMAKE_CXX_STANDARD 17)
add_subdirectory("${MIND_SIM_NRN_SOURCE_DIR}/src/nmodl" "${PROJECT_BINARY_DIR}/src/nmodl" EXCLUDE_FROM_ALL)
set(CMAKE_CXX_STANDARD "${MIND_SIM_PARENT_CXX_STANDARD}")
if(MIND_SIM_BINARY_DIST_BUILD)
  set(MIND_SIM_NMODL_BUILD_RPATH "${PROJECT_BINARY_DIR}/lib")
  set(MIND_SIM_NMODL_INSTALL_RPATH "$ORIGIN/../lib")
else()
  set(MIND_SIM_NMODL_BUILD_RPATH ${Python_RUNTIME_LIBRARY_DIRS} ${Python_LIBRARY_DIRS})
  set(MIND_SIM_NMODL_INSTALL_RPATH ${Python_RUNTIME_LIBRARY_DIRS} ${Python_LIBRARY_DIRS})
endif()
if(NOT MIND_SIM_NMODL_LINK_AGAINST_PYTHON AND NOT MIND_SIM_BINARY_DIST_BUILD)
  list(APPEND MIND_SIM_NMODL_BUILD_RPATH "${PROJECT_BINARY_DIR}/lib")
  list(APPEND MIND_SIM_NMODL_INSTALL_RPATH "$ORIGIN/../lib")
endif()
set_target_properties(nmodl PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
  BUILD_RPATH "${MIND_SIM_NMODL_BUILD_RPATH}"
  INSTALL_RPATH "${MIND_SIM_NMODL_INSTALL_RPATH}"
)
if(MIND_SIM_BINARY_DIST_BUILD)
  set_target_properties(nmodl PROPERTIES INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE)
endif()
if(NOT MIND_SIM_NMODL_LINK_AGAINST_PYTHON)
  if(APPLE)
    set(MIND_SIM_UNDEFINED_SYMBOLS_IGNORE_FLAG "-Wl,-undefined,dynamic_lookup")
  elseif(UNIX)
    set(MIND_SIM_UNDEFINED_SYMBOLS_IGNORE_FLAG "-Wl,--unresolved-symbols=ignore-all")
  endif()
  if(MIND_SIM_UNDEFINED_SYMBOLS_IGNORE_FLAG)
    target_link_options(nmodl PRIVATE "${MIND_SIM_UNDEFINED_SYMBOLS_IGNORE_FLAG}")
  endif()
endif()
set(MIND_SIM_NMODL "${PROJECT_BINARY_DIR}/bin/nmodl${CMAKE_EXECUTABLE_SUFFIX}")
if(MIND_SIM_BINARY_DIST_BUILD)
  set(MIND_SIM_EMBEDDED_NMODL_PATH "nmodl${CMAKE_EXECUTABLE_SUFFIX}")
else()
  set(MIND_SIM_EMBEDDED_NMODL_PATH "${MIND_SIM_NMODL}")
endif()
add_custom_target(mind_sim_nmodl DEPENDS nmodl)
get_filename_component(MIND_SIM_NMODL_BIN_DIR "${MIND_SIM_NMODL}" DIRECTORY)
get_filename_component(MIND_SIM_NMODL_HOME "${MIND_SIM_NMODL_BIN_DIR}" DIRECTORY)
if(MIND_SIM_NMODL_BACKEND STREQUAL "cpu")
  set(MIND_SIM_NMODL_BACKEND_ARGS host --c)
else()
  set(MIND_SIM_NMODL_BACKEND_ARGS acc --oacc)
endif()
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
if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC|PGI")
  list(APPEND MIND_SIM_CORENEURON_MOD_DEFINES R123_USE_INTRIN_H=0)
endif()

function(mind_sim_add_mod_compile_target)
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
      message(FATAL_ERROR "mind_sim_add_mod_compile_target requires ${required_arg}")
    endif()
  endforeach()
  if(NOT MODGEN_MODFILES)
    message(FATAL_ERROR "mind_sim_add_mod_compile_target requires at least one MODFILES entry")
  endif()
  if(NOT MODGEN_COMMENT)
    set(MODGEN_COMMENT "Generating MOD mechanisms for ${MODGEN_TARGET}")
  endif()

  set(generated_dir "${MODGEN_OUTPUT_DIR}/core")
  set(scratch_dir "${MODGEN_OUTPUT_DIR}/scratch")
  set(generated_sources)
  set(mod_names)
  foreach(modfile IN LISTS MODGEN_MODFILES)
    get_filename_component(mod_name "${modfile}" NAME_WE)
    list(APPEND mod_names "${mod_name}")
    list(APPEND generated_sources "${generated_dir}/${mod_name}.cpp")
  endforeach()
  string(JOIN "," mod_names_csv ${mod_names})
  set(modl_reg_source "${MODGEN_MODL_REG_NAME}.cpp")
  if(MODGEN_MODL_REG_NAME STREQUAL "modl_reg")
    set(modl_reg_source "mind_nmodl_reg.cpp")
  endif()
  list(APPEND generated_sources "${generated_dir}/${modl_reg_source}")

  file(MAKE_DIRECTORY "${MODGEN_OUTPUT_DIR}")
  set(stamp "${MODGEN_OUTPUT_DIR}/${MODGEN_TARGET}.stamp")
  set(signature "${MODGEN_OUTPUT_DIR}/${MODGEN_TARGET}.signature")
  file(WRITE "${signature}"
    "backend=${MIND_SIM_NMODL_BACKEND}\n"
    "nmodl=${MIND_SIM_NMODL}\n"
    "cxx=${MIND_SIM_CXX_COMPILER_FOR_MODS}\n"
    "cxx_flags=${MIND_SIM_MOD_CXX_FLAGS}\n"
    "link_flags=${MIND_SIM_MOD_LINK_FLAGS}\n"
    "defines=${MODGEN_DEFINES}\n"
  )
  add_custom_command(
    OUTPUT
      "${stamp}"
      ${generated_sources}
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${generated_dir}" "${scratch_dir}"
    COMMAND "${CMAKE_COMMAND}" -E env
            "NMODLHOME=${MIND_SIM_NMODL_HOME}"
            "${MIND_SIM_NMODL}"
            ${MODGEN_MODFILES}
            -o "${generated_dir}"
            --scratch "${scratch_dir}"
            ${MIND_SIM_NMODL_BACKEND_ARGS}
            passes
            --inline
    COMMAND "${CMAKE_COMMAND}"
            "-DREG_CPP=${generated_dir}/${modl_reg_source}"
            "-DFUNCTION_NAME=${MODGEN_MODL_REG_NAME}"
            "-DMOD_NAMES=${mod_names_csv}"
            -P "${PROJECT_SOURCE_DIR}/cmake/write_modl_reg.cmake"
    COMMAND "${CMAKE_COMMAND}" -E touch "${stamp}"
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
    DEPENDS "${signature}"
            "${PROJECT_SOURCE_DIR}/cmake/write_modl_reg.cmake"
            ${MODGEN_MODFILES}
            ${MODGEN_DEPENDS}
    COMMENT "${MODGEN_COMMENT}"
    COMMAND_EXPAND_LISTS
    VERBATIM
  )
  add_custom_target("${MODGEN_TARGET}" DEPENDS "${stamp}")
  add_dependencies("${MODGEN_TARGET}" mind_sim_nmodl)
  set_source_files_properties(${generated_sources} PROPERTIES GENERATED TRUE)
  set("${MODGEN_OUT_SOURCES}" ${generated_sources} PARENT_SCOPE)
endfunction()

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

mind_sim_add_mod_compile_target(
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

set(MIND_SIM_COSIM_TRANSFORM_SOURCES
  src/cosim/transform/interfaces.cpp
)

set(MIND_SIM_MACRO_SOURCES
  src/macro/frontend/connectivity.cpp
  src/macro/frontend/network.cpp
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
  src/micro/sim/recording.cpp
)

set(MIND_SIM_UTILS_SOURCES
  src/utils/dynamic_library.cpp
)

set(MIND_SIM_COSIM_SOURCES
  src/cosim/hybrid_simulator.cpp
)

set(MIND_SIM_NATIVE_CORE_SOURCES
  ${MIND_SIM_COSIM_TRANSFORM_SOURCES}
  ${MIND_SIM_MACRO_SOURCES}
  ${MIND_SIM_MICRO_SOURCES}
  ${MIND_SIM_UTILS_SOURCES}
  ${MIND_SIM_COSIM_SOURCES}
)

set(MIND_SIM_PYTHON_BINDING_SOURCES
  src/python_api/bindings/bindings_macro.cpp
  src/python_api/bindings/bindings_micro.cpp
  src/python_api/bindings/bindings_rules.cpp
  src/python_api/bindings/module.cpp
  src/python_api/bindings/network_builder.cpp
)

set(MIND_SIM_BUILD_PYTHON_API_DIR "${PROJECT_BINARY_DIR}/mind_sim")
set(MIND_SIM_PYTHON_API_SOURCES
  src/python_api/mind_sim/__init__.py
  src/python_api/mind_sim/_network.py
  src/python_api/mind_sim/cli.py
  src/python_api/mind_sim/cosim/__init__.py
  src/python_api/mind_sim/macro/__init__.py
  src/python_api/mind_sim/macro/_io.py
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
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang|NVHPC|PGI")
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

if(CMAKE_CXX_COMPILER_ID MATCHES "NVHPC|PGI")
  target_compile_definitions(_native PRIVATE __GXX_ABI_VERSION=1002)
  foreach(MIND_SIM_NANOBIND_STATIC_TARGET nanobind-static nanobind-static-ps)
    if(TARGET ${MIND_SIM_NANOBIND_STATIC_TARGET})
      target_compile_definitions(${MIND_SIM_NANOBIND_STATIC_TARGET} PRIVATE __GXX_ABI_VERSION=1002)
    endif()
  endforeach()
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
if(MIND_SIM_BINARY_DIST_BUILD)
  set_target_properties(_native PROPERTIES
    INSTALL_RPATH ""
    INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE
  )
endif()

target_link_libraries(_native PRIVATE
  ${CMAKE_DL_LIBS}
  OpenMP::OpenMP_CXX
  ${MIND_SIM_GPU_RUNTIME_LIBRARY}
)
if(MIND_SIM_GPU_LINK_OPTIONS)
  target_link_options(_native PRIVATE ${MIND_SIM_GPU_LINK_OPTIONS})
endif()

if(MIND_SIM_INSTALL_DATA_PREFIX)
  set(MIND_SIM_INSTALL_INCLUDE_DIR "${MIND_SIM_INSTALL_DATA_PREFIX}/include/src")
  set(MIND_SIM_INSTALL_TOOL_DIR "${MIND_SIM_INSTALL_DATA_PREFIX}/bin")
  set(MIND_SIM_INSTALL_LIBRARY_DIR "${MIND_SIM_INSTALL_DATA_PREFIX}/lib")
  set(MIND_SIM_INSTALL_NMODL_SHARE_DIR "${MIND_SIM_INSTALL_DATA_PREFIX}/share/nmodl")
else()
  set(MIND_SIM_INSTALL_INCLUDE_DIR "include/src")
  set(MIND_SIM_INSTALL_TOOL_DIR "${CMAKE_INSTALL_BINDIR}")
  set(MIND_SIM_INSTALL_LIBRARY_DIR "${CMAKE_INSTALL_LIBDIR}")
  set(MIND_SIM_INSTALL_NMODL_SHARE_DIR "share/nmodl")
endif()

install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/src/python_api/mind_sim/"
  DESTINATION "${MIND_SIM_INSTALL_PYTHON_PREFIX}"
  FILES_MATCHING
    PATTERN "*.py"
)
install(TARGETS _native LIBRARY DESTINATION "${MIND_SIM_INSTALL_PYTHON_PREFIX}")
install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/src/"
  DESTINATION "${MIND_SIM_INSTALL_INCLUDE_DIR}"
  FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp"
    PATTERN "*.inc"
)
install(
  DIRECTORY "${MIND_SIM_EXTERNAL_SOURCE_DIR}/Random123/include/Random123"
  DESTINATION "${MIND_SIM_INSTALL_INCLUDE_DIR}/external/Random123/include"
)
install(
  DIRECTORY "${MIND_SIM_EXTERNAL_SOURCE_DIR}/eigen/Eigen"
  DESTINATION "${MIND_SIM_INSTALL_INCLUDE_DIR}/external/eigen"
)
install(
  DIRECTORY "${MIND_SIM_EXTERNAL_SOURCE_DIR}/eigen/unsupported/Eigen"
  DESTINATION "${MIND_SIM_INSTALL_INCLUDE_DIR}/external/eigen/unsupported"
)
install(
  FILES
    "${MIND_SIM_EXTERNAL_SOURCE_DIR}/eigen/COPYING.README"
    "${MIND_SIM_EXTERNAL_SOURCE_DIR}/eigen/COPYING.MPL2"
    "${MIND_SIM_EXTERNAL_SOURCE_DIR}/eigen/COPYING.BSD"
    "${MIND_SIM_EXTERNAL_SOURCE_DIR}/eigen/COPYING.APACHE"
    "${MIND_SIM_EXTERNAL_SOURCE_DIR}/eigen/COPYING.MINPACK"
    "${MIND_SIM_EXTERNAL_SOURCE_DIR}/eigen/COPYING.LGPL"
    "${MIND_SIM_EXTERNAL_SOURCE_DIR}/eigen/COPYING.GPL"
  DESTINATION "${MIND_SIM_INSTALL_INCLUDE_DIR}/external/eigen"
)

add_executable(mind_nrnivmodl src/mod/mind_nrnivmodl.cpp)
target_compile_features(mind_nrnivmodl PRIVATE cxx_std_20)
add_dependencies(mind_nrnivmodl mind_sim_nmodl)
set_target_properties(mind_nrnivmodl PROPERTIES OUTPUT_NAME "mind-nrnivmodl")
if(MIND_SIM_BINARY_DIST_BUILD)
  set_target_properties(mind_nrnivmodl PROPERTIES
    INSTALL_RPATH ""
    INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE
  )
endif()
target_compile_definitions(mind_nrnivmodl PRIVATE
  MIND_SIM_NMODL_PATH="${MIND_SIM_EMBEDDED_NMODL_PATH}"
  MIND_SIM_CXX_COMPILER="${MIND_SIM_EMBEDDED_CXX_COMPILER}"
  MIND_SIM_SOURCE_DIR="${MIND_SIM_EMBEDDED_SOURCE_DIR}"
  MIND_SIM_MECHANISM_INCLUDE_DIR="${MIND_SIM_EMBEDDED_MECHANISM_INCLUDE_DIR}"
  MIND_SIM_MOD_CXX_FLAGS="${MIND_SIM_MOD_CXX_FLAGS}"
  MIND_SIM_MOD_LINK_FLAGS="${MIND_SIM_MOD_LINK_FLAGS}"
)
if(MIND_SIM_ENABLE_GPU)
  target_compile_definitions(mind_nrnivmodl PRIVATE MIND_SIM_ENABLE_GPU=1)
endif()
install(TARGETS nmodl RUNTIME DESTINATION "${MIND_SIM_INSTALL_TOOL_DIR}")
install(TARGETS mind_nrnivmodl RUNTIME DESTINATION "${MIND_SIM_INSTALL_TOOL_DIR}")
if(NOT SKBUILD)
  set(MIND_SIM_CLI_SCRIPT "${PROJECT_BINARY_DIR}/mind-simulator")
  file(GENERATE
    OUTPUT "${MIND_SIM_CLI_SCRIPT}"
    CONTENT "#!/usr/bin/env sh\nexec \"${Python_EXECUTABLE}\" -m mind_sim.cli \"$@\"\n"
  )
  install(PROGRAMS "${MIND_SIM_CLI_SCRIPT}" DESTINATION "${MIND_SIM_INSTALL_TOOL_DIR}")
endif()
if(TARGET pywrapper)
  if(MIND_SIM_BINARY_DIST_BUILD)
    set_target_properties(pywrapper PROPERTIES
      INSTALL_RPATH ""
      INSTALL_REMOVE_ENVIRONMENT_RPATH TRUE
    )
  endif()
  install(TARGETS pywrapper LIBRARY DESTINATION "${MIND_SIM_INSTALL_LIBRARY_DIR}")
endif()
install(
  FILES "${PROJECT_BINARY_DIR}/src/nmodl/share/nmodl/nrnunits.lib"
  DESTINATION "${MIND_SIM_INSTALL_NMODL_SHARE_DIR}"
)
if(MIND_SIM_BINARY_DIST_BUILD AND UNIX AND NOT APPLE)
  install(CODE "
    set(_mind_sim_rpath_files
      \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${MIND_SIM_INSTALL_TOOL_DIR}/nmodl\"
      \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${MIND_SIM_INSTALL_TOOL_DIR}/mind-nrnivmodl\"
      \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${MIND_SIM_INSTALL_LIBRARY_DIR}/libpywrapper.so\"
    )
    file(GLOB _mind_sim_native_files
      \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${MIND_SIM_INSTALL_PYTHON_PREFIX}/_native*.so\"
    )
    list(APPEND _mind_sim_rpath_files \${_mind_sim_native_files})
    foreach(_mind_sim_rpath_file IN LISTS _mind_sim_rpath_files)
      if(EXISTS \"\${_mind_sim_rpath_file}\")
        file(RPATH_REMOVE FILE \"\${_mind_sim_rpath_file}\")
      endif()
    endforeach()
  ")
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
      ENVIRONMENT "PYTHONPATH=${PROJECT_BINARY_DIR};PYTHONDONTWRITEBYTECODE=1"
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

add_custom_target(
  mind_sim_check
  COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
  DEPENDS _native
  VERBATIM
)
