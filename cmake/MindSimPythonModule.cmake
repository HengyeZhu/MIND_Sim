include_guard(GLOBAL)

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

function(mind_sim_apply_coreneuron_abi target)
  target_compile_definitions("${target}" PRIVATE
    ${MIND_SIM_CORENEURON_MOD_DEFINES}
  )
endfunction()

function(mind_sim_apply_mind_runtime_definitions target)
  target_compile_definitions("${target}" PRIVATE
    MIND_CORE_MECH_ABI_VERSION=4
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
mind_sim_apply_coreneuron_abi(mind_sim_coreneuron)
mind_sim_apply_gpu_definitions(mind_sim_coreneuron)
mind_sim_apply_runtime_compile_options(mind_sim_coreneuron)
mind_sim_suppress_vendor_diagnostics(mind_sim_coreneuron)

add_library(mind_sim_default_mechanism_objects OBJECT ${MIND_SIM_DEFAULT_MECHANISM_GENERATED_SOURCES})
add_dependencies(mind_sim_default_mechanism_objects mind_sim_default_mechanisms)
mind_sim_apply_native_includes(mind_sim_default_mechanism_objects)
mind_sim_apply_coreneuron_abi(mind_sim_default_mechanism_objects)
mind_sim_apply_gpu_definitions(mind_sim_default_mechanism_objects)
mind_sim_apply_runtime_compile_options(mind_sim_default_mechanism_objects)
mind_sim_suppress_vendor_diagnostics(mind_sim_default_mechanism_objects)

add_library(mind_sim_native_core OBJECT ${MIND_SIM_NATIVE_CORE_SOURCES})
add_dependencies(mind_sim_native_core mind_sim_default_mechanisms)
mind_sim_apply_native_includes(mind_sim_native_core)
mind_sim_apply_coreneuron_abi(mind_sim_native_core)
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
mind_sim_apply_coreneuron_abi(_native)
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
