include_guard(GLOBAL)

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
  list(APPEND generated_sources "${generated_dir}/${MODGEN_MODL_REG_NAME}.cpp")

  set(include_args)
  foreach(include_dir IN LISTS MODGEN_INCLUDES)
    list(APPEND include_args --include "${include_dir}")
  endforeach()

  set(define_args)
  foreach(define IN LISTS MODGEN_DEFINES)
    list(APPEND define_args --define "${define}")
  endforeach()

  set(stamp "${MODGEN_OUTPUT_DIR}/${MODGEN_TARGET}.stamp")
  file(MAKE_DIRECTORY "${MODGEN_OUTPUT_DIR}")
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
