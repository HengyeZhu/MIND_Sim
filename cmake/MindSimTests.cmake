include_guard(GLOBAL)

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
mind_sim_add_python_test(mind_sim_node_to_roi_map_file_check tests/macro/node_to_roi_map_file_check.py)
mind_sim_add_python_test(mind_sim_neural_field_check tests/macro/neural_field_check.py)
mind_sim_add_python_test(mind_sim_neural_field_global_coupling_check tests/macro/neural_field_global_coupling_check.py)

add_custom_target(
  mind_sim_check
  COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure
  DEPENDS _native
  VERBATIM
)
