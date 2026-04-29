#pragma once

#include <string>
#include <vector>

namespace mind_sim::bridge::sim::codegen {

[[nodiscard]] std::string micro_input_rule_source(const std::vector<std::string>& inputs,
                                                  const std::vector<std::string>& ports,
                                                  const std::vector<std::string>& states,
                                                  const std::vector<std::string>& params,
                                                  const std::string& emit);

[[nodiscard]] std::string micro_output_rule_source(const std::vector<std::string>& exposures,
                                                   const std::vector<std::string>& states,
                                                   const std::vector<std::string>& params,
                                                   const std::string& spike,
                                                   const std::string& finish);

}  // namespace mind_sim::bridge::sim::codegen
