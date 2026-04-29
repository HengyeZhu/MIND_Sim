#pragma once

#include <string>
#include <vector>

namespace mind_sim::macro::sim::codegen {

[[nodiscard]] std::string kernel_name(const std::string& name, const std::string& what);

[[nodiscard]] std::string region_rule_source(const std::vector<std::string>& inputs,
                                             const std::vector<std::string>& exposures,
                                             const std::vector<std::string>& states,
                                             const std::vector<std::string>& params,
                                             const std::string& update);

[[nodiscard]] std::string coupling_projection_rule_source(
    const std::vector<std::string>& inputs,
    const std::vector<std::string>& exposures,
    const std::vector<std::string>& params,
    const std::string& edge,
    const std::string& finish,
    int roi_count);

}  // namespace mind_sim::macro::sim::codegen
