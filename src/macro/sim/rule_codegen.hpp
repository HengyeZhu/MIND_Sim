#pragma once

#include <string>
#include <vector>

namespace mind_sim::macro::sim::codegen {

struct RegionRuleFields {
    std::vector<std::string> inputs{};
    std::vector<std::string> exposures{};
};

[[nodiscard]] std::string kernel_name(const std::string& name, const std::string& what);

[[nodiscard]] RegionRuleFields region_rule_fields(const std::vector<std::string>& states,
                                                  const std::vector<std::string>& params,
                                                  const std::string& update);

[[nodiscard]] std::string region_rule_source(const std::vector<std::string>& inputs,
                                             const std::vector<std::string>& exposures,
                                             const std::vector<std::string>& states,
                                             const std::vector<std::string>& params,
                                             const std::string& update);

}  // namespace mind_sim::macro::sim::codegen
