#pragma once

#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace mind_sim::mind_mod::internal {

struct BareAnalysis {
    std::vector<std::string> read{};
    std::vector<std::string> write{};
    std::vector<std::string> state{};
    std::vector<std::string> params{};
    std::vector<std::string> ports{};
    std::vector<std::string> random{};
    std::vector<std::string> edge{};
    std::vector<std::string> locals{};
    std::string code{};
};

void add_field(std::vector<std::string>& ordered,
               std::unordered_set<std::string>& seen,
               const std::string& field);

[[nodiscard]] BareAnalysis analyze_bare_code(std::string_view kind,
                                             const std::string& code,
                                             const std::vector<std::string>& read_fields,
                                             const std::vector<std::string>& write_fields,
                                             const std::vector<std::string>& state_fields,
                                             const std::vector<std::string>& param_fields,
                                             const std::vector<std::string>& port_names,
                                             const std::vector<std::string>& random_fields,
                                             const std::vector<std::string>& edge_fields,
                                             const std::vector<std::string>& runtime_fields);

}  // namespace mind_sim::mind_mod::internal
