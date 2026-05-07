#pragma once

#include <string>
#include <vector>

namespace mind_sim::mind_mod {

enum class RuleKind {
    Coupling,
    MicroInput,
    MicroOutput,
};

struct NamedDefault {
    std::string name{};
    double value{0.0};
};

struct RuleSpec {
    RuleKind kind{RuleKind::Coupling};
    std::string name{};
    std::vector<std::string> read{};
    std::vector<std::string> write{};
    std::vector<std::string> emit{};
    std::vector<std::string> random{};
    std::vector<NamedDefault> params{};
    std::vector<NamedDefault> state{};
    std::string edge{};
    std::string input{};
    std::string net_receive{};
    std::string breakpoint{};
};

[[nodiscard]] RuleSpec parse_rule_source(const std::string& source, const std::string& origin);
[[nodiscard]] std::string kind_name(RuleKind kind);
[[nodiscard]] std::string compiled_rule_source(const std::string& source,
                                               const std::string& origin);

}  // namespace mind_sim::mind_mod
