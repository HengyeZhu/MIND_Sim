#pragma once

#include "mod/abi.hpp"
#include "utils/dynamic_library.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace mind_sim::mod {

inline const char* rule_kind_name(AbiRuleKind kind) {
    switch (kind) {
    case AbiRuleKind::MacroToMacro:
        return "MacroToMacro";
    case AbiRuleKind::MicroInput:
        return "MicroInput";
    case AbiRuleKind::MicroOutput:
        return "MicroOutput";
    case AbiRuleKind::Region:
        return "Region";
    }
    return "unknown";
}

inline const AbiRuleRegistry& rule_registry_of(
    const mind_sim::utils::DynamicLibrary& library,
    const char* what) {
    const auto registry_fn =
        reinterpret_cast<RuleRegistryFn>(library.symbol("mind_rule_registry"));
    const auto* registry = registry_fn();
    if (registry == nullptr) {
        throw std::runtime_error(std::string(what) + " has null MIND rule registry");
    }
    if (registry->abi_version != kModAbiVersion) {
        throw std::runtime_error(std::string(what) + " MIND rule registry ABI version mismatch");
    }
    if (registry->rule_count < 0) {
        throw std::runtime_error(std::string(what) + " MIND rule registry has negative rule count");
    }
    if (registry->rule_count > 0 && registry->rules == nullptr) {
        throw std::runtime_error(std::string(what) + " MIND rule registry has null entries");
    }
    return *registry;
}

inline std::vector<const AbiRuleDescriptor*> rule_descriptors(
    const mind_sim::utils::DynamicLibrary& library,
    const char* what) {
    const auto& registry = rule_registry_of(library, what);
    std::vector<const AbiRuleDescriptor*> out;
    out.reserve(static_cast<std::size_t>(registry.rule_count));
    for (int index = 0; index < registry.rule_count; ++index) {
        const auto& entry = registry.rules[index];
        if (entry.descriptor == nullptr) {
            throw std::runtime_error(std::string(what) + " MIND rule registry has null descriptor");
        }
        const auto* descriptor = entry.descriptor;
        if (descriptor->abi_version != kModAbiVersion) {
            throw std::runtime_error(std::string(what) + " MIND rule descriptor ABI version mismatch");
        }
        if (!descriptor->name || descriptor->name[0] == '\0') {
            throw std::runtime_error(std::string(what) + " MIND rule descriptor has empty name");
        }
        out.push_back(descriptor);
    }
    return out;
}

inline const AbiRuleEntry& find_rule_entry(
    const mind_sim::utils::DynamicLibrary& library,
    AbiRuleKind expected,
    const std::string& rule_name,
    const char* what) {
    if (rule_name.empty()) {
        throw std::runtime_error(std::string(what) + " rule name must be non-empty");
    }
    const auto& registry = rule_registry_of(library, what);
    for (int index = 0; index < registry.rule_count; ++index) {
        const auto& entry = registry.rules[index];
        if (entry.descriptor == nullptr) {
            throw std::runtime_error(std::string(what) + " MIND rule registry has null descriptor");
        }
        const auto* descriptor = entry.descriptor;
        if (descriptor->abi_version != kModAbiVersion) {
            throw std::runtime_error(std::string(what) + " MIND rule descriptor ABI version mismatch");
        }
        if (descriptor->kind != static_cast<int>(expected)) {
            continue;
        }
        if (descriptor->name != nullptr && rule_name == descriptor->name) {
            return entry;
        }
    }
    throw std::runtime_error(std::string(what) + " rule '" + rule_name + "' was not found as " +
                             rule_kind_name(expected));
}

}  // namespace mind_sim::mod
