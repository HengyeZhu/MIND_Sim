#pragma once

#include "mod/rule_api.hpp"
#include "utils/dynamic_library.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mind_sim::mod {

struct RegisteredRule {
    int id{-1};
    RuleKind kind{RuleKind::Region};
    std::string name{};
    std::shared_ptr<mind_sim::utils::DynamicLibrary> library{};
    std::vector<std::string> exposure_names{};
    std::vector<std::string> read_source_exposure_names{};
    std::vector<std::string> read_source_variable_names{};
    std::vector<std::string> read_target_exposure_names{};
    std::vector<std::string> read_target_variable_names{};
    std::vector<std::string> write_source_exposure_names{};
    std::vector<std::string> write_source_variable_names{};
    std::vector<std::string> write_target_exposure_names{};
    std::vector<std::string> write_target_variable_names{};
    std::vector<std::string> param_names{};
    std::vector<double> param_defaults{};
    std::vector<std::string> state_names{};
    std::vector<double> state_defaults{};
    MacroToMacroApplyFn macro_to_macro_apply{nullptr};
    MicroInputApplyFn micro_input_apply{nullptr};
    MicroOutputApplyFn micro_output_apply{nullptr};
    RegionApplyFn region_apply{nullptr};
};

inline const char* rule_kind_name(RuleKind kind) {
    switch (kind) {
    case RuleKind::MacroToMacro:
        return "MacroToMacro";
    case RuleKind::MicroInput:
        return "MicroInput";
    case RuleKind::MicroOutput:
        return "MicroOutput";
    case RuleKind::Region:
        return "Region";
    }
    return "unknown";
}

namespace detail {

struct RuleCatalog {
    std::mutex mutex{};
    int next_id{0};
    std::unordered_map<std::string, std::shared_ptr<RegisteredRule>> by_key{};
    std::unordered_map<std::string, std::vector<std::string>> library_keys{};
};

struct RuleCollector {
    const char* what{nullptr};
    std::vector<RegisteredRule> rules{};
};

inline RuleCatalog& rule_catalog() {
    static RuleCatalog catalog;
    return catalog;
}

inline std::string rule_key(const std::string& library_path,
                            RuleKind kind,
                            const std::string& name) {
    return library_path + '\n' + std::to_string(static_cast<int>(kind)) + '\n' + name;
}

inline void require_nonnegative_count(int count, const std::string& field, const char* what) {
    if (count < 0) {
        throw std::runtime_error(std::string(what) + " MIND rule registration has negative " +
                                 field);
    }
}

inline std::string copy_rule_name(const char* name, const char* what) {
    if (name == nullptr || name[0] == '\0') {
        throw std::runtime_error(std::string(what) + " MIND rule registration has empty name");
    }
    return name;
}

inline std::vector<std::string> copy_names(NameList names, const std::string& field, const char* what) {
    require_nonnegative_count(names.count, field, what);
    if (names.count > 0 && names.names == nullptr) {
        throw std::runtime_error(std::string(what) + " MIND rule registration has null " + field);
    }
    std::vector<std::string> out;
    out.reserve(static_cast<std::size_t>(names.count));
    for (int index = 0; index < names.count; ++index) {
        if (names.names[index] == nullptr) {
            throw std::runtime_error(std::string(what) +
                                     " MIND rule registration has null entry in " + field);
        }
        out.emplace_back(names.names[index]);
    }
    return out;
}

inline std::vector<double> copy_defaults(ValueList values, const std::string& field, const char* what) {
    require_nonnegative_count(values.count, field, what);
    if (values.count > 0 && values.defaults == nullptr) {
        throw std::runtime_error(std::string(what) + " MIND rule registration has null " + field);
    }
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(values.count));
    for (int index = 0; index < values.count; ++index) {
        out.push_back(values.defaults[index]);
    }
    return out;
}

inline void copy_binding(BindingList binding,
                         const char* field,
                         std::vector<std::string>& exposure_names,
                         std::vector<std::string>& variable_names,
                         const char* what) {
    require_nonnegative_count(binding.count, field, what);
    exposure_names = copy_names(NameList{binding.count, binding.exposure_names},
                                std::string(field) + "_exposure_names",
                                what);
    variable_names = copy_names(NameList{binding.count, binding.variable_names},
                                std::string(field) + "_variable_names",
                                what);
}

inline void copy_values(ValueList values,
                        const char* field,
                        std::vector<std::string>& names,
                        std::vector<double>& defaults,
                        const char* what) {
    names = copy_names(NameList{values.count, values.names},
                       std::string(field) + "_names",
                       what);
    defaults = copy_defaults(values, std::string(field) + "_defaults", what);
}

inline RuleCollector& collector_from(void* user_data) {
    if (user_data == nullptr) {
        throw std::runtime_error("MIND rule registrar has null user_data");
    }
    return *static_cast<RuleCollector*>(user_data);
}

inline void register_region(void* user_data,
                            const char* name,
                            NameList exposures,
                            ValueList states,
                            ValueList params,
                            RegionApplyFn apply) {
    auto& collector = collector_from(user_data);
    if (apply == nullptr) {
        throw std::runtime_error(std::string(collector.what) +
                                 " Region rule has null apply function");
    }
    RegisteredRule rule;
    rule.kind = RuleKind::Region;
    rule.name = copy_rule_name(name, collector.what);
    rule.exposure_names = copy_names(exposures, "exposure_names", collector.what);
    copy_values(states, "state", rule.state_names, rule.state_defaults, collector.what);
    copy_values(params, "param", rule.param_names, rule.param_defaults, collector.what);
    rule.region_apply = apply;
    collector.rules.push_back(std::move(rule));
}

inline void register_macro_to_macro(void* user_data,
                                    const char* name,
                                    BindingList read_source,
                                    BindingList read_target,
                                    BindingList write_source,
                                    BindingList write_target,
                                    ValueList params,
                                    MacroToMacroApplyFn apply) {
    auto& collector = collector_from(user_data);
    if (apply == nullptr) {
        throw std::runtime_error(std::string(collector.what) +
                                 " MacroToMacro rule has null apply function");
    }
    RegisteredRule rule;
    rule.kind = RuleKind::MacroToMacro;
    rule.name = copy_rule_name(name, collector.what);
    copy_binding(read_source,
                 "read_source",
                 rule.read_source_exposure_names,
                 rule.read_source_variable_names,
                 collector.what);
    copy_binding(read_target,
                 "read_target",
                 rule.read_target_exposure_names,
                 rule.read_target_variable_names,
                 collector.what);
    copy_binding(write_source,
                 "write_source",
                 rule.write_source_exposure_names,
                 rule.write_source_variable_names,
                 collector.what);
    copy_binding(write_target,
                 "write_target",
                 rule.write_target_exposure_names,
                 rule.write_target_variable_names,
                 collector.what);
    copy_values(params, "param", rule.param_names, rule.param_defaults, collector.what);
    rule.macro_to_macro_apply = apply;
    collector.rules.push_back(std::move(rule));
}

inline void register_micro_input(void* user_data,
                                 const char* name,
                                 BindingList read_source,
                                 ValueList states,
                                 ValueList params,
                                 MicroInputApplyFn apply) {
    auto& collector = collector_from(user_data);
    if (apply == nullptr) {
        throw std::runtime_error(std::string(collector.what) +
                                 " MicroInput rule has null apply function");
    }
    RegisteredRule rule;
    rule.kind = RuleKind::MicroInput;
    rule.name = copy_rule_name(name, collector.what);
    copy_binding(read_source,
                 "read_source",
                 rule.read_source_exposure_names,
                 rule.read_source_variable_names,
                 collector.what);
    copy_values(states, "state", rule.state_names, rule.state_defaults, collector.what);
    copy_values(params, "param", rule.param_names, rule.param_defaults, collector.what);
    rule.micro_input_apply = apply;
    collector.rules.push_back(std::move(rule));
}

inline void register_micro_output(void* user_data,
                                  const char* name,
                                  BindingList write_target,
                                  ValueList states,
                                  ValueList params,
                                  MicroOutputApplyFn apply) {
    auto& collector = collector_from(user_data);
    if (apply == nullptr) {
        throw std::runtime_error(std::string(collector.what) +
                                 " MicroOutput rule has null apply function");
    }
    RegisteredRule rule;
    rule.kind = RuleKind::MicroOutput;
    rule.name = copy_rule_name(name, collector.what);
    copy_binding(write_target,
                 "write_target",
                 rule.write_target_exposure_names,
                 rule.write_target_variable_names,
                 collector.what);
    copy_values(states, "state", rule.state_names, rule.state_defaults, collector.what);
    copy_values(params, "param", rule.param_names, rule.param_defaults, collector.what);
    rule.micro_output_apply = apply;
    collector.rules.push_back(std::move(rule));
}

inline std::vector<std::shared_ptr<const RegisteredRule>> rules_for_keys_locked(
    const RuleCatalog& catalog,
    const std::vector<std::string>& keys) {
    std::vector<std::shared_ptr<const RegisteredRule>> out;
    out.reserve(keys.size());
    for (const auto& key : keys) {
        const auto found = catalog.by_key.find(key);
        if (found != catalog.by_key.end()) {
            out.push_back(found->second);
        }
    }
    return out;
}

}  // namespace detail

inline std::vector<std::shared_ptr<const RegisteredRule>> register_rules_from_library(
    std::shared_ptr<mind_sim::utils::DynamicLibrary> library,
    const char* what) {
    if (!library) {
        throw std::runtime_error(std::string(what) + " library handle is null");
    }
    auto& catalog = detail::rule_catalog();
    {
        std::lock_guard<std::mutex> guard(catalog.mutex);
        const auto found = catalog.library_keys.find(library->path());
        if (found != catalog.library_keys.end()) {
            return detail::rules_for_keys_locked(catalog, found->second);
        }
    }

    const auto registration_fn =
        reinterpret_cast<RuleRegFn>(library->symbol("mind_rule_reg"));
    detail::RuleCollector collector{
        .what = what,
        .rules = {},
    };
    RuleRegistrar registrar{
        .user_data = &collector,
        .register_region = detail::register_region,
        .register_macro_to_macro = detail::register_macro_to_macro,
        .register_micro_input = detail::register_micro_input,
        .register_micro_output = detail::register_micro_output,
    };
    registration_fn(&registrar);

    std::lock_guard<std::mutex> guard(catalog.mutex);
    std::vector<std::shared_ptr<const RegisteredRule>> out;
    std::vector<std::string> keys;
    std::unordered_set<std::string> seen_keys;
    out.reserve(collector.rules.size());
    keys.reserve(collector.rules.size());
    for (auto& rule : collector.rules) {
        rule.library = library;
        const auto key = detail::rule_key(library->path(), rule.kind, rule.name);
        if (!seen_keys.insert(key).second) {
            throw std::runtime_error(std::string(what) + " has duplicate MIND rule '" +
                                     rule.name + "'");
        }
        keys.push_back(key);
        auto registered = std::make_shared<RegisteredRule>(std::move(rule));
        registered->id = catalog.next_id++;
        out.push_back(registered);
        catalog.by_key[key] = std::move(registered);
    }
    catalog.library_keys[library->path()] = std::move(keys);
    return out;
}

inline std::shared_ptr<const RegisteredRule> find_rule_entry(
    std::shared_ptr<mind_sim::utils::DynamicLibrary> library,
    RuleKind expected,
    const std::string& rule_name,
    const char* what) {
    if (rule_name.empty()) {
        throw std::runtime_error(std::string(what) + " rule name must be non-empty");
    }
    register_rules_from_library(library, what);
    const auto key = detail::rule_key(library->path(), expected, rule_name);
    auto& catalog = detail::rule_catalog();
    std::lock_guard<std::mutex> guard(catalog.mutex);
    const auto found = catalog.by_key.find(key);
    if (found != catalog.by_key.end()) {
        return found->second;
    }
    throw std::runtime_error(std::string(what) + " rule '" + rule_name + "' was not found as " +
                             rule_kind_name(expected));
}

}  // namespace mind_sim::mod
