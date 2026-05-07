#include "macro/sim/rule_codegen.hpp"

#include "utils/rule_source_common.hpp"

#include <unordered_set>

namespace mind_sim::macro::sim::codegen {

using namespace mind_sim::utils::rule_source;

namespace {

struct RegionAnalysis {
    std::vector<std::string> inputs{};
    std::vector<std::string> values{};
    std::vector<std::string> locals{};
    std::string code{};
};

void add_field(std::vector<std::string>& ordered,
               std::unordered_set<std::string>& seen,
               const std::string& field) {
    if (seen.insert(field).second) {
        ordered.push_back(field);
    }
}

bool needs_space(const std::string& previous, const std::string& current) {
    if (previous.empty()) {
        return false;
    }
    if (current == ")" || current == "]" || current == "}" || current == ";" ||
        current == "," || current == "." || current == "(" || current == "[") {
        return false;
    }
    if (previous == "(" || previous == "[" || previous == "{" || previous == "." ||
        previous == "!" || previous == "~") {
        return false;
    }
    return true;
}

void require_unique_region_names(const std::vector<std::string>& states,
                                 const std::vector<std::string>& params,
                                 const std::vector<std::string>& inputs) {
    std::unordered_set<std::string> seen;
    for (const auto& name: states) {
        if (!seen.insert(name).second) {
            throw std::runtime_error("RegionRule duplicate name: " + name);
        }
    }
    for (const auto& name: params) {
        if (!seen.insert(name).second) {
            throw std::runtime_error("RegionRule duplicate name: " + name);
        }
    }
    for (const auto& name: inputs) {
        if (!seen.insert(name).second) {
            throw std::runtime_error("RegionRule duplicate name: " + name);
        }
    }
}

RegionAnalysis analyze_region_rule(const std::vector<std::string>& states,
                                   const std::vector<std::string>& params,
                                   const std::string& update) {
    constexpr std::string_view kind = "RegionRule";
    const auto state_fields = names(states, "state");
    const auto param_fields = names(params, "param");
    const auto state_set = member_set(state_fields);
    const auto param_set = member_set(param_fields);
    const auto tokens = scan(update, kind);
    check_balanced(tokens, kind);

    std::unordered_set<std::string> locals = declared_locals(tokens, kind);
    std::vector<std::string> local_fields;
    std::unordered_set<std::string> local_seen;
    for (const auto& token: tokens) {
        if (token.kind == TokenKind::Identifier && contains(locals, token.text)) {
            add_field(local_fields, local_seen, token.text);
        }
    }

    static const std::unordered_set<std::string> runtime{"t", "dt", "roi"};
    std::ostringstream out;
    std::string previous;
    bool statement_start = true;
    int paren_depth = 0;
    int bracket_depth = 0;
    RegionAnalysis analysis;
    std::unordered_set<std::string> input_seen;
    std::unordered_set<std::string> state_used;
    std::unordered_set<std::string> param_used;

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const auto& token = tokens[index];
        if (token.kind == TokenKind::Identifier && contains(forbidden_identifiers(), token.text)) {
            fail(kind, token, "unsupported C++ construct '" + token.text + "'");
        }
        const bool top_level = paren_depth == 0 && bracket_depth == 0;
        const bool member_access = index > 0 && tokens[index - 1].text == ".";
        if (!member_access && index + 1 < tokens.size() && tokens[index + 1].text == ".") {
            fail(kind,
                 token,
                 "ROI dynamics use bare equation names; member access like '" + token.text +
                     ".' is not supported");
        }
        const bool local_assignment = statement_start && top_level &&
                                      token.kind == TokenKind::Identifier &&
                                      index + 1 < tokens.size() &&
                                      tokens[index + 1].text == "=" &&
                                      !member_access &&
                                      !contains(state_set, token.text) &&
                                      !contains(param_set, token.text) &&
                                      !contains(locals, token.text) &&
                                      !contains(runtime, token.text) &&
                                      !contains(keywords(), token.text) &&
                                      !contains(type_words(), token.text) &&
                                      !contains(helper_symbols(), token.text);

        if (local_assignment) {
            names({token.text}, "local");
            if (needs_space(previous, "double")) {
                out << ' ';
            }
            out << "double";
            previous = "double";
            locals.insert(token.text);
            add_field(local_fields, local_seen, token.text);
        }

        if (needs_space(previous, token.text)) {
            out << ' ';
        }
        out << token.text;
        previous = token.text;

        if (token.kind == TokenKind::Identifier && !member_access) {
            const bool is_known =
                contains(locals, token.text) || contains(state_set, token.text) ||
                contains(param_set, token.text) || contains(runtime, token.text) ||
                contains(keywords(), token.text) || contains(type_words(), token.text) ||
                contains(helper_symbols(), token.text);
            if (contains(state_set, token.text)) {
                state_used.insert(token.text);
            } else if (contains(param_set, token.text)) {
                param_used.insert(token.text);
            } else if (!is_known) {
                names({token.text}, "input");
                add_field(analysis.inputs, input_seen, token.text);
            }
        }

        if (token.text == "(") {
            ++paren_depth;
        } else if (token.text == ")" && paren_depth > 0) {
            --paren_depth;
        } else if (token.text == "[") {
            ++bracket_depth;
        } else if (token.text == "]" && bracket_depth > 0) {
            --bracket_depth;
        }

        if ((token.text == ";" && top_level) || token.text == "{" || token.text == "}") {
            statement_start = true;
        } else {
            statement_start = false;
        }
    }

    std::vector<std::string> state_used_ordered;
    for (const auto& field: state_fields) {
        if (contains(state_used, field)) {
            state_used_ordered.push_back(field);
        }
    }
    std::vector<std::string> param_used_ordered;
    for (const auto& field: param_fields) {
        if (contains(param_used, field)) {
            param_used_ordered.push_back(field);
        }
    }
    require_declared_fields_used("RegionRule", "state", state_used_ordered, state_fields);
    require_declared_fields_used("RegionRule", "param", param_used_ordered, param_fields);

    analysis.values = state_fields;
    std::unordered_set<std::string> value_seen = member_set(analysis.values);
    for (const auto& local: local_fields) {
        add_field(analysis.values, value_seen, local);
    }
    analysis.locals = std::move(local_fields);
    analysis.code = out.str();
    return analysis;
}

void require_subset_names(const std::vector<std::string>& values,
                          const std::vector<std::string>& available,
                          std::string_view what) {
    const auto available_set = member_set(available);
    for (const auto& value: values) {
        if (!contains(available_set, value)) {
            throw std::runtime_error(std::string("RegionRule unknown ") +
                                     std::string(what) + ": " + value);
        }
    }
}

std::vector<std::string> intersection_ordered(const std::vector<std::string>& fields,
                                              const std::vector<std::string>& available) {
    const auto available_set = member_set(available);
    std::vector<std::string> out;
    for (const auto& field: fields) {
        if (contains(available_set, field)) {
            out.push_back(field);
        }
    }
    return out;
}

}  // namespace

std::string kernel_name(const std::string& name, const std::string& what) {
    return names({name}, what).front();
}

RegionRuleFields region_rule_fields(const std::vector<std::string>& states,
                                    const std::vector<std::string>& params,
                                    const std::string& update) {
    auto analysis = analyze_region_rule(states, params, update);
    return RegionRuleFields{
        .inputs = std::move(analysis.inputs),
        .exposures = std::move(analysis.values),
    };
}

std::string region_rule_source(const std::vector<std::string>& inputs,
                               const std::vector<std::string>& exposures,
                               const std::vector<std::string>& states,
                               const std::vector<std::string>& params,
                               const std::string& update) {
    auto analysis = analyze_region_rule(states, params, update);
    const auto input_fields = names(inputs, "input");
    const auto output_schema = names(exposures, "exposure");
    const auto state_fields = names(states, "state");
    const auto param_fields = names(params, "param");
    require_unique_region_names(state_fields, param_fields, input_fields);
    require_subset_names(analysis.inputs, input_fields, "input");
    const auto output_fields = intersection_ordered(output_schema, analysis.values);
    const auto state_set = member_set(state_fields);
    const auto local_set = member_set(analysis.locals);

    std::ostringstream source;
    source << common_helpers();
    source << R"cpp(
	extern "C" void mind_region_rule_step(
	    int owner_count,
    const int* roi_indices,
    int roi_count,
    int input_count,
    int exposure_count,
    const double* input_soa,
    double* exposure_soa,
    int state_count,
    double* state_soa,
    int param_count,
    const double* params_soa,
	    double t,
	    double dt) {
	    (void)input_count;
	    (void)exposure_count;
	    (void)state_count;
	    (void)param_count;
	    (void)t;
	    (void)dt;
	)cpp";
    source << R"cpp(    for (int unit = 0; unit < owner_count; ++unit) {
        const int roi = roi_indices[unit];
	)cpp";
    for (const auto& input: analysis.inputs) {
        source << "        const double " << input << " = input_soa[("
               << schema_field_index(input_fields, input) << " * roi_count) + roi];\n";
    }
    for (std::size_t state = 0; state < state_fields.size(); ++state) {
        source << "        double& " << state_fields[state] << " = state_soa[("
               << state << " * owner_count) + unit];\n";
    }
    for (std::size_t param = 0; param < param_fields.size(); ++param) {
        source << "        const double " << param_fields[param] << " = params_soa[("
               << param << " * owner_count) + unit];\n";
    }
    source << indent_block(analysis.code, 8);
    for (const auto& output: output_fields) {
        if (!contains(state_set, output) && !contains(local_set, output)) {
            throw std::runtime_error("RegionRule cannot expose unknown value: " + output);
        }
        source << "        exposure_soa[(" << schema_field_index(output_schema, output)
               << " * roi_count) + roi] = " << output << ";\n";
    }
    source << R"cpp(    }
}
)cpp";
    return source.str();
}

}  // namespace mind_sim::macro::sim::codegen
