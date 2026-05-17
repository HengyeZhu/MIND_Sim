#include "mind_mod/rule_mod.hpp"

#include "mind_mod/rule_mod_internal.hpp"
#include "utils/rule_source_common.hpp"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mind_sim::mind_mod {

namespace {

using namespace mind_sim::utils::rule_source;
using internal::add_field;
using internal::analyze_bare_code;

struct RegionStepAnalysis {
    std::vector<std::string> read{};
    std::vector<std::string> state{};
    std::vector<std::string> params{};
    std::vector<std::string> locals{};
    std::vector<std::string> values{};
    std::string code{};
};

struct FieldStepAnalysis {
    std::vector<std::string> read{};
    std::vector<std::string> state{};
    std::vector<std::string> params{};
    std::vector<std::string> locals{};
    std::vector<std::string> local_states{};
    std::string code{};
};

[[nodiscard]] std::vector<std::string> names_from_defaults(const std::vector<NamedDefault>& values) {
    std::vector<std::string> out;
    out.reserve(values.size());
    for (const auto& value: values) {
        out.push_back(value.name);
    }
    return out;
}

[[nodiscard]] std::string cpp_string(std::string_view value) {
    std::ostringstream out;
    out << '"';
    for (char c: value) {
        switch (c) {
        case '\\':
            out << "\\\\";
            break;
        case '"':
            out << "\\\"";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            out << c;
            break;
        }
    }
    out << '"';
    return out.str();
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

void require_disjoint_names(
    std::string_view kind,
    std::initializer_list<std::pair<std::string_view, std::vector<std::string>>> groups) {
    std::unordered_set<std::string> seen;
    for (const auto& [group_name, values]: groups) {
        for (const auto& value: values) {
            if (!seen.insert(value).second) {
                throw std::runtime_error(std::string(kind) +
                                         " uses ambiguous bare variable name: " + value +
                                         " (" + std::string(group_name) + ")");
            }
        }
    }
}

[[nodiscard]] std::vector<std::string> ordered_used(const std::vector<std::string>& declared,
                                                    const std::unordered_set<std::string>& used) {
    std::vector<std::string> out;
    for (const auto& field: declared) {
        if (contains(used, field)) {
            out.push_back(field);
        }
    }
    return out;
}

[[nodiscard]] RegionStepAnalysis analyze_region_step(const RuleSpec& spec) {
    constexpr std::string_view kind = "MindMod region STEP";
    const auto input_fields = names(spec.read, "READ input");
    const auto output_fields = names(spec.write, "WRITE exposure");
    const auto state_fields = names_from_defaults(spec.state);
    const auto param_fields = names_from_defaults(spec.params);
    names(state_fields, "STATE");
    names(param_fields, "PARAMETER");
    const std::vector<std::string> runtime_fields{"t", "dt", "roi"};
    require_disjoint_names(kind,
                           {{"READ", input_fields},
                            {"STATE", state_fields},
                            {"PARAMETER", param_fields},
                            {"runtime", runtime_fields}});

    const auto input_set = member_set(input_fields);
    const auto state_set = member_set(state_fields);
    const auto param_set = member_set(param_fields);
    const auto runtime_set = member_set(runtime_fields);
    const auto tokens = scan(spec.step, kind);
    check_balanced(tokens, kind);

    std::unordered_set<std::string> locals = declared_locals(tokens, kind);
    std::vector<std::string> local_fields;
    std::unordered_set<std::string> local_seen;
    for (const auto& token: tokens) {
        if (token.kind == TokenKind::Identifier && contains(locals, token.text)) {
            add_field(local_fields, local_seen, token.text);
        }
    }

    RegionStepAnalysis analysis;
    std::unordered_set<std::string> input_used;
    std::unordered_set<std::string> state_used;
    std::unordered_set<std::string> param_used;
    std::ostringstream out;
    std::string previous;
    bool statement_start = true;
    int paren_depth = 0;
    int bracket_depth = 0;

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const auto& token = tokens[index];
        if (token.kind == TokenKind::Identifier &&
            (contains(forbidden_identifiers(), token.text) || token.text == "static_cast")) {
            fail(kind, token, "unsupported C++ construct '" + token.text + "'");
        }
        const bool top_level = paren_depth == 0 && bracket_depth == 0;
        const bool member_access = index > 0 && tokens[index - 1].text == ".";
        if (!member_access && index + 1 < tokens.size() && tokens[index + 1].text == ".") {
            fail(kind,
                 token,
                 "MIND .mod rules use bare variable names; member access like '" + token.text +
                     ".' is not supported");
        }
        const bool known_symbol =
            contains(locals, token.text) || contains(input_set, token.text) ||
            contains(state_set, token.text) || contains(param_set, token.text) ||
            contains(runtime_set, token.text) || contains(keywords(), token.text) ||
            contains(type_words(), token.text) || contains(helper_symbols(), token.text);
        const bool local_assignment = statement_start && top_level &&
                                      token.kind == TokenKind::Identifier &&
                                      index + 1 < tokens.size() &&
                                      tokens[index + 1].text == "=" &&
                                      !member_access &&
                                      !known_symbol;
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
            if (contains(input_set, token.text)) {
                input_used.insert(token.text);
            } else if (contains(state_set, token.text)) {
                state_used.insert(token.text);
            } else if (contains(param_set, token.text)) {
                param_used.insert(token.text);
            } else if (!contains(locals, token.text) && !contains(runtime_set, token.text) &&
                       !contains(keywords(), token.text) && !contains(type_words(), token.text) &&
                       !contains(helper_symbols(), token.text)) {
                names({token.text}, "symbol");
                fail(kind, token, "unknown symbol '" + token.text + "'");
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

    analysis.read = ordered_used(input_fields, input_used);
    analysis.state = ordered_used(state_fields, state_used);
    analysis.params = ordered_used(param_fields, param_used);
    require_declared_fields_used("MindMod region", "READ input", analysis.read, input_fields);
    require_declared_fields_used("MindMod region", "state", analysis.state, state_fields);
    require_declared_fields_used("MindMod region", "param", analysis.params, param_fields);

    analysis.values = state_fields;
    std::unordered_set<std::string> value_seen = member_set(analysis.values);
    for (const auto& local: local_fields) {
        add_field(analysis.values, value_seen, local);
    }
    const auto value_set = member_set(analysis.values);
    for (const auto& output: output_fields) {
        if (!contains(value_set, output)) {
            throw std::runtime_error("MindMod region WRITE exposure is not a state or STEP local: " + output);
        }
    }
    analysis.locals = std::move(local_fields);
    analysis.code = out.str();
    return analysis;
}

[[nodiscard]] FieldStepAnalysis analyze_field_step(const RuleSpec& spec) {
    constexpr std::string_view kind = "MindMod neural field STEP";
    const auto input_fields = names(spec.read, "READ input");
    const auto output_fields = names(spec.write, "WRITE exposure");
    const auto state_fields = names_from_defaults(spec.state);
    const auto param_fields = names_from_defaults(spec.params);
    names(state_fields, "STATE");
    names(param_fields, "PARAMETER");
    const std::vector<std::string> runtime_fields{"t", "dt", "roi", "node"};
    require_disjoint_names(kind,
                           {{"READ", input_fields},
                            {"STATE", state_fields},
                            {"PARAMETER", param_fields},
                            {"runtime", runtime_fields}});

    const auto input_set = member_set(input_fields);
    const auto state_set = member_set(state_fields);
    const auto param_set = member_set(param_fields);
    const auto runtime_set = member_set(runtime_fields);
    for (const auto& output: output_fields) {
        if (!contains(state_set, output)) {
            throw std::runtime_error("MindMod neural field WRITE exposure must name a STATE: " + output);
        }
    }

    const auto tokens = scan(spec.step, kind);
    check_balanced(tokens, kind);
    std::unordered_set<std::string> locals = declared_locals(tokens, kind);
    std::vector<std::string> local_fields;
    std::unordered_set<std::string> local_seen;
    for (const auto& token: tokens) {
        if (token.kind == TokenKind::Identifier && contains(locals, token.text)) {
            add_field(local_fields, local_seen, token.text);
        }
    }

    FieldStepAnalysis analysis;
    std::unordered_set<std::string> input_used;
    std::unordered_set<std::string> state_used;
    std::unordered_set<std::string> param_used;
    std::unordered_set<std::string> local_state_seen;
    std::ostringstream out;
    std::string previous;
    bool statement_start = true;
    int paren_depth = 0;
    int bracket_depth = 0;

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const auto& token = tokens[index];
        if (token.kind == TokenKind::Identifier &&
            (contains(forbidden_identifiers(), token.text) || token.text == "static_cast")) {
            fail(kind, token, "unsupported C++ construct '" + token.text + "'");
        }
        const bool top_level = paren_depth == 0 && bracket_depth == 0;
        const bool member_access = index > 0 && tokens[index - 1].text == ".";
        if (!member_access && index + 1 < tokens.size() && tokens[index + 1].text == ".") {
            fail(kind,
                 token,
                 "MIND .mod rules use bare variable names; member access like '" + token.text +
                     ".' is not supported");
        }

        if (token.kind == TokenKind::Identifier && token.text == "local" && !member_access) {
            if (index + 3 >= tokens.size() || tokens[index + 1].text != "(" ||
                tokens[index + 2].kind != TokenKind::Identifier ||
                tokens[index + 3].text != ")") {
                fail(kind, token, "local coupling must be written as local(state_name)");
            }
            const auto& state = tokens[index + 2];
            if (!contains(state_set, state.text)) {
                fail(kind, state, "local() argument must be a declared STATE");
            }
            state_used.insert(state.text);
            add_field(analysis.local_states, local_state_seen, state.text);
            const auto replacement = "local_" + state.text;
            if (needs_space(previous, replacement)) {
                out << ' ';
            }
            out << replacement;
            previous = replacement;
            index += 3;
            statement_start = false;
            continue;
        }

        const bool known_symbol =
            contains(locals, token.text) || contains(input_set, token.text) ||
            contains(state_set, token.text) || contains(param_set, token.text) ||
            contains(runtime_set, token.text) || contains(keywords(), token.text) ||
            contains(type_words(), token.text) || contains(helper_symbols(), token.text) ||
            token.text == "local";
        const bool local_assignment = statement_start && top_level &&
                                      token.kind == TokenKind::Identifier &&
                                      index + 1 < tokens.size() &&
                                      tokens[index + 1].text == "=" &&
                                      !member_access &&
                                      !known_symbol;
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
            if (contains(input_set, token.text)) {
                input_used.insert(token.text);
            } else if (contains(state_set, token.text)) {
                state_used.insert(token.text);
            } else if (contains(param_set, token.text)) {
                param_used.insert(token.text);
            } else if (!contains(locals, token.text) && !contains(runtime_set, token.text) &&
                       !contains(keywords(), token.text) && !contains(type_words(), token.text) &&
                       !contains(helper_symbols(), token.text) && token.text != "local") {
                names({token.text}, "symbol");
                fail(kind, token, "unknown symbol '" + token.text + "'");
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

    analysis.read = ordered_used(input_fields, input_used);
    analysis.state = ordered_used(state_fields, state_used);
    analysis.params = ordered_used(param_fields, param_used);
    require_declared_fields_used("MindMod neural field", "READ input", analysis.read, input_fields);
    require_declared_fields_used("MindMod neural field", "state", analysis.state, state_fields);
    require_declared_fields_used("MindMod neural field", "param", analysis.params, param_fields);
    for (const auto& state: analysis.local_states) {
        const auto generated = "local_" + state;
        if (contains(locals, generated) || contains(param_set, generated) ||
            contains(state_set, generated) || contains(input_set, generated)) {
            throw std::runtime_error("MindMod neural field local() helper conflicts with name: " + generated);
        }
    }
    analysis.locals = std::move(local_fields);
    analysis.code = out.str();
    return analysis;
}

void append_name_array(std::ostringstream& source,
                       const std::string& symbol,
                       const std::vector<std::string>& values) {
    if (values.empty()) {
        return;
    }
    source << "static const char* const " << symbol << "[] = {";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            source << ", ";
        }
        source << cpp_string(values[index]);
    }
    source << "};\n";
}

void append_default_array(std::ostringstream& source,
                          const std::string& symbol,
                          const std::vector<NamedDefault>& values) {
    if (values.empty()) {
        return;
    }
    source << "static const double " << symbol << "[] = {";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index != 0) {
            source << ", ";
        }
        source << values[index].value;
    }
    source << "};\n";
}

[[nodiscard]] std::string array_pointer(const std::string& symbol,
                                        const std::vector<std::string>& values) {
    return values.empty() ? "nullptr" : symbol;
}

[[nodiscard]] std::string default_pointer(const std::string& symbol,
                                          const std::vector<NamedDefault>& values) {
    return values.empty() ? "nullptr" : symbol;
}

[[nodiscard]] std::string compiled_abi_source() {
    return R"cpp(
struct mind_rule_descriptor {
    int abi_version;
    int kind;
    const char* name;
    int read_count;
    const char* const* read_names;
    int write_count;
    const char* const* write_names;
    int emit_count;
    const char* const* emit_names;
    int param_count;
    const char* const* param_names;
    const double* param_defaults;
    int state_count;
    const char* const* state_names;
    const double* state_defaults;
    int random_count;
    const char* const* random_names;
    int local_state_count;
    const char* const* local_state_names;
};

struct mind_event_writer {
    void* user;
    void (*write)(void*, double, int);
};

struct mind_spike_table {
    const double* time;
    const int* gid;
    int size;
};

struct mind_random_stream {
    void* user;
    double (*uniform)(void*, int, int);
};

static inline double uniform(mind_random_stream& stream, int index = 0, int draw = 0) {
    return stream.uniform(stream.user, index, draw);
}

struct mind_coupling_context {
    int roi_count;
    int input_count;
    int exposure_count;
    int history_capacity;
    int step;
    int target_count;
    const int* target_indices;
    const int* target_edge_offsets;
    const int* edge_sources;
    const double* edge_weights;
    const int* edge_delay_steps;
    const int* edge_delay_offsets;
    const double* history;
    double* inputs;
    int param_count;
    const double* params;
    const int* read_exposure_offsets;
    const int* write_input_offsets;
};

struct mind_micro_input_context {
    int input_count;
    int roi_count;
    int roi;
    const double* input_soa;
    int state_count;
    double* state;
    int param_count;
    const double* params;
    double start_time;
    double stop_time;
    int input_port_count;
    const int* input_port_bases;
    mind_event_writer* event_writer;
    const int* read_input_offsets;
    int random_count;
    mind_random_stream* random_streams;
};

struct mind_micro_output_context {
    int exposure_count;
    const mind_spike_table* spikes;
    int roi_count;
    int roi;
    double* exposure_soa;
    int state_count;
    double* state;
    int param_count;
    const double* params;
    double start_time;
    double stop_time;
    const int* write_exposure_offsets;
};

struct mind_region_context {
    int owner_count;
    const int* roi_indices;
    int roi_count;
    int input_count;
    const double* input_soa;
    int exposure_count;
    double* exposure_soa;
    int state_count;
    double* state_soa;
    int param_count;
    const double* params_soa;
    const int* read_input_offsets;
    const int* write_exposure_offsets;
    double t;
    double dt;
};

struct mind_neural_field_context {
    int node_count;
    const int* node_to_roi;
    int roi_count;
    int input_count;
    const double* input_soa;
    int state_count;
    const double* previous_state_soa;
    double* state_soa;
    int param_count;
    const double* params;
    const int* local_indptr;
    const int* local_indices;
    const double* local_weights;
    const int* read_input_offsets;
    double t;
    double dt;
};
)cpp";
}

void append_descriptor(std::ostringstream& source,
                       const RuleSpec& spec,
                       const std::vector<std::string>& local_states) {
    const auto param_names = names_from_defaults(spec.params);
    const auto state_names = names_from_defaults(spec.state);
    append_name_array(source, "mind_read_names", spec.read);
    append_name_array(source, "mind_write_names", spec.write);
    append_name_array(source, "mind_emit_names", spec.emit);
    append_name_array(source, "mind_random_names", spec.random);
    append_name_array(source, "mind_local_state_names", local_states);
    append_name_array(source, "mind_param_names", param_names);
    append_name_array(source, "mind_state_names", state_names);
    append_default_array(source, "mind_param_defaults", spec.params);
    append_default_array(source, "mind_state_defaults", spec.state);
    int kind = 0;
    if (spec.kind == RuleKind::MicroInput) {
        kind = 1;
    } else if (spec.kind == RuleKind::MicroOutput) {
        kind = 2;
    } else if (spec.kind == RuleKind::Region) {
        kind = 3;
    } else if (spec.kind == RuleKind::NeuralField) {
        kind = 4;
    }
    source << "static const mind_rule_descriptor mind_descriptor = {\n";
    source << "    3,\n";
    source << "    " << kind << ",\n";
    source << "    " << cpp_string(spec.name) << ",\n";
    source << "    " << spec.read.size() << ", " << array_pointer("mind_read_names", spec.read) << ",\n";
    source << "    " << spec.write.size() << ", " << array_pointer("mind_write_names", spec.write) << ",\n";
    source << "    " << spec.emit.size() << ", " << array_pointer("mind_emit_names", spec.emit) << ",\n";
    source << "    " << spec.params.size() << ", "
           << array_pointer("mind_param_names", param_names) << ", "
           << default_pointer("mind_param_defaults", spec.params) << ",\n";
    source << "    " << spec.state.size() << ", "
           << array_pointer("mind_state_names", state_names) << ", "
           << default_pointer("mind_state_defaults", spec.state) << ",\n";
    source << "    " << spec.random.size() << ", "
           << array_pointer("mind_random_names", spec.random) << ",\n";
    source << "    " << local_states.size() << ", "
           << array_pointer("mind_local_state_names", local_states) << ",\n";
    source << "};\n";
    source << R"cpp(
extern "C" const mind_rule_descriptor* mind_rule_descriptor() {
    return &mind_descriptor;
}
)cpp";
}

void append_param_values(std::ostringstream& source,
                         const std::vector<std::string>& fields,
                         int indent) {
    const auto spaces = std::string(static_cast<std::size_t>(indent), ' ');
    for (std::size_t index = 0; index < fields.size(); ++index) {
        source << spaces << "const double " << fields[index] << " = params[" << index << "];\n";
    }
}

void append_state_refs(std::ostringstream& source,
                       const std::vector<std::string>& fields,
                       int indent) {
    const auto spaces = std::string(static_cast<std::size_t>(indent), ' ');
    for (std::size_t index = 0; index < fields.size(); ++index) {
        source << spaces << "double& " << fields[index] << " = state[" << index << "];\n";
    }
}

void append_compiled_region(std::ostringstream& source,
                            const RuleSpec& spec,
                            const RegionStepAnalysis& analysis) {
    const auto input_fields = names(spec.read, "READ input");
    const auto output_fields = names(spec.write, "WRITE exposure");
    const auto state_fields = names_from_defaults(spec.state);
    const auto param_fields = names_from_defaults(spec.params);

    source << R"cpp(
extern "C" void mind_region_rule_apply(const mind_region_context* ctx) {
    const int owner_count = ctx->owner_count;
    const int* roi_indices = ctx->roi_indices;
    const int roi_count = ctx->roi_count;
    const double* input_soa = ctx->input_soa;
    double* exposure_soa = ctx->exposure_soa;
    double* state_soa = ctx->state_soa;
    const double* params_soa = ctx->params_soa;
    const int* read_offsets = ctx->read_input_offsets;
    const int* write_offsets = ctx->write_exposure_offsets;
    const double t = ctx->t;
    const double dt = ctx->dt;
    (void)owner_count;
    (void)roi_count;
    (void)input_soa;
    (void)exposure_soa;
    (void)state_soa;
    (void)params_soa;
    (void)read_offsets;
    (void)write_offsets;
    (void)t;
    (void)dt;
    for (int unit = 0; unit < owner_count; ++unit) {
        const int roi = roi_indices[unit];
)cpp";
    for (std::size_t index = 0; index < input_fields.size(); ++index) {
        source << "        const double " << input_fields[index]
               << " = input_soa[read_offsets[" << index << "] + roi];\n";
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
    for (std::size_t index = 0; index < output_fields.size(); ++index) {
        source << "        exposure_soa[write_offsets[" << index << "] + roi] = "
               << output_fields[index] << ";\n";
    }
    source << R"cpp(    }
}
)cpp";
}

void append_compiled_neural_field(std::ostringstream& source,
                                  const RuleSpec& spec,
                                  const FieldStepAnalysis& analysis) {
    const auto input_fields = names(spec.read, "READ input");
    const auto state_fields = names_from_defaults(spec.state);
    const auto param_fields = names_from_defaults(spec.params);

    source << R"cpp(
extern "C" void mind_neural_field_rule_apply(const mind_neural_field_context* ctx) {
    const int node_count = ctx->node_count;
    const int* node_to_roi = ctx->node_to_roi;
    const int roi_count = ctx->roi_count;
    const double* input_soa = ctx->input_soa;
    const double* previous_state_soa = ctx->previous_state_soa;
    double* state_soa = ctx->state_soa;
    const double* params = ctx->params;
    const int* local_indptr = ctx->local_indptr;
    const int* local_indices = ctx->local_indices;
    const double* local_weights = ctx->local_weights;
    const int* read_offsets = ctx->read_input_offsets;
    const double t = ctx->t;
    const double dt = ctx->dt;
    (void)node_count;
    (void)node_to_roi;
    (void)roi_count;
    (void)input_soa;
    (void)previous_state_soa;
    (void)state_soa;
    (void)params;
    (void)local_indptr;
    (void)local_indices;
    (void)local_weights;
    (void)read_offsets;
    (void)t;
    (void)dt;
    for (int node = 0; node < node_count; ++node) {
        const int roi = node_to_roi[node];
)cpp";
    for (std::size_t index = 0; index < input_fields.size(); ++index) {
        source << "        const double " << input_fields[index]
               << " = input_soa[read_offsets[" << index << "] + roi];\n";
    }
    for (std::size_t state = 0; state < state_fields.size(); ++state) {
        source << "        double& " << state_fields[state] << " = state_soa[("
               << state << " * node_count) + node];\n";
    }
    for (const auto& state: analysis.local_states) {
        const auto state_index = schema_field_index(state_fields, state);
        source << "        double local_" << state << " = 0.0;\n";
        source << "        for (int edge = local_indptr[node]; edge < local_indptr[node + 1]; ++edge) {\n";
        source << "            local_" << state << " += local_weights[edge] * previous_state_soa[("
               << state_index << " * node_count) + local_indices[edge]];\n";
        source << "        }\n";
    }
    for (std::size_t param = 0; param < param_fields.size(); ++param) {
        source << "        const double " << param_fields[param] << " = params[" << param << "];\n";
    }
    source << indent_block(analysis.code, 8);
    source << R"cpp(    }
}
)cpp";
}

void append_compiled_coupling(std::ostringstream& source, const RuleSpec& spec) {
    const auto param_fields = names_from_defaults(spec.params);
    const std::vector<std::string> edge_builtins{
        "weight",
        "delay_steps",
        "source_roi",
        "target_roi",
    };
    const auto analysis = analyze_bare_code("MindMod coupling EDGE",
                                            spec.edge,
                                            spec.read,
                                            spec.write,
                                            {},
                                            param_fields,
                                            {},
                                            {},
                                            edge_builtins,
                                            {});
    require_declared_fields_used("MindMod coupling", "READ exposure", analysis.read, spec.read);
    require_declared_fields_used("MindMod coupling", "WRITE input", analysis.write, spec.write);
    require_declared_fields_used("MindMod coupling", "param", analysis.params, param_fields);

    source << R"cpp(
extern "C" void mind_coupling_rule_apply(const mind_coupling_context* ctx) {
    const int roi_count = ctx->roi_count;
    const int history_capacity = ctx->history_capacity;
    const int step = ctx->step;
    const int target_count = ctx->target_count;
    const int* target_indices = ctx->target_indices;
    const int* target_edge_offsets = ctx->target_edge_offsets;
    const int* edge_sources = ctx->edge_sources;
    const double* edge_weights = ctx->edge_weights;
    const int* edge_delay_steps = ctx->edge_delay_steps;
    const int* edge_delay_offsets = ctx->edge_delay_offsets;
    const double* history = ctx->history;
    double* inputs = ctx->inputs;
    const double* params = ctx->params;
    const int* read_offsets = ctx->read_exposure_offsets;
    const int* write_offsets = ctx->write_input_offsets;
    (void)edge_weights;
    (void)edge_delay_steps;
    (void)params;
)cpp";
    append_param_values(source, param_fields, 4);
    source << R"cpp(    const int history_stride = ctx->exposure_count * roi_count;
    const int history_size = history_capacity * history_stride;
    const int current_history_offset = (step % history_capacity) * history_stride;
    for (int target_pos = 0; target_pos < target_count; ++target_pos) {
        const int target_roi = target_indices[target_pos];
)cpp";
    for (std::size_t index = 0; index < spec.write.size(); ++index) {
        source << "        double& " << spec.write[index]
               << " = inputs[write_offsets[" << index << "] + target_roi];\n";
    }
    source << R"cpp(        const int edge_begin = target_edge_offsets[target_roi];
        const int edge_end = target_edge_offsets[target_roi + 1];
        for (int edge_index = edge_begin; edge_index < edge_end; ++edge_index) {
            const int source_roi = edge_sources[edge_index];
)cpp";
    if (has_field(analysis.edge, "weight")) {
        source << "            const double weight = edge_weights[edge_index];\n";
    }
    if (has_field(analysis.edge, "delay_steps")) {
        source << "            const int delay_steps = edge_delay_steps[edge_index];\n";
    }
    source << R"cpp(            int history_offset = current_history_offset + edge_delay_offsets[edge_index];
            if (history_offset >= history_size) {
                history_offset -= history_size;
            }
            const double* history_slot = history + history_offset;
)cpp";
    for (std::size_t index = 0; index < spec.read.size(); ++index) {
        source << "            const double " << spec.read[index]
               << " = history_slot[read_offsets[" << index << "] + source_roi];\n";
    }
    source << indent_block(analysis.code, 12);
    source << R"cpp(        }
    }
}
)cpp";
}

void append_compiled_micro_input(std::ostringstream& source, const RuleSpec& spec) {
    const auto param_fields = names_from_defaults(spec.params);
    const auto state_fields = names_from_defaults(spec.state);
    const std::vector<std::string> runtime_fields{"t", "dt", "tstop", "roi"};
    const auto analysis = analyze_bare_code("MindMod micro input INPUT",
                                            spec.input,
                                            spec.read,
                                            {},
                                            state_fields,
                                            param_fields,
                                            spec.emit,
                                            spec.random,
                                            {},
                                            runtime_fields);
    require_declared_fields_used("MindMod micro input", "READ input", analysis.read, spec.read);
    require_declared_fields_used("MindMod micro input", "EMIT port", analysis.ports, spec.emit);
    require_declared_fields_used("MindMod micro input", "RANDOM stream", analysis.random, spec.random);
    require_declared_fields_used("MindMod micro input", "state", analysis.state, state_fields);
    require_declared_fields_used("MindMod micro input", "param", analysis.params, param_fields);

    source << R"cpp(
extern "C" void mind_micro_input_rule_apply(const mind_micro_input_context* ctx) {
    const int roi_count = ctx->roi_count;
    const int roi = ctx->roi;
    const double* input_soa = ctx->input_soa;
    double* state = ctx->state;
    const double* params = ctx->params;
    const int* input_port_bases = ctx->input_port_bases;
    mind_event_writer* event_writer = ctx->event_writer;
    const int* read_offsets = ctx->read_input_offsets;
    mind_random_stream* random_streams = ctx->random_streams;
    const double t = ctx->start_time;
    const double tstop = ctx->stop_time;
    const double dt = tstop - t;
    (void)roi_count;
    (void)roi;
    (void)input_soa;
    (void)state;
    (void)params;
    (void)input_port_bases;
    (void)event_writer;
    (void)read_offsets;
    (void)random_streams;
)cpp";
    for (std::size_t index = 0; index < spec.read.size(); ++index) {
        source << "    const double " << spec.read[index]
               << " = input_soa[read_offsets[" << index << "] + roi];\n";
    }
    append_state_refs(source, state_fields, 4);
    append_param_values(source, param_fields, 4);
    for (std::size_t port = 0; port < spec.emit.size(); ++port) {
        source << "    auto " << spec.emit[port] << " = [&](double time, int local_index) {\n";
        source << "        event_writer->write(event_writer->user, time, input_port_bases["
               << port << "] + local_index);\n";
        source << "    };\n";
    }
    for (std::size_t random = 0; random < spec.random.size(); ++random) {
        source << "    mind_random_stream& " << spec.random[random]
               << " = random_streams[" << random << "];\n";
    }
    source << indent_block(analysis.code, 4);
    source << "}\n";
}

void append_compiled_micro_output(std::ostringstream& source, const RuleSpec& spec) {
    const auto param_fields = names_from_defaults(spec.params);
    const auto state_fields = names_from_defaults(spec.state);
    const std::vector<std::string> receive_runtime_fields{"t", "gid", "dt"};
    const std::vector<std::string> breakpoint_runtime_fields{"t", "dt"};
    const auto receive_analysis = analyze_bare_code("MindMod micro output NET_RECEIVE",
                                                   spec.net_receive,
                                                   {},
                                                   spec.write,
                                                   state_fields,
                                                   param_fields,
                                                   {},
                                                   {},
                                                   {},
                                                   receive_runtime_fields);
    const auto breakpoint_analysis = analyze_bare_code("MindMod micro output BREAKPOINT",
                                                       spec.breakpoint,
                                                       {},
                                                       spec.write,
                                                       state_fields,
                                                       param_fields,
                                                       {},
                                                       {},
                                                       {},
                                                       breakpoint_runtime_fields);
    std::vector<std::string> output_fields = receive_analysis.write;
    std::unordered_set<std::string> output_seen = member_set(output_fields);
    for (const auto& field: breakpoint_analysis.write) {
        add_field(output_fields, output_seen, field);
    }
    std::vector<std::string> state_used = receive_analysis.state;
    std::unordered_set<std::string> state_seen = member_set(state_used);
    for (const auto& field: breakpoint_analysis.state) {
        add_field(state_used, state_seen, field);
    }
    std::vector<std::string> param_used = receive_analysis.params;
    std::unordered_set<std::string> param_seen = member_set(param_used);
    for (const auto& field: breakpoint_analysis.params) {
        add_field(param_used, param_seen, field);
    }
    require_declared_fields_used("MindMod micro output", "WRITE exposure", output_fields, spec.write);
    require_declared_fields_used("MindMod micro output", "state", state_used, state_fields);
    require_declared_fields_used("MindMod micro output", "param", param_used, param_fields);

    source << R"cpp(
extern "C" void mind_micro_output_rule_apply(const mind_micro_output_context* ctx) {
    const mind_spike_table* spikes = ctx->spikes;
    const int roi_count = ctx->roi_count;
    const int roi = ctx->roi;
    double* exposure_soa = ctx->exposure_soa;
    double* state = ctx->state;
    const double* params = ctx->params;
    const int* write_offsets = ctx->write_exposure_offsets;
    const double dt = ctx->stop_time - ctx->start_time;
    const int spike_count = spikes->size;
    (void)roi_count;
    (void)roi;
    (void)exposure_soa;
    (void)state;
    (void)params;
    (void)write_offsets;
)cpp";
    for (std::size_t index = 0; index < spec.write.size(); ++index) {
        source << "    double& " << spec.write[index]
               << " = exposure_soa[write_offsets[" << index << "] + roi];\n";
    }
    append_state_refs(source, state_fields, 4);
    append_param_values(source, param_fields, 4);
    source << R"cpp(    for (int spike_index = 0; spike_index < spike_count; ++spike_index) {
        const double t = spikes->time[spike_index];
        const int gid = spikes->gid[spike_index];
)cpp";
    source << indent_block(receive_analysis.code, 8);
    source << R"cpp(    }
    const double t = ctx->stop_time;
)cpp";
    source << indent_block(breakpoint_analysis.code, 4);
    source << "}\n";
}

}  // namespace

std::string compiled_rule_source(const std::string& source_text,
                                 const std::string& origin) {
    const auto spec = parse_rule_source(source_text, origin);
    RegionStepAnalysis region_analysis;
    FieldStepAnalysis field_analysis;
    std::vector<std::string> local_states;
    if (spec.kind == RuleKind::Region) {
        region_analysis = analyze_region_step(spec);
    } else if (spec.kind == RuleKind::NeuralField) {
        field_analysis = analyze_field_step(spec);
        local_states = field_analysis.local_states;
    }
    std::ostringstream source;
    source << common_helpers();
    source << compiled_abi_source();
    append_descriptor(source, spec, local_states);
    if (spec.kind == RuleKind::Coupling) {
        append_compiled_coupling(source, spec);
    } else if (spec.kind == RuleKind::MicroInput) {
        append_compiled_micro_input(source, spec);
    } else if (spec.kind == RuleKind::MicroOutput) {
        append_compiled_micro_output(source, spec);
    } else if (spec.kind == RuleKind::Region) {
        append_compiled_region(source, spec, region_analysis);
    } else {
        append_compiled_neural_field(source, spec, field_analysis);
    }
    return source.str();
}


}  // namespace mind_sim::mind_mod
