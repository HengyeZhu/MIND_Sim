#include "mind_mod/rule_mod_internal.hpp"

#include "utils/rule_source_common.hpp"

#include <algorithm>
#include <cstddef>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace mind_sim::mind_mod::internal {

namespace {

using namespace mind_sim::utils::rule_source;

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

void require_disjoint_bare_names(
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


}  // namespace

void add_field(std::vector<std::string>& ordered,
               std::unordered_set<std::string>& seen,
               const std::string& field) {
    if (seen.insert(field).second) {
        ordered.push_back(field);
    }
}

[[nodiscard]] BareAnalysis analyze_bare_code(std::string_view kind,
                                             const std::string& code,
                                             const std::vector<std::string>& read_fields,
                                             const std::vector<std::string>& write_fields,
                                             const std::vector<std::string>& state_fields,
                                             const std::vector<std::string>& param_fields,
                                             const std::vector<std::string>& port_names,
                                             const std::vector<std::string>& random_fields,
                                             const std::vector<std::string>& edge_fields,
                                             const std::vector<std::string>& runtime_fields) {
    const auto read_set = member_set(read_fields);
    const auto write_set = member_set(write_fields);
    const auto state_set = member_set(state_fields);
    const auto param_set = member_set(param_fields);
    const auto port_set = member_set(port_names);
    const auto random_set = member_set(random_fields);
    const auto edge_set = member_set(edge_fields);
    const auto runtime_set = member_set(runtime_fields);
    require_disjoint_bare_names(kind,
                                {{"READ", read_fields},
                                 {"WRITE", write_fields},
                                 {"STATE", state_fields},
                                 {"PARAMETER", param_fields},
                                 {"EMIT", port_names},
                                 {"RANDOM", random_fields},
                                 {"runtime", runtime_fields},
                                 {"edge", edge_fields}});

    const auto tokens = scan(code, kind);
    check_balanced(tokens, kind);
    std::unordered_set<std::string> locals = declared_locals(tokens, kind);
    std::vector<std::string> local_fields;
    std::unordered_set<std::string> local_seen;
    for (const auto& token: tokens) {
        if (token.kind == TokenKind::Identifier && contains(locals, token.text)) {
            add_field(local_fields, local_seen, token.text);
        }
    }

    BareAnalysis analysis;
    std::unordered_set<std::string> read_used;
    std::unordered_set<std::string> write_used;
    std::unordered_set<std::string> state_used;
    std::unordered_set<std::string> param_used;
    std::unordered_set<std::string> port_used;
    std::unordered_set<std::string> random_used;
    std::unordered_set<std::string> edge_used;

    std::ostringstream out;
    std::string previous;
    bool statement_start = true;
    int paren_depth = 0;
    int bracket_depth = 0;
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const auto& token = tokens[index];
        if (token.kind == TokenKind::Identifier && contains(forbidden_identifiers(), token.text)) {
            fail(kind, token, "unsupported C++ construct '" + token.text + "'");
        }
        if (token.kind == TokenKind::Identifier && token.text == "static_cast") {
            fail(kind, token, "MIND .mod snippets use double values directly; C++ casts are not supported");
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
            contains(locals, token.text) || contains(read_set, token.text) ||
            contains(write_set, token.text) || contains(state_set, token.text) ||
            contains(param_set, token.text) || contains(port_set, token.text) ||
            contains(random_set, token.text) || contains(edge_set, token.text) ||
            contains(runtime_set, token.text) ||
            contains(keywords(), token.text) || contains(type_words(), token.text) ||
            contains(helper_symbols(), token.text);
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
            if (contains(read_set, token.text)) {
                read_used.insert(token.text);
            } else if (contains(write_set, token.text)) {
                write_used.insert(token.text);
            } else if (contains(state_set, token.text)) {
                state_used.insert(token.text);
            } else if (contains(param_set, token.text)) {
                param_used.insert(token.text);
            } else if (contains(port_set, token.text)) {
                if (index + 1 >= tokens.size() || tokens[index + 1].text != "(") {
                    fail(kind, token, "EMIT port '" + token.text + "' must be called as a function");
                }
                port_used.insert(token.text);
            } else if (contains(random_set, token.text)) {
                const bool first_uniform_arg = index >= 2 && tokens[index - 1].text == "(" &&
                                               tokens[index - 2].text == "uniform";
                if (!first_uniform_arg) {
                    fail(kind,
                         token,
                         "RANDOM stream '" + token.text +
                             "' may only be used as the first argument to uniform(...)");
                }
                random_used.insert(token.text);
            } else if (contains(edge_set, token.text)) {
                edge_used.insert(token.text);
            } else if (token.text == "uniform") {
                if (index + 2 >= tokens.size() || tokens[index + 1].text != "(" ||
                    token.kind != TokenKind::Identifier ||
                    tokens[index + 2].kind != TokenKind::Identifier ||
                    !contains(random_set, tokens[index + 2].text)) {
                    fail(kind, token, "uniform(...) requires a declared RANDOM stream as its first argument");
                }
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

    analysis.read = ordered_used(read_fields, read_used);
    analysis.write = ordered_used(write_fields, write_used);
    analysis.state = ordered_used(state_fields, state_used);
    analysis.params = ordered_used(param_fields, param_used);
    analysis.ports = ordered_used(port_names, port_used);
    analysis.random = ordered_used(random_fields, random_used);
    analysis.edge = ordered_used(edge_fields, edge_used);
    analysis.locals = std::move(local_fields);
    analysis.code = out.str();
    return analysis;
}

}  // namespace mind_sim::mind_mod::internal
