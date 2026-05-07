#include "mind_mod/rule_mod.hpp"

#include "utils/rule_source_common.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace mind_sim::mind_mod {

namespace {

using namespace mind_sim::utils::rule_source;

[[nodiscard]] std::string trim(std::string_view value) {
    std::size_t begin = 0;
    while (begin < value.size() &&
           std::isspace(static_cast<unsigned char>(value[begin])) != 0) {
        ++begin;
    }
    std::size_t end = value.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(value[end - 1])) != 0) {
        --end;
    }
    return std::string(value.substr(begin, end - begin));
}

[[nodiscard]] std::string upper(std::string value) {
    for (char& c: value) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return value;
}

[[nodiscard]] bool is_word_boundary(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) == 0 && c != '_';
}

[[nodiscard]] bool starts_keyword_at(std::string_view source,
                                     std::size_t index,
                                     std::string_view keyword) {
    if (index + keyword.size() > source.size()) {
        return false;
    }
    for (std::size_t i = 0; i < keyword.size(); ++i) {
        if (std::toupper(static_cast<unsigned char>(source[index + i])) !=
            std::toupper(static_cast<unsigned char>(keyword[i]))) {
            return false;
        }
    }
    const bool before_ok =
        index == 0 || is_word_boundary(source[index - 1]);
    const bool after_ok =
        index + keyword.size() == source.size() ||
        is_word_boundary(source[index + keyword.size()]);
    return before_ok && after_ok;
}

[[nodiscard]] std::string strip_line_comment(std::string_view line) {
    bool in_string = false;
    char quote = '\0';
    bool escaped = false;
    for (std::size_t i = 0; i + 1 < line.size(); ++i) {
        const char c = line[i];
        if (escaped) {
            escaped = false;
            continue;
        }
        if (in_string) {
            if (c == '\\') {
                escaped = true;
            } else if (c == quote) {
                in_string = false;
            }
            continue;
        }
        if (c == '"' || c == '\'') {
            in_string = true;
            quote = c;
            continue;
        }
        if (c == '/' && line[i + 1] == '/') {
            return std::string(line.substr(0, i));
        }
    }
    return std::string(line);
}

[[nodiscard]] std::vector<std::string> split_words(std::string_view line) {
    std::vector<std::string> out;
    std::string word;
    for (char c: line) {
        if (std::isspace(static_cast<unsigned char>(c)) != 0 || c == ',') {
            if (!word.empty()) {
                out.push_back(std::move(word));
                word.clear();
            }
            continue;
        }
        word.push_back(c);
    }
    if (!word.empty()) {
        out.push_back(std::move(word));
    }
    return out;
}

[[noreturn]] void parse_fail(const std::string& origin, const std::string& message) {
    throw std::runtime_error(origin + ": " + message);
}

[[nodiscard]] std::string block_body(std::string_view source,
                                     std::string_view keyword,
                                     const std::string& origin,
                                     bool required) {
    std::size_t search = 0;
    while (search < source.size()) {
        const auto found = source.find(keyword, search);
        if (found == std::string_view::npos) {
            break;
        }
        if (!starts_keyword_at(source, found, keyword)) {
            search = found + keyword.size();
            continue;
        }
        auto open = source.find('{', found + keyword.size());
        if (open == std::string_view::npos) {
            parse_fail(origin, "block '" + std::string(keyword) + "' is missing '{'");
        }
        int depth = 1;
        bool in_string = false;
        char quote = '\0';
        bool escaped = false;
        for (std::size_t index = open + 1; index < source.size(); ++index) {
            const char c = source[index];
            if (escaped) {
                escaped = false;
                continue;
            }
            if (in_string) {
                if (c == '\\') {
                    escaped = true;
                } else if (c == quote) {
                    in_string = false;
                }
                continue;
            }
            if (c == '"' || c == '\'') {
                in_string = true;
                quote = c;
                continue;
            }
            if (c == '{') {
                ++depth;
            } else if (c == '}') {
                --depth;
                if (depth == 0) {
                    return std::string(source.substr(open + 1, index - open - 1));
                }
            }
        }
        parse_fail(origin, "block '" + std::string(keyword) + "' is missing '}'");
    }
    if (required) {
        parse_fail(origin, "missing required block '" + std::string(keyword) + "'");
    }
    return {};
}

void append_names(std::vector<std::string>& destination,
                  const std::vector<std::string>& values,
                  std::string_view what,
                  const std::string& origin) {
    for (const auto& value: values) {
        if (!is_valid_identifier(value)) {
            parse_fail(origin, std::string(what) + " name is not a valid identifier: " + value);
        }
        if (std::find(destination.begin(), destination.end(), value) == destination.end()) {
            destination.push_back(value);
        }
    }
}

[[nodiscard]] std::vector<NamedDefault> parse_named_defaults(std::string_view body,
                                                             std::string_view what,
                                                             const std::string& origin) {
    std::vector<NamedDefault> out;
    std::unordered_set<std::string> seen;
    std::istringstream lines{std::string(body)};
    std::string line;
    while (std::getline(lines, line)) {
        line = trim(strip_line_comment(line));
        if (line.empty()) {
            continue;
        }
        const auto words = split_words(line);
        if (words.empty()) {
            continue;
        }
        const auto name = words.front();
        if (!is_valid_identifier(name)) {
            parse_fail(origin, std::string(what) + " name is not a valid identifier: " + name);
        }
        if (!seen.insert(name).second) {
            parse_fail(origin, std::string(what) + " is declared more than once: " + name);
        }
        double value = 0.0;
        const auto eq = line.find('=');
        if (eq != std::string::npos) {
            const auto tail = trim(std::string_view(line).substr(eq + 1));
            char* end = nullptr;
            value = std::strtod(tail.c_str(), &end);
            if (end == tail.c_str() || !std::isfinite(value)) {
                parse_fail(origin, std::string(what) + " default is not a finite number: " + name);
            }
        }
        out.push_back(NamedDefault{.name = name, .value = value});
    }
    return out;
}

[[nodiscard]] std::vector<std::string> names_from_defaults(const std::vector<NamedDefault>& values) {
    std::vector<std::string> out;
    out.reserve(values.size());
    for (const auto& value: values) {
        out.push_back(value.name);
    }
    return out;
}

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

}  // namespace

std::string kind_name(RuleKind kind) {
    switch (kind) {
    case RuleKind::Coupling:
        return "coupling";
    case RuleKind::MicroInput:
        return "micro_input";
    case RuleKind::MicroOutput:
        return "micro_output";
    }
    return "unknown";
}

RuleSpec parse_rule_source(const std::string& source, const std::string& origin) {
    RuleSpec spec;
    const auto mind = block_body(source, "MIND", origin, true);
    bool kind_set = false;
    std::istringstream lines(mind);
    std::string line;
    while (std::getline(lines, line)) {
        line = trim(strip_line_comment(line));
        if (line.empty()) {
            continue;
        }
        const auto words = split_words(line);
        if (words.empty()) {
            continue;
        }
        const auto key = upper(words.front());
        if (key == "COUPLING" || key == "MICRO_INPUT" || key == "MICRO_OUTPUT") {
            if (kind_set) {
                parse_fail(origin, "MIND block declares the rule kind more than once");
            }
            kind_set = true;
            if (words.size() != 2) {
                parse_fail(origin, key + " declaration requires exactly one rule name");
            }
            if (key == "COUPLING") {
                spec.kind = RuleKind::Coupling;
            } else if (key == "MICRO_INPUT") {
                spec.kind = RuleKind::MicroInput;
            } else {
                spec.kind = RuleKind::MicroOutput;
            }
            spec.name = words[1];
            if (!is_valid_identifier(spec.name)) {
                parse_fail(origin, "rule name is not a valid identifier: " + spec.name);
            }
        } else if (key == "READ") {
            append_names(spec.read,
                         std::vector<std::string>(words.begin() + 1, words.end()),
                         "READ",
                         origin);
        } else if (key == "WRITE") {
            append_names(spec.write,
                         std::vector<std::string>(words.begin() + 1, words.end()),
                         "WRITE",
                         origin);
        } else if (key == "EMIT") {
            append_names(spec.emit,
                         std::vector<std::string>(words.begin() + 1, words.end()),
                         "EMIT",
                         origin);
        } else if (key == "RANDOM") {
            append_names(spec.random,
                         std::vector<std::string>(words.begin() + 1, words.end()),
                         "RANDOM",
                         origin);
        } else {
            parse_fail(origin, "unknown MIND directive: " + words.front());
        }
    }
    if (!kind_set) {
        parse_fail(origin, "MIND block must declare COUPLING, MICRO_INPUT, or MICRO_OUTPUT");
    }

    spec.params = parse_named_defaults(block_body(source, "PARAMETER", origin, false),
                                       "PARAMETER",
                                       origin);
    spec.state = parse_named_defaults(block_body(source, "STATE", origin, false),
                                      "STATE",
                                      origin);
    spec.edge = block_body(source, "EDGE", origin, false);
    spec.input = block_body(source, "INPUT", origin, false);
    spec.net_receive = block_body(source, "NET_RECEIVE", origin, false);
    spec.breakpoint = block_body(source, "BREAKPOINT", origin, false);

    if (spec.kind == RuleKind::Coupling) {
        if (spec.read.empty()) {
            parse_fail(origin, "COUPLING rule requires at least one READ exposure");
        }
        if (spec.write.empty()) {
            parse_fail(origin, "COUPLING rule requires at least one WRITE input");
        }
        if (trim(spec.edge).empty()) {
            parse_fail(origin, "COUPLING rule requires an EDGE block");
        }
    } else if (spec.kind == RuleKind::MicroInput) {
        if (spec.emit.empty()) {
            parse_fail(origin, "MICRO_INPUT rule requires at least one EMIT port");
        }
        if (trim(spec.input).empty()) {
            parse_fail(origin, "MICRO_INPUT rule requires an INPUT block");
        }
    } else if (spec.kind == RuleKind::MicroOutput) {
        if (!spec.random.empty()) {
            parse_fail(origin, "RANDOM streams are currently supported only by MICRO_INPUT rules");
        }
        if (spec.write.empty()) {
            parse_fail(origin, "MICRO_OUTPUT rule requires at least one WRITE exposure");
        }
        if (trim(spec.net_receive).empty() && trim(spec.breakpoint).empty()) {
            parse_fail(origin, "MICRO_OUTPUT rule requires NET_RECEIVE or BREAKPOINT code");
        }
    }
    if (spec.kind == RuleKind::Coupling && !spec.random.empty()) {
        parse_fail(origin, "RANDOM streams are currently supported only by MICRO_INPUT rules");
    }
    return spec;
}

namespace {

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
)cpp";
}

void append_descriptor(std::ostringstream& source, const RuleSpec& spec) {
    const auto param_names = names_from_defaults(spec.params);
    const auto state_names = names_from_defaults(spec.state);
    append_name_array(source, "mind_read_names", spec.read);
    append_name_array(source, "mind_write_names", spec.write);
    append_name_array(source, "mind_emit_names", spec.emit);
    append_name_array(source, "mind_random_names", spec.random);
    append_name_array(source, "mind_param_names", param_names);
    append_name_array(source, "mind_state_names", state_names);
    append_default_array(source, "mind_param_defaults", spec.params);
    append_default_array(source, "mind_state_defaults", spec.state);
    int kind = 0;
    if (spec.kind == RuleKind::MicroInput) {
        kind = 1;
    } else if (spec.kind == RuleKind::MicroOutput) {
        kind = 2;
    }
    source << "static const mind_rule_descriptor mind_descriptor = {\n";
    source << "    2,\n";
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
    std::ostringstream source;
    source << common_helpers();
    source << compiled_abi_source();
    append_descriptor(source, spec);
    if (spec.kind == RuleKind::Coupling) {
        append_compiled_coupling(source, spec);
    } else if (spec.kind == RuleKind::MicroInput) {
        append_compiled_micro_input(source, spec);
    } else {
        append_compiled_micro_output(source, spec);
    }
    return source.str();
}

}  // namespace mind_sim::mind_mod
