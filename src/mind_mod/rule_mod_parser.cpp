#include "mind_mod/rule_mod.hpp"

#include "utils/rule_source_common.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
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

}  // namespace

std::string kind_name(RuleKind kind) {
    switch (kind) {
    case RuleKind::Coupling:
        return "coupling";
    case RuleKind::MicroInput:
        return "micro_input";
    case RuleKind::MicroOutput:
        return "micro_output";
    case RuleKind::Region:
        return "region";
    case RuleKind::NeuralField:
        return "neural_field";
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
        if (key == "COUPLING" || key == "MICRO_INPUT" || key == "MICRO_OUTPUT" ||
            key == "REGION" || key == "NEURAL_FIELD") {
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
            } else if (key == "MICRO_OUTPUT") {
                spec.kind = RuleKind::MicroOutput;
            } else if (key == "REGION") {
                spec.kind = RuleKind::Region;
            } else {
                spec.kind = RuleKind::NeuralField;
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
        parse_fail(origin, "MIND block must declare COUPLING, MICRO_INPUT, MICRO_OUTPUT, REGION, or NEURAL_FIELD");
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
    spec.step = block_body(source, "STEP", origin, false);

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
    } else if (spec.kind == RuleKind::Region) {
        if (spec.write.empty()) {
            parse_fail(origin, "REGION rule requires at least one WRITE exposure");
        }
        if (trim(spec.step).empty()) {
            parse_fail(origin, "REGION rule requires a STEP block");
        }
    } else if (spec.kind == RuleKind::NeuralField) {
        if (spec.state.empty()) {
            parse_fail(origin, "NEURAL_FIELD rule requires at least one STATE variable");
        }
        if (spec.write.empty()) {
            parse_fail(origin, "NEURAL_FIELD rule requires at least one WRITE exposure");
        }
        if (trim(spec.step).empty()) {
            parse_fail(origin, "NEURAL_FIELD rule requires a STEP block");
        }
    }
    if ((spec.kind == RuleKind::Coupling || spec.kind == RuleKind::Region ||
         spec.kind == RuleKind::NeuralField) && !spec.random.empty()) {
        parse_fail(origin, "RANDOM streams are currently supported only by MICRO_INPUT rules");
    }
    return spec;
}

}  // namespace mind_sim::mind_mod
