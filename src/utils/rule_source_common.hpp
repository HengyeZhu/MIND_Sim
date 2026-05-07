#pragma once

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mind_sim::utils::rule_source {

enum class TokenKind {
    Identifier,
    Literal,
    Number,
    Operator,
};

struct Token {
    TokenKind kind{TokenKind::Operator};
    std::string text{};
    int line{1};
    int column{1};
};

inline const std::unordered_set<std::string>& cpp_reserved() {
    static const std::unordered_set<std::string> values{
        "alignas", "alignof", "and", "and_eq", "asm", "atomic_cancel", "atomic_commit",
        "atomic_noexcept", "auto", "bitand", "bitor", "bool", "break", "case", "catch",
        "char", "char8_t", "char16_t", "char32_t", "class", "compl", "concept", "const",
        "consteval", "constexpr", "constinit", "const_cast", "continue", "co_await",
        "co_return", "co_yield", "decltype", "default", "delete", "do", "double",
        "dynamic_cast", "else", "enum", "explicit", "export", "extern", "false",
        "float", "for", "friend", "goto", "if", "inline", "int", "long", "mutable",
        "namespace", "new", "noexcept", "not", "not_eq", "nullptr", "operator", "or",
        "or_eq", "private", "protected", "public", "reflexpr", "register",
        "reinterpret_cast", "requires", "return", "short", "signed", "sizeof",
        "static", "static_assert", "static_cast", "struct", "switch", "synchronized",
        "template", "this", "thread_local", "throw", "true", "try", "typedef",
        "typeid", "typename", "union", "unsigned", "using", "virtual", "void",
        "volatile", "wchar_t", "while", "xor", "xor_eq",
    };
    return values;
}

inline const std::unordered_set<std::string>& forbidden_identifiers() {
    static const std::unordered_set<std::string> values{
        "asm", "catch", "class", "co_await", "co_return", "co_yield", "const_cast",
        "delete", "dynamic_cast", "extern", "friend", "goto", "namespace", "new",
        "operator", "private", "protected", "public", "reinterpret_cast", "requires",
        "static_assert", "struct", "template", "this", "thread_local", "throw", "try",
        "typedef", "typeid", "typename", "union", "using", "virtual", "malloc",
        "calloc", "realloc", "free",
    };
    return values;
}

inline const std::unordered_set<std::string>& keywords() {
    static const std::unordered_set<std::string> values{
        "if", "else", "for", "while", "do", "switch", "case", "default", "break",
        "continue", "return", "const", "true", "false", "nullptr", "static_cast",
    };
    return values;
}

inline const std::unordered_set<std::string>& type_words() {
    static const std::unordered_set<std::string> values{
        "auto", "bool", "double", "float", "int", "long", "short", "signed",
        "size_t", "unsigned", "std", "uint8_t", "int8_t", "uint16_t", "int16_t",
        "uint32_t", "int32_t", "uint64_t", "int64_t",
    };
    return values;
}

inline const std::unordered_set<std::string>& declaration_words() {
    static const std::unordered_set<std::string> values{
        "auto", "bool", "const", "double", "float", "int", "long", "short",
        "signed", "size_t", "unsigned", "uint8_t", "int8_t", "uint16_t", "int16_t",
        "uint32_t", "int32_t", "uint64_t", "int64_t",
    };
    return values;
}

inline const std::unordered_set<std::string>& helper_symbols() {
    static const std::unordered_set<std::string> values{
        "abs", "acos", "asin", "atan", "atan2", "ceil", "clamp", "cos", "cosh", "erf",
        "erfc", "exp", "expm1", "fabs", "floor", "fmax", "fmin", "isfinite", "isinf",
        "isnan", "log", "log10", "log1p", "max", "min", "pow", "round", "sigmoid",
        "sin", "sinh", "sqrt", "tan", "tanh", "uniform", "pi",
    };
    return values;
}

inline bool is_identifier_start(unsigned char c) {
    return std::isalpha(c) || c == '_';
}

inline bool is_identifier_char(unsigned char c) {
    return std::isalnum(c) || c == '_';
}

[[noreturn]] inline void fail(std::string_view kind, const Token& token, const std::string& message) {
    throw std::runtime_error(std::string(kind) + ":" + std::to_string(token.line) + ":" +
                             std::to_string(token.column) + ": " + message);
}

inline bool is_valid_identifier(const std::string& value) {
    if (value.empty() || !is_identifier_start(static_cast<unsigned char>(value.front()))) {
        return false;
    }
    for (unsigned char c: value) {
        if (!is_identifier_char(c)) {
            return false;
        }
    }
    return cpp_reserved().find(value) == cpp_reserved().end();
}

inline std::vector<std::string> names(const std::vector<std::string>& values, std::string_view what) {
    std::vector<std::string> out;
    out.reserve(values.size());
    std::unordered_set<std::string> seen;
    for (const auto& name: values) {
        if (!is_valid_identifier(name)) {
            throw std::runtime_error(std::string(what) +
                                     " name is not a valid C++ identifier: " + name);
        }
        if (!seen.insert(name).second) {
            throw std::runtime_error(std::string(what) + " names must be unique: " + name);
        }
        out.push_back(name);
    }
    return out;
}

inline std::vector<Token> scan(std::string_view source, std::string_view kind) {
    std::vector<Token> tokens;
    std::size_t index = 0;
    int line = 1;
    int column = 1;
    while (index < source.size()) {
        const char c = source[index];
        if (c == ' ' || c == '\t' || c == '\r') {
            ++index;
            ++column;
            continue;
        }
        if (c == '\n') {
            ++index;
            ++line;
            column = 1;
            continue;
        }
        if (c == '#') {
            throw std::runtime_error(std::string(kind) + ":" + std::to_string(line) + ":" +
                                     std::to_string(column) +
                                     ": preprocessor directives are not supported");
        }
        if (c == '/' && index + 1 < source.size() && source[index + 1] == '/') {
            index += 2;
            column += 2;
            while (index < source.size() && source[index] != '\n') {
                ++index;
                ++column;
            }
            continue;
        }
        if (c == '/' && index + 1 < source.size() && source[index + 1] == '*') {
            const int start_line = line;
            const int start_column = column;
            index += 2;
            column += 2;
            while (index + 1 < source.size() &&
                   !(source[index] == '*' && source[index + 1] == '/')) {
                if (source[index] == '\n') {
                    ++line;
                    column = 1;
                    ++index;
                } else {
                    ++index;
                    ++column;
                }
            }
            if (index + 1 >= source.size()) {
                throw std::runtime_error(std::string(kind) + ":" + std::to_string(start_line) +
                                         ":" + std::to_string(start_column) +
                                         ": unterminated block comment");
            }
            index += 2;
            column += 2;
            continue;
        }
        if (c == '\'' || c == '"') {
            const char quote = c;
            const int start_line = line;
            const int start_column = column;
            const auto start = index;
            ++index;
            ++column;
            bool escaped = false;
            bool closed = false;
            while (index < source.size()) {
                const char current = source[index];
                if (current == '\n') {
                    ++line;
                    column = 1;
                    ++index;
                    escaped = false;
                    continue;
                }
                if (escaped) {
                    escaped = false;
                    ++index;
                    ++column;
                    continue;
                }
                if (current == '\\') {
                    escaped = true;
                    ++index;
                    ++column;
                    continue;
                }
                if (current == quote) {
                    ++index;
                    ++column;
                    closed = true;
                    tokens.push_back(Token{
                        .kind = TokenKind::Literal,
                        .text = std::string(source.substr(start, index - start)),
                        .line = start_line,
                        .column = start_column,
                    });
                    break;
                }
                ++index;
                ++column;
            }
            if (!closed) {
                throw std::runtime_error(std::string(kind) + ":" + std::to_string(start_line) +
                                         ":" + std::to_string(start_column) +
                                         ": unterminated string literal");
            }
            continue;
        }
        if (is_identifier_start(static_cast<unsigned char>(c))) {
            const auto start = index;
            const int start_column = column;
            while (index < source.size() &&
                   is_identifier_char(static_cast<unsigned char>(source[index]))) {
                ++index;
                ++column;
            }
            tokens.push_back(Token{
                .kind = TokenKind::Identifier,
                .text = std::string(source.substr(start, index - start)),
                .line = line,
                .column = start_column,
            });
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c)) ||
            (c == '.' && index + 1 < source.size() &&
             std::isdigit(static_cast<unsigned char>(source[index + 1])))) {
            const auto start = index;
            const int start_column = column;
            ++index;
            ++column;
            while (index < source.size()) {
                const char current = source[index];
                if (std::isalnum(static_cast<unsigned char>(current)) || current == '.' ||
                    current == '_' || current == '+' || current == '-') {
                    if ((current == '+' || current == '-') &&
                        source[index - 1] != 'e' && source[index - 1] != 'E' &&
                        source[index - 1] != 'p' && source[index - 1] != 'P') {
                        break;
                    }
                    ++index;
                    ++column;
                    continue;
                }
                break;
            }
            tokens.push_back(Token{
                .kind = TokenKind::Number,
                .text = std::string(source.substr(start, index - start)),
                .line = line,
                .column = start_column,
            });
            continue;
        }

        const auto two = (index + 1 < source.size()) ? source.substr(index, 2) : std::string_view{};
        const auto three = (index + 2 < source.size()) ? source.substr(index, 3) : std::string_view{};
        if (three == "<<=" || three == ">>=") {
            tokens.push_back(Token{.kind = TokenKind::Operator,
                                   .text = std::string(three),
                                   .line = line,
                                   .column = column});
            index += 3;
            column += 3;
            continue;
        }
        if (two == "++" || two == "--" || two == "+=" || two == "-=" || two == "*=" ||
            two == "/=" || two == "%=" || two == "==" || two == "!=" || two == "<=" ||
            two == ">=" || two == "&&" || two == "||" || two == "<<" || two == ">>" ||
            two == "&=" || two == "|=" || two == "^=" || two == "::" || two == "->") {
            if (two == "->" || two == "::") {
                throw std::runtime_error(std::string(kind) + ":" + std::to_string(line) + ":" +
                                         std::to_string(column) + ": operator " +
                                         std::string(two) + " is not supported");
            }
            tokens.push_back(Token{.kind = TokenKind::Operator,
                                   .text = std::string(two),
                                   .line = line,
                                   .column = column});
            index += 2;
            column += 2;
            continue;
        }
        tokens.push_back(Token{
            .kind = TokenKind::Operator,
            .text = std::string(1, c),
            .line = line,
            .column = column,
        });
        ++index;
        ++column;
    }
    return tokens;
}

inline void check_balanced(const std::vector<Token>& tokens, std::string_view kind) {
    std::vector<Token> stack;
    for (const auto& token: tokens) {
        if (token.text == "(" || token.text == "[" || token.text == "{") {
            stack.push_back(token);
            continue;
        }
        if (token.text != ")" && token.text != "]" && token.text != "}") {
            continue;
        }
        if (stack.empty()) {
            fail(kind, token, "unmatched '" + token.text + "'");
        }
        const auto& open = stack.back().text;
        const bool ok = (token.text == ")" && open == "(") ||
                        (token.text == "]" && open == "[") ||
                        (token.text == "}" && open == "{");
        if (!ok) {
            fail(kind, token, "unmatched '" + token.text + "'");
        }
        stack.pop_back();
    }
    if (!stack.empty()) {
        fail(kind, stack.back(), "unclosed '" + stack.back().text + "'");
    }
}

inline bool looks_like_cast(const std::vector<Token>& tokens, std::size_t index) {
    return index > 0 && index + 1 < tokens.size() &&
           tokens[index - 1].text == "(" && tokens[index + 1].text == ")";
}

inline std::unordered_set<std::string> declared_locals(const std::vector<Token>& tokens,
                                                std::string_view kind) {
    std::unordered_set<std::string> locals;
    std::size_t index = 0;
    while (index < tokens.size()) {
        const auto& token = tokens[index];
        if (declaration_words().find(token.text) == declaration_words().end()) {
            ++index;
            continue;
        }
        if (looks_like_cast(tokens, index)) {
            ++index;
            continue;
        }

        const auto start = index;
        while (index < tokens.size() &&
               declaration_words().find(tokens[index].text) != declaration_words().end()) {
            ++index;
        }
        while (index < tokens.size() &&
               (tokens[index].text == "*" || tokens[index].text == "&")) {
            ++index;
        }

        bool expect_name = true;
        int depth = 0;
        while (index < tokens.size()) {
            const auto& current = tokens[index];
            if ((current.text == ";" || current.text == ")") && depth == 0) {
                break;
            }
            if (current.text == "(" || current.text == "[" || current.text == "{") {
                ++depth;
            } else if ((current.text == ")" || current.text == "]" || current.text == "}") &&
                       depth > 0) {
                --depth;
            }

            if (expect_name && depth == 0 && current.kind == TokenKind::Identifier) {
                const auto next = (index + 1 < tokens.size()) ? tokens[index + 1].text : "";
                if (next == "(") {
                    fail(kind, current, "function definitions are not supported in code snippets");
                }
                locals.insert(current.text);
                expect_name = false;
            } else if (depth == 0 && current.text == ",") {
                expect_name = true;
            }
            ++index;
        }
        if (index == start) {
            ++index;
        }
    }
    return locals;
}

inline bool contains(const std::unordered_set<std::string>& values, const std::string& key) {
    return values.find(key) != values.end();
}

inline std::unordered_set<std::string> member_set(const std::vector<std::string>& values) {
    std::unordered_set<std::string> out;
    out.reserve(values.size());
    for (const auto& value: values) {
        out.insert(value);
    }
    return out;
}

inline std::unordered_set<std::string> member_set(std::initializer_list<const char*> values) {
    std::unordered_set<std::string> out;
    out.reserve(values.size());
    for (const char* value: values) {
        out.emplace(value);
    }
    return out;
}

inline std::string common_helpers() {
    return R"cpp(
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

using std::abs;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::ceil;
using std::clamp;
using std::cos;
using std::cosh;
using std::erf;
using std::erfc;
using std::exp;
using std::expm1;
using std::fabs;
using std::floor;
using std::fmax;
using std::fmin;
using std::isfinite;
using std::isinf;
using std::isnan;
using std::log;
using std::log10;
using std::log1p;
using std::max;
using std::min;
using std::pow;
using std::round;
using std::size_t;
using std::sin;
using std::sinh;
using std::sqrt;
using std::tan;
using std::tanh;

constexpr double pi = 3.141592653589793238462643383279502884;

template <typename Value, typename Low, typename High>
static inline Value clamp(Value value, Low low, High high) {
    const Value lo = static_cast<Value>(low);
    const Value hi = static_cast<Value>(high);
    return value < lo ? lo : (value > hi ? hi : value);
}

static inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

)cpp";
}

inline int schema_field_index(const std::vector<std::string>& fields, const std::string& field) {
    const auto found = std::find(fields.begin(), fields.end(), field);
    if (found == fields.end()) {
        throw std::runtime_error("internal codegen schema lookup failed for field: " + field);
    }
    return static_cast<int>(found - fields.begin());
}

inline std::string indent_line(const std::string& line, int indent) {
    return std::string(static_cast<std::size_t>(indent), ' ') + line;
}

inline std::string indent_block(const std::string& block, int indent) {
    std::ostringstream out;
    std::istringstream in(block);
    std::string line;
    while (std::getline(in, line)) {
        out << indent_line(line, indent) << "\n";
    }
    return out.str();
}

inline void require_declared_fields_used(std::string_view kind,
                                  std::string_view object,
                                  const std::vector<std::string>& used,
                                  const std::vector<std::string>& declared) {
    if (used.size() == declared.size()) {
        return;
    }
    const auto used_set = member_set(used);
    for (const auto& field: declared) {
        if (!contains(used_set, field)) {
            throw std::runtime_error(std::string(kind) + " declared " +
                                     std::string(object) + " is unused: " + field);
        }
    }
}

inline bool has_field(const std::vector<std::string>& fields, const std::string& field) {
    return std::find(fields.begin(), fields.end(), field) != fields.end();
}

}  // namespace mind_sim::utils::rule_source
