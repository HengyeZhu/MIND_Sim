#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

#ifndef MIND_SIM_NMODL_PATH
#error "MIND_SIM_NMODL_PATH must point to the MIND-aware NMODL built by CMake"
#endif

#ifndef MIND_SIM_CXX_COMPILER
#define MIND_SIM_CXX_COMPILER "c++"
#endif

#ifndef MIND_SIM_NRNIVMODL_EXECUTABLE
#define MIND_SIM_NRNIVMODL_EXECUTABLE "nrnivmodl"
#endif

#ifndef MIND_SIM_SOURCE_DIR
#define MIND_SIM_SOURCE_DIR "."
#endif

#ifndef MIND_SIM_MECHANISM_INCLUDE_DIR
#define MIND_SIM_MECHANISM_INCLUDE_DIR "."
#endif

#ifndef MIND_SIM_MOD_CXX_FLAGS
#define MIND_SIM_MOD_CXX_FLAGS ""
#endif

#ifndef MIND_SIM_MOD_LINK_FLAGS
#define MIND_SIM_MOD_LINK_FLAGS ""
#endif

namespace {

struct Variable {
    std::string name;
    std::string default_value{"0.0"};
    int array_size{1};
};

struct MindSpec {
    std::string role;
    std::vector<std::string> target_inputs;
    std::vector<std::string> source_exposures;
};

struct Args {
    fs::path source;
};

struct MechanismFieldLayout {
    std::string cpp_type;
    std::string name;
    std::string semantic;
    std::string role{"range"};
    int array_size{1};
};

struct MechanismDataLayout {
    std::string mechanism;
    std::vector<std::string> defaults;
    std::vector<MechanismFieldLayout> data_fields;
    std::vector<MechanismFieldLayout> dparam_fields;
};

struct IonUse {
    std::string ion;
    std::vector<std::string> reads;
    std::vector<std::string> writes;
};

struct NeuronBlockSpec {
    std::string mechanism;
    bool point_process{false};
    bool artificial_cell{false};
    std::set<std::string> range_vars;
    std::vector<IonUse> ions;
    std::vector<std::string> pointers;
    std::vector<std::string> bbcore_pointers;
    std::vector<std::string> randoms;
    std::set<std::string> current_vars;
};

[[nodiscard]] bool contains_name(const std::vector<std::string>& names, const std::string& name);
void append_unique(std::vector<std::string>& names, std::string name);
[[nodiscard]] bool is_external_neuron_scalar(const std::string& name);
void add_data_field(std::vector<MechanismFieldLayout>& fields,
                    const std::string& name,
                    std::string role,
                    int array_size = 1);
void add_dparam_field(std::vector<MechanismFieldLayout>& fields,
                      std::string cpp_type,
                      std::string name,
                      std::string semantic);
void add_ion_dparams(std::vector<MechanismFieldLayout>& fields, const IonUse& ion);

struct JsonValue {
    enum class Kind { Null, Bool, Number, String, Array, Object };

    Kind kind{Kind::Null};
    bool boolean{false};
    std::string text;
    std::vector<JsonValue> array;
    std::map<std::string, JsonValue> object;

    [[nodiscard]] bool is_array() const {
        return kind == Kind::Array;
    }

    [[nodiscard]] bool is_object() const {
        return kind == Kind::Object;
    }

    [[nodiscard]] bool is_string() const {
        return kind == Kind::String;
    }
};

class JsonParser {
  public:
    explicit JsonParser(std::string input)
        : text_(std::move(input)) {}

    [[nodiscard]] JsonValue parse() {
        JsonValue value = parse_value();
        skip_space();
        if (pos_ != text_.size()) {
            throw std::runtime_error("unexpected trailing data in JSON");
        }
        return value;
    }

  private:
    std::string text_;
    std::size_t pos_{0};

    void skip_space() {
        while (pos_ < text_.size() &&
               std::isspace(static_cast<unsigned char>(text_[pos_])) != 0) {
            ++pos_;
        }
    }

    void require(char expected) {
        skip_space();
        if (pos_ >= text_.size() || text_[pos_] != expected) {
            throw std::runtime_error(std::string("expected JSON character ") + expected);
        }
        ++pos_;
    }

    [[nodiscard]] bool consume(std::string_view token) {
        skip_space();
        if (text_.compare(pos_, token.size(), token) != 0) {
            return false;
        }
        pos_ += token.size();
        return true;
    }

    [[nodiscard]] JsonValue parse_value() {
        skip_space();
        if (pos_ >= text_.size()) {
            throw std::runtime_error("unexpected end of JSON");
        }
        const char c = text_[pos_];
        if (c == '{') {
            return parse_object();
        }
        if (c == '[') {
            return parse_array();
        }
        if (c == '"') {
            JsonValue value;
            value.kind = JsonValue::Kind::String;
            value.text = parse_string();
            return value;
        }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c)) != 0) {
            return parse_number();
        }
        if (consume("true")) {
            JsonValue value;
            value.kind = JsonValue::Kind::Bool;
            value.boolean = true;
            return value;
        }
        if (consume("false")) {
            JsonValue value;
            value.kind = JsonValue::Kind::Bool;
            value.boolean = false;
            return value;
        }
        if (consume("null")) {
            return {};
        }
        throw std::runtime_error("invalid JSON value");
    }

    [[nodiscard]] JsonValue parse_object() {
        JsonValue value;
        value.kind = JsonValue::Kind::Object;
        require('{');
        skip_space();
        if (pos_ < text_.size() && text_[pos_] == '}') {
            ++pos_;
            return value;
        }
        while (true) {
            skip_space();
            if (pos_ >= text_.size() || text_[pos_] != '"') {
                throw std::runtime_error("expected JSON object key");
            }
            std::string key = parse_string();
            require(':');
            value.object.emplace(std::move(key), parse_value());
            skip_space();
            if (pos_ < text_.size() && text_[pos_] == '}') {
                ++pos_;
                return value;
            }
            require(',');
        }
    }

    [[nodiscard]] JsonValue parse_array() {
        JsonValue value;
        value.kind = JsonValue::Kind::Array;
        require('[');
        skip_space();
        if (pos_ < text_.size() && text_[pos_] == ']') {
            ++pos_;
            return value;
        }
        while (true) {
            value.array.push_back(parse_value());
            skip_space();
            if (pos_ < text_.size() && text_[pos_] == ']') {
                ++pos_;
                return value;
            }
            require(',');
        }
    }

    [[nodiscard]] std::string parse_string() {
        require('"');
        std::string out;
        while (pos_ < text_.size()) {
            const char c = text_[pos_++];
            if (c == '"') {
                return out;
            }
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos_ >= text_.size()) {
                throw std::runtime_error("unfinished JSON escape");
            }
            const char escaped = text_[pos_++];
            switch (escaped) {
            case '"':
            case '\\':
            case '/':
                out += escaped;
                break;
            case 'b':
                out += '\b';
                break;
            case 'f':
                out += '\f';
                break;
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;
            case 'u':
                if (pos_ + 4 > text_.size()) {
                    throw std::runtime_error("unfinished JSON unicode escape");
                }
                out += '?';
                pos_ += 4;
                break;
            default:
                throw std::runtime_error("invalid JSON escape");
            }
        }
        throw std::runtime_error("unfinished JSON string");
    }

    [[nodiscard]] JsonValue parse_number() {
        const std::size_t start = pos_;
        if (text_[pos_] == '-') {
            ++pos_;
        }
        while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_])) != 0) {
            ++pos_;
        }
        if (pos_ < text_.size() && text_[pos_] == '.') {
            ++pos_;
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_])) != 0) {
                ++pos_;
            }
        }
        if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
            ++pos_;
            if (pos_ < text_.size() && (text_[pos_] == '+' || text_[pos_] == '-')) {
                ++pos_;
            }
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_])) != 0) {
                ++pos_;
            }
        }
        JsonValue value;
        value.kind = JsonValue::Kind::Number;
        value.text = text_.substr(start, pos_ - start);
        return value;
    }
};

[[nodiscard]] std::string read_text(const fs::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("failed to read " + path.string());
    }
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

void write_text(const fs::path& path, const std::string& text) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to write " + path.string());
    }
    out << text;
}

[[nodiscard]] JsonValue read_json(const fs::path& path) {
    return JsonParser(read_text(path)).parse();
}

[[nodiscard]] std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
                          return std::isspace(c) != 0;
                      }).base();
    if (first >= last) {
        return {};
    }
    return std::string(first, last);
}

[[nodiscard]] std::string upper(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

[[nodiscard]] std::string shell_quote(const fs::path& path) {
    std::string text = path.string();
    std::string out{"'"};
    for (char c : text) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

void run(const std::vector<std::string>& command) {
    std::ostringstream shell;
    for (std::size_t i = 0; i < command.size(); ++i) {
        if (i != 0) {
            shell << ' ';
        }
        shell << command[i];
    }
    const int status = std::system(shell.str().c_str());
    if (status != 0) {
        throw std::runtime_error("command failed: " + shell.str());
    }
}

void run_in_dir(const fs::path& directory, const std::vector<std::string>& command) {
    std::ostringstream shell;
    shell << "cd " << shell_quote(directory) << " && ";
    for (std::size_t i = 0; i < command.size(); ++i) {
        if (i != 0) {
            shell << ' ';
        }
        shell << command[i];
    }
    const int status = std::system(shell.str().c_str());
    if (status != 0) {
        throw std::runtime_error("command failed: " + shell.str());
    }
}

[[nodiscard]] std::vector<std::string> split_names(std::string_view text) {
    std::vector<std::string> out;
    std::string current;
    for (char c : text) {
        if (std::isspace(static_cast<unsigned char>(c)) || c == ',') {
            if (!current.empty()) {
                out.push_back(current);
                current.clear();
            }
        } else {
            current += c;
        }
    }
    if (!current.empty()) {
        out.push_back(current);
    }
    return out;
}

[[nodiscard]] std::string cpp_string(std::string_view value) {
    std::string out{"\""};
    for (char c : value) {
        if (c == '\\') {
            out += "\\\\";
        } else if (c == '"') {
            out += "\\\"";
        } else {
            out += c;
        }
    }
    out += '"';
    return out;
}

[[nodiscard]] const JsonValue* object_field(const JsonValue& value, const std::string& key) {
    if (!value.is_object()) {
        return nullptr;
    }
    const auto found = value.object.find(key);
    return found == value.object.end() ? nullptr : &found->second;
}

void collect_node_payloads(const JsonValue& value,
                           const std::string& key,
                           std::vector<const JsonValue*>& out) {
    if (value.is_object()) {
        if (const auto* payload = object_field(value, key)) {
            out.push_back(payload);
        }
        for (const auto& [_, child] : value.object) {
            collect_node_payloads(child, key, out);
        }
    } else if (value.is_array()) {
        for (const auto& child : value.array) {
            collect_node_payloads(child, key, out);
        }
    }
}

[[nodiscard]] std::vector<const JsonValue*> node_payloads(const JsonValue& ast,
                                                          const std::string& key) {
    std::vector<const JsonValue*> out;
    collect_node_payloads(ast, key, out);
    return out;
}

void collect_string_names(const JsonValue& value, std::vector<std::string>& out) {
    if (value.is_object()) {
        if (const auto* name = object_field(value, "name"); name != nullptr && name->is_string()) {
            out.push_back(name->text);
            return;
        }
        for (const auto& [_, child] : value.object) {
            collect_string_names(child, out);
        }
    } else if (value.is_array()) {
        for (const auto& child : value.array) {
            collect_string_names(child, out);
        }
    }
}

[[nodiscard]] std::vector<std::string> string_names(const JsonValue& value) {
    std::vector<std::string> out;
    collect_string_names(value, out);
    return out;
}

[[nodiscard]] std::optional<std::string> first_string_name(const JsonValue& value) {
    auto names = string_names(value);
    if (names.empty()) {
        return std::nullopt;
    }
    return names.front();
}

[[nodiscard]] std::optional<std::string> first_numeric_literal(const JsonValue& value) {
    if (value.is_object()) {
        for (const auto& key : {"Double", "Integer"}) {
            if (const auto* payload = object_field(value, key)) {
                if (auto name = first_string_name(*payload)) {
                    return name;
                }
            }
        }
        for (const auto& [_, child] : value.object) {
            if (auto number = first_numeric_literal(child)) {
                return number;
            }
        }
    } else if (value.is_array()) {
        for (const auto& child : value.array) {
            if (auto number = first_numeric_literal(child)) {
                return number;
            }
        }
    }
    return std::nullopt;
}

void collect_named_children(const JsonValue& value,
                            const std::string& key,
                            std::vector<std::string>& out) {
    if (value.is_object()) {
        if (const auto* payload = object_field(value, key)) {
            if (auto name = first_string_name(*payload)) {
                append_unique(out, *name);
            }
        }
        for (const auto& [_, child] : value.object) {
            collect_named_children(child, key, out);
        }
    } else if (value.is_array()) {
        for (const auto& child : value.array) {
            collect_named_children(child, key, out);
        }
    }
}

[[nodiscard]] std::vector<std::string> named_children(const JsonValue& value,
                                                      const std::string& key) {
    std::vector<std::string> out;
    collect_named_children(value, key, out);
    return out;
}

[[nodiscard]] std::vector<Variable> variables_from_ast_block(const JsonValue& ast,
                                                             const std::string& block_key,
                                                             const std::string& item_key,
                                                             bool read_default) {
    std::vector<Variable> out;
    for (const auto* block : node_payloads(ast, block_key)) {
        if (!block->is_array()) {
            continue;
        }
        for (const auto& item : block->array) {
            const JsonValue* payload = object_field(item, item_key);
            if (payload == nullptr) {
                payload = &item;
            }
            if (auto name = first_string_name(*payload)) {
                out.push_back(Variable{
                    .name = *name,
                    .default_value = read_default ? first_numeric_literal(*payload).value_or("0.0") : "0.0",
                    .array_size = 1,
                });
            }
        }
    }
    return out;
}

[[nodiscard]] std::vector<Variable> params_from_ast(const JsonValue& ast) {
    return variables_from_ast_block(ast, "ParamBlock", "ParamAssign", true);
}

[[nodiscard]] std::vector<Variable> assigned_from_ast(const JsonValue& ast) {
    return variables_from_ast_block(ast, "AssignedBlock", "AssignedDefinition", false);
}

[[nodiscard]] std::vector<Variable> states_from_ast(const JsonValue& ast) {
    return variables_from_ast_block(ast, "StateBlock", "AssignedDefinition", false);
}

[[nodiscard]] std::optional<std::string> binary_operator(const JsonValue& value) {
    if (const auto* payload = object_field(value, "BinaryOperator")) {
        return first_string_name(*payload);
    }
    return std::nullopt;
}

void collect_initial_defaults(const JsonValue& value, std::map<std::string, std::string>& defaults) {
    if (const auto* expression = object_field(value, "BinaryExpression");
        expression != nullptr && expression->is_array() && expression->array.size() >= 3) {
        const auto op = binary_operator(expression->array[1]);
        if (op && *op == "=") {
            const auto lhs_names = named_children(expression->array[0], "VarName");
            const auto rhs_value = first_numeric_literal(expression->array[2]);
            if (!lhs_names.empty() && rhs_value) {
                defaults[lhs_names.front()] = *rhs_value;
            }
        }
    }
    if (value.is_object()) {
        for (const auto& [_, child] : value.object) {
            collect_initial_defaults(child, defaults);
        }
    } else if (value.is_array()) {
        for (const auto& child : value.array) {
            collect_initial_defaults(child, defaults);
        }
    }
}

void apply_initial_defaults_from_ast(const JsonValue& ast, std::vector<Variable>& states) {
    std::map<std::string, std::string> defaults;
    for (const auto* block : node_payloads(ast, "InitialBlock")) {
        collect_initial_defaults(*block, defaults);
    }
    for (auto& state : states) {
        if (const auto found = defaults.find(state.name); found != defaults.end()) {
            state.default_value = found->second;
        }
    }
}

[[nodiscard]] std::string mind_block_source_from_ast(const JsonValue& ast) {
    const auto blocks = node_payloads(ast, "MindBlock");
    if (blocks.empty()) {
        return {};
    }
    auto names = string_names(*blocks.front());
    if (names.empty()) {
        throw std::runtime_error("MIND block AST has no text payload");
    }
    return names.front();
}

[[nodiscard]] MindSpec parse_mind_block_ast(const JsonValue& ast) {
    const std::string text = mind_block_source_from_ast(ast);
    if (text.empty()) {
        throw std::runtime_error("MOD file is missing a MIND block");
    }
    const std::size_t begin = text.find('{');
    const std::size_t end = text.rfind('}');
    if (begin == std::string::npos || end == std::string::npos || begin >= end) {
        throw std::runtime_error("MIND block AST payload is malformed");
    }

    MindSpec spec;
    std::istringstream lines(text.substr(begin + 1, end - begin - 1));
    std::string raw;
    while (std::getline(lines, raw)) {
        const std::string line = trim(raw);
        if (line.empty()) {
            continue;
        }
        auto words = split_names(line);
        if (words.empty()) {
            continue;
        }
        const std::string key = upper(words.front());
        words.erase(words.begin());
        if (key == "ROLE" && !words.empty()) {
            spec.role = upper(words.front());
        } else if (key == "TARGET_INPUT") {
            spec.target_inputs.insert(spec.target_inputs.end(), words.begin(), words.end());
        } else if (key == "SOURCE_EXPOSURE") {
            spec.source_exposures.insert(spec.source_exposures.end(), words.begin(), words.end());
        } else {
            throw std::runtime_error("unknown MIND key: " + key);
        }
    }
    if (spec.role.empty()) {
        throw std::runtime_error("MIND block is missing ROLE");
    }
    return spec;
}

[[nodiscard]] bool contains_mind_block_ast(const JsonValue& ast) {
    return !node_payloads(ast, "MindBlock").empty();
}

[[nodiscard]] NeuronBlockSpec parse_neuron_block_ast(const JsonValue& ast) {
    const auto blocks = node_payloads(ast, "NeuronBlock");
    if (blocks.empty()) {
        throw std::runtime_error("MOD AST is missing a NEURON block");
    }
    const JsonValue& block = *blocks.front();

    NeuronBlockSpec spec;
    for (const auto* suffix : node_payloads(block, "Suffix")) {
        const auto names = string_names(*suffix);
        if (names.size() >= 2) {
            const std::string key = upper(names[0]);
            spec.mechanism = names[1];
            spec.point_process = key == "POINT_PROCESS" || key == "ARTIFICIAL_CELL";
            spec.artificial_cell = key == "ARTIFICIAL_CELL";
        }
    }
    for (const auto* range : node_payloads(block, "Range")) {
        for (const auto& name : named_children(*range, "RangeVar")) {
            spec.range_vars.insert(name);
        }
    }
    for (const auto* current : node_payloads(block, "Nonspecific")) {
        for (const auto& name : named_children(*current, "NonspecificCurVar")) {
            spec.current_vars.insert(name);
        }
    }
    for (const auto* current : node_payloads(block, "ElectrodeCurrent")) {
        for (const auto& name : named_children(*current, "ElectrodeCurVar")) {
            spec.current_vars.insert(name);
        }
    }
    for (const auto* randoms : node_payloads(block, "RandomVarList")) {
        for (const auto& name : named_children(*randoms, "RandomVar")) {
            append_unique(spec.randoms, name);
        }
    }
    for (const auto* pointers : node_payloads(block, "Pointer")) {
        for (const auto& name : named_children(*pointers, "PointerVar")) {
            append_unique(spec.pointers, name);
        }
    }
    for (const auto* pointers : node_payloads(block, "BbcorePointer")) {
        for (const auto& name : named_children(*pointers, "BbcorePointerVar")) {
            append_unique(spec.bbcore_pointers, name);
        }
    }
    for (const auto* useion : node_payloads(block, "Useion")) {
        IonUse ion;
        if (const auto name = first_string_name(*useion)) {
            ion.ion = *name;
        }
        for (const auto& name : named_children(*useion, "ReadIonVar")) {
            append_unique(ion.reads, name);
        }
        for (const auto& name : named_children(*useion, "WriteIonVar")) {
            append_unique(ion.writes, name);
        }
        if (!ion.ion.empty()) {
            spec.ions.push_back(std::move(ion));
        }
    }
    if (spec.mechanism.empty()) {
        throw std::runtime_error("NEURON block must contain SUFFIX, POINT_PROCESS, or ARTIFICIAL_CELL");
    }
    return spec;
}

[[nodiscard]] bool has_ast_node(const JsonValue& ast, const std::string& key) {
    return !node_payloads(ast, key).empty();
}

[[nodiscard]] bool has_ast_name(const JsonValue& value, const std::string& name) {
    const auto names = string_names(value);
    return std::find(names.begin(), names.end(), name) != names.end();
}

[[nodiscard]] std::vector<std::string> net_receive_args_from_ast(const JsonValue& ast) {
    const auto blocks = node_payloads(ast, "NetReceiveBlock");
    if (blocks.empty()) {
        return {};
    }
    std::vector<std::string> out;
    for (const auto& name : named_children(*blocks.front(), "Argument")) {
        append_unique(out, name);
    }
    return out;
}

void validate_micro_input_net_receive_ast(const JsonValue& ast) {
    const auto args = net_receive_args_from_ast(ast);
    const std::vector<std::string> expected{"weight"};
    if (args != expected) {
        throw std::runtime_error("MACRO2MICRO requires NET_RECEIVE(weight)");
    }
}

void validate_micro_output_net_receive_ast(const JsonValue& ast) {
    const auto args = net_receive_args_from_ast(ast);
    const std::vector<std::string> expected{"weight", "gid"};
    if (args != expected) {
        throw std::runtime_error("MICRO2MACRO requires NET_RECEIVE(weight, gid)");
    }
}

[[nodiscard]] MechanismDataLayout data_layout_from_ast(const JsonValue& ast) {
    const auto spec = parse_neuron_block_ast(ast);
    const auto params = params_from_ast(ast);
    const auto assigned = assigned_from_ast(ast);
    const auto states = states_from_ast(ast);

    MechanismDataLayout layout{.mechanism = spec.mechanism};
    for (const auto& param : params) {
        if (is_external_neuron_scalar(param.name)) {
            continue;
        }
        add_data_field(layout.data_fields, param.name, "parameter", param.array_size);
        layout.defaults.push_back(param.default_value);
    }
    for (const auto& variable : assigned) {
        if (variable.name == "v" || is_external_neuron_scalar(variable.name) ||
            (!spec.range_vars.contains(variable.name) && !spec.current_vars.contains(variable.name))) {
            continue;
        }
        add_data_field(layout.data_fields, variable.name, "assigned", variable.array_size);
    }
    for (const auto& state : states) {
        add_data_field(layout.data_fields, state.name, "state", state.array_size);
    }
    for (const auto& variable : assigned) {
        if (variable.name == "v" || is_external_neuron_scalar(variable.name) ||
            spec.range_vars.contains(variable.name) || spec.current_vars.contains(variable.name)) {
            continue;
        }
        add_data_field(layout.data_fields, variable.name, "assigned", variable.array_size);
    }
    for (const auto& state : states) {
        add_data_field(layout.data_fields, "D" + state.name, "range", state.array_size);
    }
    for (const auto& ion : spec.ions) {
        const std::string interior = ion.ion + "i";
        const std::string exterior = ion.ion + "o";
        const std::string erev = "e" + ion.ion;
        for (const auto& read : ion.reads) {
            if (read == erev || read == interior || read == exterior) {
                add_data_field(layout.data_fields, read, "assigned");
            }
        }
    }
    for (const auto& ion : spec.ions) {
        const std::string interior = ion.ion + "i";
        const std::string exterior = ion.ion + "o";
        for (const auto& write : ion.writes) {
            if (write != interior && write != exterior) {
                add_data_field(layout.data_fields, write, "assigned");
            }
        }
    }
    if (spec.artificial_cell) {
        add_data_field(layout.data_fields, "v_unused", "range");
    } else if (spec.point_process) {
        add_data_field(layout.data_fields, "v_unused", "range");
        add_data_field(layout.data_fields, "g_unused", "range");
    } else {
        add_data_field(layout.data_fields, "v_unused", "range");
        add_data_field(layout.data_fields, "g_unused", "range");
    }
    if (spec.point_process && has_ast_node(ast, "NetReceiveBlock")) {
        add_data_field(layout.data_fields, "tsave", "range");
    }

    if (spec.point_process) {
        add_dparam_field(layout.dparam_fields, "double*", "node_area", "area");
        add_dparam_field(layout.dparam_fields, "Point_process*", "point_process", "pntproc");
    }
    for (const auto& ion : spec.ions) {
        add_ion_dparams(layout.dparam_fields, ion);
    }
    for (const auto& pointer : spec.pointers) {
        add_dparam_field(layout.dparam_fields, "double*", pointer, "pointer");
    }
    for (const auto& pointer : spec.bbcore_pointers) {
        add_dparam_field(layout.dparam_fields, "double*", pointer, "bbcorepointer");
    }
    for (const auto& random : spec.randoms) {
        add_dparam_field(layout.dparam_fields, "double*", random, "random");
    }
    if (has_ast_name(ast, "diam")) {
        add_dparam_field(layout.dparam_fields, "double*", "diam", "diam");
    }
    if (has_ast_name(ast, "area")) {
        add_dparam_field(layout.dparam_fields, "double*", "area", "area");
    }
    if (has_ast_name(ast, "net_send")) {
        add_dparam_field(layout.dparam_fields, "void*", "tqitem", "netsend");
    }
    if (has_ast_node(ast, "DerivativeBlock") || has_ast_node(ast, "KineticBlock")) {
        add_dparam_field(layout.dparam_fields, "int", "_cvode_ieq", "cvodeieq");
    }
    if (layout.data_fields.empty()) {
        throw std::runtime_error("could not generate mechanism layout from NMODL AST");
    }
    return layout;
}

[[nodiscard]] bool contains_name(const std::vector<std::string>& names, const std::string& name) {
    return std::find(names.begin(), names.end(), name) != names.end();
}

void append_unique(std::vector<std::string>& names, std::string name) {
    if (!name.empty() && !contains_name(names, name)) {
        names.push_back(std::move(name));
    }
}

[[nodiscard]] bool is_external_neuron_scalar(const std::string& name) {
    return name == "celsius" || name == "dt" || name == "t" || name == "diam" ||
           name == "area" || name == "PI";
}

void add_data_field(std::vector<MechanismFieldLayout>& fields,
                    const std::string& name,
                    std::string role,
                    int array_size) {
    if (name.empty() || is_external_neuron_scalar(name)) {
        return;
    }
    if (std::find_if(fields.begin(), fields.end(), [&](const auto& field) {
            return field.name == name;
        }) != fields.end()) {
        return;
    }
    fields.push_back(MechanismFieldLayout{
        .cpp_type = "double",
        .name = name,
        .role = std::move(role),
        .array_size = array_size,
    });
}

void add_dparam_field(std::vector<MechanismFieldLayout>& fields,
                      std::string cpp_type,
                      std::string name,
                      std::string semantic) {
    if (name.empty()) {
        return;
    }
    fields.push_back(MechanismFieldLayout{
        .cpp_type = std::move(cpp_type),
        .name = std::move(name),
        .semantic = std::move(semantic),
    });
}

[[nodiscard]] std::string current_derivative_name(const std::string& current) {
    if (current.size() >= 2 && current[0] == 'i') {
        return "d" + current + "dv";
    }
    return "d" + current + "_dv";
}

[[nodiscard]] bool ion_writes_concentration(const IonUse& ion) {
    const std::string interior = ion.ion + "i";
    const std::string exterior = ion.ion + "o";
    return contains_name(ion.writes, interior) || contains_name(ion.writes, exterior);
}

void add_ion_dparams(std::vector<MechanismFieldLayout>& fields, const IonUse& ion) {
    const std::string semantic = ion.ion + "_ion";
    const std::string interior = ion.ion + "i";
    const std::string exterior = ion.ion + "o";
    const std::string erev = "e" + ion.ion;

    for (const auto& name : ion.writes) {
        if (name == interior || name == exterior) {
            continue;
        }
        add_dparam_field(fields, "double*", "_ion_" + name, semantic);
        add_dparam_field(fields, "double*", "_ion_" + current_derivative_name(name), semantic);
    }

    if (ion_writes_concentration(ion)) {
        for (const auto& name : ion.reads) {
            if (name != interior && name != exterior) {
                add_dparam_field(fields, "double*", "_ion_" + name, semantic);
            }
        }
        if (contains_name(ion.reads, interior)) {
            add_dparam_field(fields, "double*", "_ion_" + interior, semantic);
            add_dparam_field(fields, "double*", "_ion_" + exterior, semantic);
        } else {
            add_dparam_field(fields, "double*", "_ion_" + exterior, semantic);
            add_dparam_field(fields, "double*", "_ion_" + interior, semantic);
        }
        add_dparam_field(fields, "double*", "_ion_" + ion.ion + "_erev", semantic);
        add_dparam_field(fields, "int*", "_style_" + ion.ion, "#" + semantic);
        return;
    }

    for (const auto& name : ion.reads) {
        if (name == interior) {
            add_dparam_field(fields, "double*", "_ion_" + interior, semantic);
            add_dparam_field(fields, "double*", "_ion_" + exterior, semantic);
        } else if (name == exterior) {
            add_dparam_field(fields, "double*", "_ion_" + exterior, semantic);
        } else if (name == erev) {
            add_dparam_field(fields, "double*", "_ion_" + ion.ion + "_erev", semantic);
        } else {
            add_dparam_field(fields, "double*", "_ion_" + name, semantic);
        }
    }
}

[[nodiscard]] std::string name_array(const std::string& symbol,
                                     const std::vector<std::string>& values) {
    if (values.empty()) {
        return {};
    }
    std::ostringstream out;
    out << "constexpr const char* " << symbol << "[] = {";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << cpp_string(values[i]);
    }
    out << "};\n";
    return out.str();
}

[[nodiscard]] std::string default_array(const std::string& symbol,
                                        const std::vector<Variable>& values) {
    if (values.empty()) {
        return {};
    }
    std::ostringstream out;
    out << "constexpr double " << symbol << "[] = {";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ", ";
        }
        out << values[i].default_value;
    }
    out << "};\n";
    return out.str();
}

template <typename T>
[[nodiscard]] const char* array_ref(const std::vector<T>& values, const char* symbol) {
    return values.empty() ? "nullptr" : symbol;
}

[[nodiscard]] std::string role_kind(const std::string& role) {
    if (role == "REGION") {
        return "Region";
    }
    if (role == "MACRO2MACRO") {
        return "MacroToMacro";
    }
    if (role == "MACRO2MICRO") {
        return "MicroInput";
    }
    if (role == "MICRO2MACRO") {
        return "MicroOutput";
    }
    throw std::runtime_error("unsupported MIND ROLE " + role);
}

[[nodiscard]] std::vector<std::string> names_of(const std::vector<Variable>& values) {
    std::vector<std::string> out;
    out.reserve(values.size());
    for (const auto& value : values) {
        out.push_back(value.name);
    }
    return out;
}

struct DescriptorNames {
    std::vector<std::string> target_inputs;
    std::vector<std::string> source_exposures;
};

[[nodiscard]] DescriptorNames descriptor_names(const MindSpec& spec) {
    if (spec.role == "REGION") {
        return {.target_inputs = spec.target_inputs, .source_exposures = spec.source_exposures};
    }
    if (spec.role == "MACRO2MACRO") {
        return {.target_inputs = spec.target_inputs, .source_exposures = spec.source_exposures};
    }
    if (spec.role == "MACRO2MICRO") {
        return {.target_inputs = spec.target_inputs, .source_exposures = {}};
    }
    if (spec.role == "MICRO2MACRO") {
        return {.target_inputs = {}, .source_exposures = spec.source_exposures};
    }
    throw std::runtime_error("unsupported MIND ROLE " + spec.role);
}

[[nodiscard]] std::map<std::string, int> data_offsets_from_layout(
    const MechanismDataLayout& layout) {
    std::map<std::string, int> offsets;
    int offset = 0;
    for (const auto& field : layout.data_fields) {
        offsets.emplace(field.name, offset);
        offset += field.array_size;
    }
    if (offsets.empty()) {
        throw std::runtime_error("mechanism layout has no data fields for " + layout.mechanism);
    }
    return offsets;
}

[[nodiscard]] int data_field_count(const MechanismDataLayout& layout) {
    int count = 0;
    for (const auto& field : layout.data_fields) {
        count += field.array_size;
    }
    return count;
}

[[nodiscard]] std::map<std::string, int> datum_semantic_offsets_from_layout(
    const MechanismDataLayout& layout) {
    std::map<std::string, int> offsets;
    int offset = 0;
    for (const auto& field : layout.dparam_fields) {
        if (!field.semantic.empty() && !offsets.contains(field.semantic)) {
            offsets.emplace(field.semantic, offset);
        }
        ++offset;
    }
    return offsets;
}

void require_offsets(const std::map<std::string, int>& offsets,
                     const std::vector<std::string>& names,
                     const std::string& what) {
    std::vector<std::string> missing;
    for (const auto& name : names) {
        if (!offsets.contains(name)) {
            missing.push_back(name);
        }
    }
    if (!missing.empty()) {
        std::ostringstream out;
        out << what << " variables are not mechanism data fields:";
        for (const auto& name : missing) {
            out << ' ' << name;
        }
        throw std::runtime_error(out.str());
    }
}

[[nodiscard]] std::string offset_map(const std::string& symbol,
                                     const std::vector<std::string>& names,
                                     const std::map<std::string, int>& offsets) {
    std::ostringstream out;
    out << "constexpr int " << symbol << "[] = {";
    if (names.empty()) {
        out << "0";
    } else {
        for (std::size_t i = 0; i < names.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            out << offsets.at(names[i]);
        }
    }
    out << "};\n";
    return out.str();
}

[[nodiscard]] std::string render_descriptor(const std::string& model,
                                            const MindSpec& spec,
                                            const std::vector<Variable>& states,
                                            const std::vector<Variable>& params) {
    const auto descriptor = descriptor_names(spec);
    const auto state_names = names_of(states);
    const auto param_names = names_of(params);
    const std::string kind = role_kind(spec.role);

    std::ostringstream out;
    out << name_array("kTargetInputNames", descriptor.target_inputs);
    out << name_array("kSourceExposureNames", descriptor.source_exposures);
    out << name_array("kParamNames", param_names);
    out << name_array("kStateNames", state_names);
    out << default_array("kParamDefaults", params);
    out << default_array("kStateDefaults", states);
    out << "\nconstexpr mind_sim::mod::AbiRuleDescriptor kDescriptor{\n";
    out << "    .abi_version = mind_sim::mod::kModAbiVersion,\n";
    out << "    .kind = static_cast<int>(mind_sim::mod::AbiRuleKind::" << kind
        << "),\n";
    out << "    .name = " << cpp_string(model) << ",\n";
    out << "    .target_input_count = " << descriptor.target_inputs.size() << ",\n";
    out << "    .target_input_names = " << array_ref(descriptor.target_inputs, "kTargetInputNames") << ",\n";
    out << "    .source_exposure_count = " << descriptor.source_exposures.size() << ",\n";
    out << "    .source_exposure_names = " << array_ref(descriptor.source_exposures, "kSourceExposureNames") << ",\n";
    out << "    .param_count = " << param_names.size() << ",\n";
    out << "    .param_names = " << array_ref(param_names, "kParamNames") << ",\n";
    out << "    .param_defaults = " << array_ref(params, "kParamDefaults") << ",\n";
    out << "    .state_count = " << state_names.size() << ",\n";
    out << "    .state_names = " << array_ref(state_names, "kStateNames") << ",\n";
    out << "    .state_defaults = " << array_ref(states, "kStateDefaults") << ",\n";
    out << "    .local_state_count = 0,\n";
    out << "    .local_state_names = nullptr,\n";
    out << "};\n";
    return out.str();
}

[[nodiscard]] std::string render_common(const std::string& model,
                                        int field_count,
                                        int datum_count,
                                        int pntproc_datum_offset,
                                        int netsend_datum_offset,
                                        int random_datum_offset) {
    std::ostringstream out;
    out << R"(
struct CoreScratch {
    int capacity{0};
    std::vector<int> nodeindices;
    std::vector<coreneuron::Datum> pdata;
    std::vector<void*> vdata;
    std::vector<double> data;
    std::vector<double> voltage;
    std::vector<double> area;
    std::vector<coreneuron::Point_process> pntprocs;
    std::vector<double> weights;
    std::vector<coreneuron::nrnran123_State*> random_states;
    std::vector<coreneuron::Memb_list*> ml_list;
    coreneuron::NrnThread nt{};
    coreneuron::Memb_list ml{};

    ~CoreScratch() {
        for (auto* state: random_states) {
            coreneuron::nrnran123_deletestream(state, false);
        }
    }

    void resize(int count) {
        count = std::max(count, 1);
        if (capacity != count) {
            for (auto* state: random_states) {
                coreneuron::nrnran123_deletestream(state, false);
            }
            random_states.clear();
            capacity = count;
            nodeindices.resize(static_cast<std::size_t>(count));
            pdata.resize(static_cast<std::size_t>()" << datum_count
        << R"() * static_cast<std::size_t>(count));
            vdata.resize(static_cast<std::size_t>()" << datum_count
        << R"() * static_cast<std::size_t>(count));
            data.resize(static_cast<std::size_t>()" << field_count
        << R"() * static_cast<std::size_t>(count));
            voltage.resize(static_cast<std::size_t>(count));
            area.assign(static_cast<std::size_t>(count), 1.0);
            pntprocs.resize(static_cast<std::size_t>(count));
            weights.resize(2);
            ml_list.resize(1);
)";
    if (random_datum_offset >= 0) {
        out << R"(            random_states.resize(static_cast<std::size_t>(count));
            for (int i = 0; i < count; ++i) {
                random_states[static_cast<std::size_t>(i)] =
                    coreneuron::nrnran123_newstream3(0, 0, 0, false);
            }
)";
    }
    out << R"(            for (int i = 0; i < count; ++i) {
                nodeindices[static_cast<std::size_t>(i)] = i;
            }
        }
        std::fill(data.begin(), data.end(), 0.0);
        std::fill(voltage.begin(), voltage.end(), 0.0);
        std::fill(vdata.begin(), vdata.end(), nullptr);
        ml.nodecount = count;
        ml._nodecount_padded = count;
        ml.nodeindices = nodeindices.data();
        ml.data = data.data();
        ml.pdata = pdata.empty() ? nullptr : pdata.data();
        nt._actual_v = voltage.data();
        nt._data = area.data();
        nt._vdata = vdata.empty() ? nullptr : vdata.data();
        nt.pntprocs = pntprocs.data();
        nt.weights = weights.data();
        nt.n_pntproc = count;
        nt.n_weight = static_cast<int>(weights.size());
        nt.id = 0;
        nt._dt = 0.0;
        nt._t = 0.0;
        ml_list[0] = &ml;
        nt._ml_list = ml_list.data();
        for (int i = 0; i < count; ++i) {
            pntprocs[static_cast<std::size_t>(i)]._i_instance = i;
            pntprocs[static_cast<std::size_t>(i)]._type = 0;
            pntprocs[static_cast<std::size_t>(i)]._tid = 0;
        }
)";
    if (pntproc_datum_offset >= 0) {
        out << "        for (int i = 0; i < count; ++i) {\n";
        out << "            const auto index = static_cast<std::size_t>(" << pntproc_datum_offset
            << ") * static_cast<std::size_t>(count) + static_cast<std::size_t>(i);\n";
        out << "            pdata[index] = static_cast<coreneuron::Datum>(index);\n";
        out << "            vdata[index] = &pntprocs[static_cast<std::size_t>(i)];\n";
        out << "        }\n";
    }
    if (netsend_datum_offset >= 0) {
        out << "        for (int i = 0; i < count; ++i) {\n";
        out << "            const auto index = static_cast<std::size_t>(" << netsend_datum_offset
            << ") * static_cast<std::size_t>(count) + static_cast<std::size_t>(i);\n";
        out << "            pdata[index] = static_cast<coreneuron::Datum>(index);\n";
        out << "            vdata[index] = nullptr;\n";
        out << "        }\n";
    }
    if (random_datum_offset >= 0) {
        out << "        for (int i = 0; i < count; ++i) {\n";
        out << "            const auto index = static_cast<std::size_t>(" << random_datum_offset
            << ") * static_cast<std::size_t>(count) + static_cast<std::size_t>(i);\n";
        out << "            pdata[index] = static_cast<coreneuron::Datum>(index);\n";
        out << "            vdata[index] = random_states[static_cast<std::size_t>(i)];\n";
        out << "        }\n";
    }
    out << R"(
        if (ml.instance == nullptr) {
            coreneuron::nrn_private_constructor_)" << model
        << R"((&nt, &ml, 0);
        }
        coreneuron::setup_instance(&nt, &ml);
    }
};

thread_local CoreScratch kScratch;

inline double get_field(const CoreScratch& scratch, int offset, int count, int index) {
    return scratch.data[static_cast<std::size_t>(offset) * static_cast<std::size_t>(count) +
                        static_cast<std::size_t>(index)];
}

inline void set_field(CoreScratch& scratch, int offset, int count, int index, double value) {
    scratch.data[static_cast<std::size_t>(offset) * static_cast<std::size_t>(count) +
                 static_cast<std::size_t>(index)] = value;
}
)";
    return out.str();
}

[[nodiscard]] std::string render_region(const std::string& model,
                                        const MindSpec& spec,
                                        const std::vector<Variable>& states,
                                        const std::vector<Variable>& params,
                                        const std::map<std::string, int>& offsets) {
    const auto state_names = names_of(states);
    const auto param_names = names_of(params);
    auto required = spec.target_inputs;
    required.insert(required.end(), spec.source_exposures.begin(), spec.source_exposures.end());
    required.insert(required.end(), state_names.begin(), state_names.end());
    required.insert(required.end(), param_names.begin(), param_names.end());
    require_offsets(offsets, required, "macro");

    std::ostringstream out;
    out << offset_map("kTargetInputFieldOffsets", spec.target_inputs, offsets);
    out << offset_map("kSourceExposureFieldOffsets", spec.source_exposures, offsets);
    out << offset_map("kStateFieldOffsets", state_names, offsets);
    out << offset_map("kParamFieldOffsets", param_names, offsets);
    out << "\nextern \"C\" MIND_RULE_EXPORT void mind_region_rule_apply(const "
           "mind_sim::mod::AbiRegionContext* ctx) {\n";
    out << "    const int count = ctx->owner_count;\n";
    out << "    kScratch.resize(count);\n";
    out << "    kScratch.nt._dt = ctx->dt;\n";
    out << "    kScratch.nt._t = ctx->t;\n";
    out << "    for (int unit = 0; unit < count; ++unit) {\n";
    out << "        const int roi = ctx->roi_indices[unit];\n";
    out << "        for (int i = 0; i < " << spec.target_inputs.size() << "; ++i) {\n";
    out << "            set_field(kScratch, kTargetInputFieldOffsets[i], count, unit,\n";
    out << "                      ctx->input_soa[ctx->target_input_offsets[i] + roi]);\n";
    out << "        }\n";
    out << "        for (int i = 0; i < " << state_names.size() << "; ++i) {\n";
    out << "            set_field(kScratch, kStateFieldOffsets[i], count, unit,\n";
    out << "                      ctx->state_soa[(i * count) + unit]);\n";
    out << "        }\n";
    out << "        for (int i = 0; i < " << param_names.size() << "; ++i) {\n";
    out << "            set_field(kScratch, kParamFieldOffsets[i], count, unit,\n";
    out << "                      ctx->params_soa[(i * count) + unit]);\n";
    out << "        }\n";
    out << "    }\n";
    out << "    coreneuron::nrn_state_" << model << "(&kScratch.nt, &kScratch.ml, 0);\n";
    out << "    for (int unit = 0; unit < count; ++unit) {\n";
    out << "        const int roi = ctx->roi_indices[unit];\n";
    out << "        for (int i = 0; i < " << state_names.size() << "; ++i) {\n";
    out << "            ctx->state_soa[(i * count) + unit] = get_field(kScratch, "
           "kStateFieldOffsets[i], count, unit);\n";
    out << "        }\n";
    out << "        for (int i = 0; i < " << spec.source_exposures.size() << "; ++i) {\n";
    out << "            ctx->exposure_soa[ctx->source_exposure_offsets[i] + roi] =\n";
    out << "                get_field(kScratch, kSourceExposureFieldOffsets[i], count, unit);\n";
    out << "        }\n";
    out << "    }\n";
    out << "}\n";
    return out.str();
}

[[nodiscard]] std::string render_macro_to_macro(const std::string& model,
                                          const MindSpec& spec,
                                          const std::vector<Variable>& params,
                                          const std::map<std::string, int>& offsets) {
    const auto param_names = names_of(params);
    auto required = spec.source_exposures;
    required.insert(required.end(), spec.target_inputs.begin(), spec.target_inputs.end());
    required.insert(required.end(), param_names.begin(), param_names.end());
    require_offsets(offsets, required, "macro-to-macro");

    std::ostringstream out;
    out << offset_map("kSourceExposureFieldOffsets", spec.source_exposures, offsets);
    out << offset_map("kTargetInputFieldOffsets", spec.target_inputs, offsets);
    out << offset_map("kParamFieldOffsets", param_names, offsets);
    out << "\nextern \"C\" MIND_RULE_EXPORT void mind_macro_to_macro_rule_apply(const "
           "mind_sim::mod::AbiMacroToMacroContext* ctx) {\n";
    out << R"(    std::vector<int> edge_indices;
    std::vector<int> target_indices;
    for (int target_pos = 0; target_pos < ctx->target_count; ++target_pos) {
        const int target_roi = ctx->target_indices[target_pos];
        for (int edge = ctx->target_edge_offsets[target_roi]; edge < ctx->target_edge_offsets[target_roi + 1]; ++edge) {
            edge_indices.push_back(edge);
            target_indices.push_back(target_roi);
        }
    }
    const int count = static_cast<int>(edge_indices.size());
    if (count == 0) {
        return;
    }
    kScratch.resize(count);
    const int roi_count = ctx->roi_count;
    const int history_stride = ctx->exposure_count * roi_count;
    const int history_size = ctx->history_capacity * history_stride;
    const int current_history_offset = (ctx->step % ctx->history_capacity) * history_stride;
    for (int unit = 0; unit < count; ++unit) {
        const int edge = edge_indices[static_cast<std::size_t>(unit)];
        const int target_roi = target_indices[static_cast<std::size_t>(unit)];
        const int source_roi = ctx->edge_sources[edge];
        int history_offset = current_history_offset + ctx->edge_delay_offsets[edge];
        while (history_offset >= history_size) {
            history_offset -= history_size;
        }
)";
    out << "        for (int i = 0; i < " << spec.source_exposures.size() << "; ++i) {\n";
    out << "            set_field(kScratch, kSourceExposureFieldOffsets[i], count, unit,\n";
    out << "                      ctx->history[history_offset + ctx->source_exposure_offsets[i] "
           "+ source_roi]);\n";
    out << "        }\n";
    out << "        for (int i = 0; i < " << spec.target_inputs.size() << "; ++i) {\n";
    out << "            set_field(kScratch, kTargetInputFieldOffsets[i], count, unit, 0.0);\n";
    out << "        }\n";
    out << "        for (int i = 0; i < " << param_names.size() << "; ++i) {\n";
    out << "            set_field(kScratch, kParamFieldOffsets[i], count, unit, ctx->params[i]);\n";
    out << "        }\n";
    const std::map<std::string, std::string> edge_values{
        {"weight", "ctx->edge_weights[edge]"},
        {"delay", "static_cast<double>(ctx->edge_delay_steps[edge])"},
        {"delay_steps", "static_cast<double>(ctx->edge_delay_steps[edge])"},
        {"source_roi", "static_cast<double>(source_roi)"},
        {"target_roi", "static_cast<double>(target_roi)"},
    };
    for (const auto& [name, expression] : edge_values) {
        if (offsets.contains(name)) {
            out << "        set_field(kScratch, " << offsets.at(name)
                << ", count, unit, " << expression << ");\n";
        }
    }
    out << "    }\n";
    out << "    coreneuron::nrn_state_" << model << "(&kScratch.nt, &kScratch.ml, 0);\n";
    out << "    for (int unit = 0; unit < count; ++unit) {\n";
    out << "        const int target_roi = target_indices[static_cast<std::size_t>(unit)];\n";
    out << "        for (int i = 0; i < " << spec.target_inputs.size() << "; ++i) {\n";
    out << "            ctx->inputs[ctx->target_input_offsets[i] + target_roi] +=\n";
    out << "                get_field(kScratch, kTargetInputFieldOffsets[i], count, unit);\n";
    out << "        }\n";
    out << "    }\n";
    out << "}\n";
    return out.str();
}

[[nodiscard]] std::string render_micro_output(const std::string& model,
                                              const MindSpec& spec,
                                              const std::vector<Variable>& states,
                                              const std::vector<Variable>& params,
                                              const std::map<std::string, int>& offsets) {
    const auto state_names = names_of(states);
    const auto param_names = names_of(params);
    auto required = spec.source_exposures;
    required.insert(required.end(), state_names.begin(), state_names.end());
    required.insert(required.end(), param_names.begin(), param_names.end());
    require_offsets(offsets, required, "micro output");

    std::ostringstream out;
    out << offset_map("kSourceExposureFieldOffsets", spec.source_exposures, offsets);
    out << offset_map("kStateFieldOffsets", state_names, offsets);
    out << offset_map("kParamFieldOffsets", param_names, offsets);
    out << "\nextern \"C\" MIND_RULE_EXPORT void mind_micro_output_rule_apply(const "
           "mind_sim::mod::AbiMicroOutputContext* ctx) {\n";
    out << R"(    constexpr int count = 1;
    kScratch.resize(count);
    kScratch.nt._dt = ctx->stop_time - ctx->start_time;
    kScratch.nt._t = ctx->start_time;
)";
    out << "    for (int i = 0; i < " << state_names.size() << "; ++i) {\n";
    out << "        set_field(kScratch, kStateFieldOffsets[i], count, 0, ctx->state[i]);\n";
    out << "    }\n";
    out << "    for (int i = 0; i < " << param_names.size() << "; ++i) {\n";
    out << "        set_field(kScratch, kParamFieldOffsets[i], count, 0, ctx->params[i]);\n";
    out << "    }\n";
    if (offsets.contains("dt")) {
        out << "    set_field(kScratch, " << offsets.at("dt") << ", count, 0, kScratch.nt._dt);\n";
    }
    if (offsets.contains("start_time")) {
        out << "    set_field(kScratch, " << offsets.at("start_time")
            << ", count, 0, ctx->start_time);\n";
    }
    if (offsets.contains("stop_time")) {
        out << "    set_field(kScratch, " << offsets.at("stop_time")
            << ", count, 0, ctx->stop_time);\n";
    }
    out << "    auto* const inst = static_cast<coreneuron::" << model
        << "_Instance*>(kScratch.ml.instance);\n";
    out << R"(    const int sample_count = ctx->sample_count;
    if (sample_count <= 0) {
        return;
    }
    double current_time = ctx->start_time;
    const auto advance_state = [&](double next_time) {
        if (next_time < current_time) {
            next_time = current_time;
        }
        kScratch.nt._t = current_time;
        kScratch.nt._dt = next_time - current_time;
)";
    out << "        coreneuron::nrn_state_" << model << "(&kScratch.nt, &kScratch.ml, 0);\n";
    out << R"(        current_time = next_time;
    };
    const int roi = ctx->target_roi;
    const int exposure_stride = ctx->roi_count * ctx->exposure_count;
    int spike = 0;
    for (int sample = 0; sample < sample_count; ++sample) {
        double sample_time = ctx->start_time + (static_cast<double>(sample) + 1.0) * ctx->sample_dt;
        if (sample_time > ctx->stop_time) {
            sample_time = ctx->stop_time;
        }
        while (spike < ctx->spikes->size && ctx->spikes->time[spike] <= sample_time + 1e-12) {
            double event_time = ctx->spikes->time[spike];
            if (event_time > sample_time) {
                event_time = sample_time;
            }
            const int gid = ctx->spikes->gid[spike];
            if (event_time < current_time - 1e-12) {
                ++spike;
                continue;
            }
            advance_state(event_time);
            kScratch.nt._t = current_time;
            kScratch.weights[0] = 1.0;
            kScratch.weights[1] = static_cast<double>(gid);
)";
    out << "        coreneuron::net_receive_kernel_" << model
        << "(current_time, &kScratch.pntprocs[0], inst, &kScratch.nt, &kScratch.ml, 0, 0.0);\n";
    out << R"(            ++spike;
        }
        advance_state(sample_time);
)";
    out << "        for (int i = 0; i < " << spec.source_exposures.size() << "; ++i) {\n";
    out << "            ctx->exposure_trace_soa[(sample * exposure_stride) + ctx->source_exposure_offsets[i] + roi] =\n";
    out << "                get_field(kScratch, kSourceExposureFieldOffsets[i], count, 0);\n";
    out << "        }\n";
    out << R"(    }
    if (current_time < ctx->stop_time) {
        advance_state(ctx->stop_time);
    }
)";
    out << "    for (int i = 0; i < " << state_names.size() << "; ++i) {\n";
    out << "        ctx->state[i] = get_field(kScratch, kStateFieldOffsets[i], count, 0);\n";
    out << "    }\n";
    out << "    for (int i = 0; i < " << spec.source_exposures.size() << "; ++i) {\n";
    out << "        ctx->exposure_soa[ctx->source_exposure_offsets[i] + roi] =\n";
    out << "            get_field(kScratch, kSourceExposureFieldOffsets[i], count, 0);\n";
    out << "    }\n";
    out << "}\n";
    return out.str();
}

[[nodiscard]] std::string render_micro_input(const std::string& model,
                                             const MindSpec& spec,
                                             const std::vector<Variable>& states,
                                             const std::vector<Variable>& params,
                                             const std::map<std::string, int>& offsets,
                                             int random_datum_offset) {
    const auto state_names = names_of(states);
    const auto param_names = names_of(params);
    auto required = spec.target_inputs;
    required.insert(required.end(), state_names.begin(), state_names.end());
    required.insert(required.end(), param_names.begin(), param_names.end());
    require_offsets(offsets, required, "micro input");

    std::ostringstream out;
    out << offset_map("kTargetInputFieldOffsets", spec.target_inputs, offsets);
    out << offset_map("kStateFieldOffsets", state_names, offsets);
    out << offset_map("kParamFieldOffsets", param_names, offsets);
    out << R"(
namespace mind_macro2micro_runtime {

struct ScheduledSelfEvent {
    double time{0.0};
    int source{0};
    double flag{0.0};
    int weight_index{0};
};

struct LaterSelfEvent {
    bool operator()(const ScheduledSelfEvent& lhs, const ScheduledSelfEvent& rhs) const noexcept {
        return lhs.time > rhs.time;
    }
};

thread_local const mind_sim::mod::AbiMicroInputContext* gContext = nullptr;
thread_local std::vector<ScheduledSelfEvent>* gQueue = nullptr;
thread_local std::vector<ScheduledSelfEvent> gScratchQueue;

void schedule_self_event(coreneuron::Point_process* pnt,
                         double time,
                         double flag,
                         int weight_index) {
    if (gContext == nullptr || gQueue == nullptr) {
        throw std::runtime_error("macro2micro net_send called outside an active event-source window");
    }
    if (pnt == nullptr) {
        throw std::runtime_error("macro2micro net_send received a null Point_process");
    }
    if (time < gContext->start_time - 1e-12) {
        throw std::runtime_error("macro2micro net_send scheduled an event before the current window");
    }
    if (time >= gContext->stop_time - 1e-12) {
        return;
    }
    gQueue->push_back(ScheduledSelfEvent{
        .time = time,
        .source = pnt->_i_instance,
        .flag = flag,
        .weight_index = weight_index,
    });
    std::push_heap(gQueue->begin(), gQueue->end(), LaterSelfEvent{});
}

}  // namespace mind_macro2micro_runtime

namespace coreneuron {

void mind_macro2micro_net_event(Point_process* pnt, double time) {
    auto* const ctx = mind_macro2micro_runtime::gContext;
    if (ctx == nullptr || ctx->emit_event == nullptr) {
        throw std::runtime_error("macro2micro net_event called outside an active event-source window");
    }
    if (pnt == nullptr) {
        throw std::runtime_error("macro2micro net_event received a null Point_process");
    }
    if (time < ctx->start_time - 1e-12 || time >= ctx->stop_time - 1e-12) {
        return;
    }
    const int source = pnt->_i_instance;
    if (source < 0 || source >= ctx->source_count) {
        throw std::runtime_error("macro2micro net_event source index is out of range");
    }
    if (ctx->source_indices == nullptr) {
        throw std::runtime_error("macro2micro source index mapping is null");
    }
    ctx->emit_event(ctx->event_user_data, time, ctx->source_indices[source]);
}

void mind_macro2micro_net_send(void**, int weight_index, Point_process* pnt, double time, double flag) {
    mind_macro2micro_runtime::schedule_self_event(pnt, time, flag, weight_index);
}

void mind_macro2micro_artcell_net_send(void**, int weight_index, Point_process* pnt, double time, double flag) {
    mind_macro2micro_runtime::schedule_self_event(pnt, time, flag, weight_index);
}

}  // namespace coreneuron
)";
    out << "\nextern \"C\" MIND_RULE_EXPORT void mind_micro_input_rule_apply(const "
           "mind_sim::mod::AbiMicroInputContext* ctx) {\n";
    out << R"(    const int count = ctx->source_count;
    if (count <= 0) {
        return;
    }
    if (ctx->emit_event == nullptr) {
        throw std::runtime_error("macro2micro event sink is null");
    }
    if (ctx->sample_count <= 0 || ctx->sample_dt <= 0.0) {
        return;
    }
    if (ctx->input_trace_soa == nullptr) {
        throw std::runtime_error("macro2micro input trace is null");
    }
    kScratch.resize(count);
    kScratch.nt._dt = ctx->sample_dt;
    kScratch.nt._t = ctx->start_time;
    const int roi = ctx->target_roi;
    const int input_stride = ctx->input_count * ctx->roi_count;
)";
    if (random_datum_offset >= 0) {
        out << R"(    const auto seed_low = static_cast<std::uint32_t>(ctx->rng_seed & 0xffffffffULL);
    const auto seed_high = static_cast<std::uint32_t>((ctx->rng_seed >> 32) & 0xffffffffULL);
    const auto roi_id = static_cast<std::uint32_t>(ctx->target_roi);
    const auto seq = static_cast<std::uint32_t>(std::max(ctx->exchange_start_step, 0));
    for (int source = 0; source < count; ++source) {
        auto* const rng = kScratch.random_states[static_cast<std::size_t>(source)];
        rng->c.v[1] = seed_high ^ roi_id;
        rng->c.v[2] = seed_low;
        rng->c.v[3] = static_cast<std::uint32_t>(ctx->source_indices[source]);
        coreneuron::nrnran123_setseq(rng, seq, 0);
    }
)";
    }
    out << "    for (int i = 0; i < " << state_names.size() << "; ++i) {\n";
    out << "        for (int source = 0; source < count; ++source) {\n";
    out << "            set_field(kScratch, kStateFieldOffsets[i], count, source,\n";
    out << "                      ctx->state[(i * count) + source]);\n";
    out << "        }\n";
    out << "    }\n";
    out << "    for (int i = 0; i < " << param_names.size() << "; ++i) {\n";
    out << "        for (int source = 0; source < count; ++source) {\n";
    out << "            set_field(kScratch, kParamFieldOffsets[i], count, source, ctx->params[i]);\n";
    out << "        }\n";
    out << "    }\n";
    out << R"(    auto& queue = mind_macro2micro_runtime::gScratchQueue;
    queue.clear();
    mind_macro2micro_runtime::gContext = ctx;
    mind_macro2micro_runtime::gQueue = &queue;
    auto* const saved_nrn_threads = coreneuron::nrn_threads_for_mind_macro2micro;
    coreneuron::nrn_threads_for_mind_macro2micro = &kScratch.nt;
    constexpr int max_events = 10000000;
    int event_count = 0;
    try {
        for (int sample = 0; sample < ctx->sample_count; ++sample) {
            const double sample_start = ctx->start_time + static_cast<double>(sample) * ctx->sample_dt;
            const double sample_stop =
                sample + 1 == ctx->sample_count ? ctx->stop_time : sample_start + ctx->sample_dt;
            kScratch.nt._dt = sample_stop - sample_start;
)";
    out << "            for (int i = 0; i < " << spec.target_inputs.size() << "; ++i) {\n";
    out << "                const double value = ctx->input_trace_soa[(static_cast<std::size_t>(sample) * input_stride) + ctx->target_input_offsets[i] + roi];\n";
    out << "                for (int source = 0; source < count; ++source) {\n";
    out << "                    set_field(kScratch, kTargetInputFieldOffsets[i], count, source, value);\n";
    out << "                }\n";
    out << "            }\n";
    if (offsets.contains("dt")) {
        out << "            for (int source = 0; source < count; ++source) {\n";
        out << "                set_field(kScratch, " << offsets.at("dt") << ", count, source, kScratch.nt._dt);\n";
        out << "            }\n";
    }
    if (offsets.contains("start_time")) {
        out << "            for (int source = 0; source < count; ++source) {\n";
        out << "                set_field(kScratch, " << offsets.at("start_time")
            << ", count, source, sample_start);\n";
        out << "            }\n";
    }
    if (offsets.contains("stop_time")) {
        out << "            for (int source = 0; source < count; ++source) {\n";
        out << "                set_field(kScratch, " << offsets.at("stop_time")
            << ", count, source, sample_stop);\n";
        out << "            }\n";
    }
    out << R"(            for (int source = 0; source < count; ++source) {
                kScratch.nt._t = sample_start;
                kScratch.weights[0] = 1.0;
)";
    out << "            coreneuron::net_receive_" << model
        << "(&kScratch.pntprocs[source], 0, 0.0);\n";
    out << R"(            }
            while (!queue.empty()) {
                const auto next = queue.front();
                if (next.time >= sample_stop - 1e-12) {
                    break;
                }
                std::pop_heap(queue.begin(), queue.end(), mind_macro2micro_runtime::LaterSelfEvent{});
                const auto event = queue.back();
                queue.pop_back();
                if (event.time < ctx->start_time - 1e-12 || event.time >= ctx->stop_time - 1e-12) {
                    continue;
                }
                if (++event_count > max_events) {
                    throw std::runtime_error("macro2micro generated too many self events in one exchange window");
                }
                kScratch.nt._t = event.time;
                kScratch.weights[0] = 1.0;
)";
    out << "            coreneuron::net_receive_" << model
        << "(&kScratch.pntprocs[event.source], event.weight_index, event.flag);\n";
    out << R"(            }
        }
    } catch (...) {
        mind_macro2micro_runtime::gContext = nullptr;
        mind_macro2micro_runtime::gQueue = nullptr;
        coreneuron::nrn_threads_for_mind_macro2micro = saved_nrn_threads;
        throw;
    }
    mind_macro2micro_runtime::gContext = nullptr;
    mind_macro2micro_runtime::gQueue = nullptr;
    coreneuron::nrn_threads_for_mind_macro2micro = saved_nrn_threads;
)";
    out << "    for (int i = 0; i < " << state_names.size() << "; ++i) {\n";
    out << "        for (int source = 0; source < count; ++source) {\n";
    out << "            ctx->state[(i * count) + source] = get_field(kScratch, kStateFieldOffsets[i], count, source);\n";
    out << "        }\n";
    out << "    }\n";
    out << "}\n";
    return out.str();
}

[[nodiscard]] std::string render_wrapper(const fs::path& generated_cpp,
                                         const MechanismDataLayout& layout,
                                         const MindSpec& spec,
                                         const std::vector<Variable>& states,
                                         const std::vector<Variable>& params) {
    const std::string model = layout.mechanism;
    const auto offsets = data_offsets_from_layout(layout);
    const int datum_count = static_cast<int>(layout.dparam_fields.size());
    const auto datum_offsets = datum_semantic_offsets_from_layout(layout);
    const int pntproc_datum_offset =
        datum_offsets.contains("pntproc") ? datum_offsets.at("pntproc") : -1;
    const int netsend_datum_offset =
        datum_offsets.contains("netsend") ? datum_offsets.at("netsend") : -1;
    const int random_datum_offset =
        datum_offsets.contains("random") ? datum_offsets.at("random") : -1;
    const int field_count = data_field_count(layout);
    require_offsets(offsets, names_of(states), "state");

    std::ostringstream body;
    body << render_descriptor(model, spec, states, params);
    body << render_common(model,
                          field_count,
                          datum_count,
                          pntproc_datum_offset,
                          netsend_datum_offset,
                          random_datum_offset);
    if (spec.role == "REGION") {
        body << render_region(model, spec, states, params, offsets);
    } else if (spec.role == "MACRO2MACRO") {
        body << render_macro_to_macro(model, spec, params, offsets);
    } else if (spec.role == "MICRO2MACRO") {
        body << render_micro_output(model, spec, states, params, offsets);
    } else if (spec.role == "MACRO2MICRO") {
        body << render_micro_input(model, spec, states, params, offsets, random_datum_offset);
    } else {
        throw std::runtime_error("unsupported MIND ROLE " + spec.role);
    }

    std::ostringstream out;
    out << "#include \"mod/abi.hpp\"\n\n";
    out << "#include <algorithm>\n";
    out << "#include <cmath>\n";
    out << "#include <cstdint>\n";
    out << "#include <stdexcept>\n";
    out << "#include <vector>\n\n";
    if (spec.role == "MACRO2MICRO") {
        out << "#define nrn_threads nrn_threads_for_mind_macro2micro\n";
        out << "#define net_event mind_macro2micro_net_event\n";
        out << "#define net_send mind_macro2micro_net_send\n";
        out << "#define artcell_net_send mind_macro2micro_artcell_net_send\n";
    }
    out << "#include " << cpp_string(generated_cpp.string()) << "\n\n";
    if (spec.role == "MACRO2MICRO") {
        out << "#undef nrn_threads\n";
        out << "#undef net_event\n";
        out << "#undef net_send\n";
        out << "#undef artcell_net_send\n\n";
        out << "namespace coreneuron {\n";
        out << "NrnThread* nrn_threads_for_mind_macro2micro = nullptr;\n";
        out << "}\n\n";
    }
    out << "#if defined(_WIN32)\n";
    out << "#define MIND_RULE_EXPORT __declspec(dllexport)\n";
    out << "#else\n";
    out << "#define MIND_RULE_EXPORT __attribute__((visibility(\"default\")))\n";
    out << "#endif\n\n";
    out << body.str() << "\n";
    out << "extern \"C\" MIND_RULE_EXPORT const "
           "mind_sim::mod::AbiRuleDescriptor* mind_rule_descriptor() {\n";
    out << "    return &kDescriptor;\n";
    out << "}\n";
    return out.str();
}

[[nodiscard]] std::vector<fs::path> collect_mods(const fs::path& source) {
    std::vector<fs::path> mods;
    if (fs::is_regular_file(source)) {
        if (source.extension() != ".mod") {
            throw std::runtime_error("expected a .mod file: " + source.string());
        }
        mods.push_back(fs::absolute(source));
        return mods;
    }
    if (!fs::is_directory(source)) {
        throw std::runtime_error("MOD source path does not exist: " + source.string());
    }
    for (const auto& entry : fs::directory_iterator(source)) {
        if (entry.is_regular_file() && entry.path().extension() == ".mod") {
            mods.push_back(fs::absolute(entry.path()));
        }
    }
    std::sort(mods.begin(), mods.end());
    if (mods.empty()) {
        throw std::runtime_error("no .mod files found in " + source.string());
    }
    return mods;
}

[[nodiscard]] fs::path output_dir_for(const fs::path& source) {
    if (fs::is_regular_file(source)) {
        return fs::absolute(source.parent_path()) / "x86_64";
    }
    return fs::absolute(source) / "x86_64";
}

[[nodiscard]] std::string include_arg(const fs::path& path) {
    return "-I" + shell_quote(path);
}

void compile_object_file(const fs::path& source_path, const fs::path& object_path) {
    fs::create_directories(object_path.parent_path());
    run({
        shell_quote(MIND_SIM_CXX_COMPILER),
        "-std=c++20",
        "-O2",
        "-fPIC",
        MIND_SIM_MOD_CXX_FLAGS,
        include_arg(fs::path{MIND_SIM_SOURCE_DIR} / "src"),
        include_arg(MIND_SIM_MECHANISM_INCLUDE_DIR),
        "-DCORENEURON_BUILD",
        "-DCORENRN_BUILD=1",
        "-DVECTORIZE=1",
        "-DHAVE_MALLOC_H",
        "-DEIGEN_DONT_PARALLELIZE",
        "-DLAYOUT=0",
        "-DENABLE_SPLAYTREE_QUEUING",
        "-DNRNMPI=0",
        "-DNRN_MULTISEND=0",
        "-DDISABLE_HOC_EXP",
        "-DNET_RECEIVE_BUFFERING=0",
        "-DNRN_PRCELLSTATE=0",
        "-c",
        shell_quote(source_path),
        "-o",
        shell_quote(object_path),
    });
}

void link_shared_library(const std::vector<fs::path>& objects, const fs::path& library_path) {
    if (objects.empty()) {
        throw std::runtime_error("cannot link an empty mechanism library");
    }
    fs::create_directories(library_path.parent_path());
    std::vector<std::string> command{
        shell_quote(MIND_SIM_CXX_COMPILER),
        "-shared",
        "-fPIC",
        "-o",
        shell_quote(library_path),
    };
    for (const auto& object : objects) {
        command.push_back(shell_quote(object));
    }
    command.push_back(MIND_SIM_MOD_LINK_FLAGS);
#if defined(__linux__)
    command.push_back("-Wl,--allow-shlib-undefined");
#endif
    run(command);
}

void compile_shared_library(const fs::path& wrapper_cpp,
                            const fs::path& object_path,
                            const fs::path& library_path) {
    compile_object_file(wrapper_cpp, object_path);
    link_shared_library({object_path}, library_path);
}

[[nodiscard]] std::string field_role_expr(const std::string& role) {
    if (role == "parameter") {
        return "neuron::mechanism::field_role::parameter";
    }
    if (role == "assigned") {
        return "neuron::mechanism::field_role::assigned";
    }
    if (role == "state") {
        return "neuron::mechanism::field_role::state";
    }
    return "neuron::mechanism::field_role::range";
}

[[nodiscard]] std::string field_cpp_type(const MechanismFieldLayout& field) {
    if (field.cpp_type == "Point_process*") {
        return "coreneuron::Point_process*";
    }
    return field.cpp_type;
}

[[nodiscard]] std::string render_data_field_expr(const MechanismFieldLayout& field) {
    std::ostringstream out;
    out << "neuron::mechanism::field<" << field_cpp_type(field) << ">("
        << cpp_string(field.name);
    if (field.cpp_type == "double") {
        out << ", " << field_role_expr(field.role);
        if (field.array_size != 1) {
            out << ", " << field.array_size;
        }
    } else {
        out << ", " << cpp_string(field.semantic);
    }
    out << ")";
    return out.str();
}

[[nodiscard]] std::string render_data_layout_registration(const MechanismDataLayout& layout) {
    std::ostringstream out;
    out << "    {\n";
    out << "        const int mech_type = nrn_get_mechtype("
        << cpp_string(layout.mechanism) << ");\n";
    out << "        if (mech_type != -1) {\n";
    if (!layout.defaults.empty()) {
        out << "            static const std::vector<double> defaults = {";
        for (std::size_t i = 0; i < layout.defaults.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            out << layout.defaults[i];
        }
        out << "};\n";
        out << "            hoc_register_parm_default(mech_type, &defaults);\n";
    }
    out << "            neuron::mechanism::register_data_fields(\n";
    out << "                mech_type";
    for (const auto& field : layout.data_fields) {
        out << ",\n                " << render_data_field_expr(field);
    }
    for (const auto& field : layout.dparam_fields) {
        out << ",\n                " << render_data_field_expr(field);
    }
    out << ");\n";
    out << "        }\n";
    out << "    }\n";
    return out.str();
}

struct StandardModUnit {
    fs::path mod;
    MechanismDataLayout layout;
};

void write_modl_reg_source(const fs::path& path, const std::vector<StandardModUnit>& units) {
    std::ostringstream out;
    out << "#include <cstdio>\n";
    out << "#include <vector>\n";
    out << "#include <coreneuron/mechanism/membfunc.hpp>\n";
    out << "#include <coreneuron/mechanism/neuron_registration.hpp>\n";
    out << "namespace coreneuron {\n";
    out << "extern int nrnmpi_myid;\n";
    out << "extern int nrn_nobanner_;\n";
    out << "void hoc_register_parm_default(int type, const std::vector<double>* defaults);\n";
    for (const auto& unit : units) {
        out << "extern void _" << unit.mod.stem().string() << "_reg();\n";
    }
    out << "void modl_reg() {\n";
    out << "    if (!nrn_nobanner_ && nrnmpi_myid < 1) {\n";
    out << "        std::fprintf(stderr, \" Additional MIND_Sim mechanisms from MOD files\\n\");\n";
    for (const auto& unit : units) {
        out << "        std::fprintf(stderr, \" " << unit.mod.filename().string() << "\");\n";
    }
    out << "        std::fprintf(stderr, \"\\n\\n\");\n";
    out << "    }\n";
    for (const auto& unit : units) {
        out << "    _" << unit.mod.stem().string() << "_reg();\n";
        out << render_data_layout_registration(unit.layout);
    }
    out << "}\n";
    out << "}\n";
    write_text(path, out.str());
}

void compile_standard_mods(const fs::path& source, const fs::path& output_dir) {
    if (!fs::is_directory(source)) {
        throw std::runtime_error("standard NEURON/CoreNEURON mechanisms must be compiled from a MOD directory");
    }

    const auto mods = collect_mods(source);
    const fs::path workdir = output_dir / "corenrn";
    const fs::path generated_dir = workdir / "mod2c";
    const fs::path scratch_root = workdir / "scratch";
    const fs::path object_dir = workdir / "build";
    fs::remove_all(workdir);
    fs::create_directories(generated_dir);
    fs::create_directories(scratch_root);
    fs::create_directories(object_dir);

    std::vector<fs::path> objects;
    std::vector<StandardModUnit> units;
    objects.reserve(mods.size() + 1);
    units.reserve(mods.size());
    for (const auto& mod : mods) {
        const std::string name = mod.stem().string();
        const fs::path scratch_dir = scratch_root / name;
        fs::create_directories(scratch_dir);
        run_in_dir(source, {
            shell_quote(MIND_SIM_NMODL_PATH),
            shell_quote(mod.filename()),
            "-o",
            shell_quote(generated_dir),
            "--scratch",
            shell_quote(scratch_dir),
            "--coreneuron",
            "host",
            "--c",
            "passes",
            "--json-ast",
            "--inline",
            "codegen",
            "--force",
        });

        const fs::path generated_cpp = generated_dir / (name + ".cpp");
        const fs::path ast_json = scratch_dir / (name + ".ast.json");
        if (!fs::is_regular_file(generated_cpp)) {
            throw std::runtime_error("NMODL did not generate " + generated_cpp.string());
        }
        if (!fs::is_regular_file(ast_json)) {
            throw std::runtime_error("NMODL did not generate " + ast_json.string());
        }
        units.push_back(StandardModUnit{
            .mod = mod,
            .layout = data_layout_from_ast(read_json(ast_json)),
        });
        const fs::path object_path = object_dir / (name + ".o");
        compile_object_file(generated_cpp, object_path);
        objects.push_back(object_path);
    }

    const fs::path modl_reg_cpp = generated_dir / "mod_func.cpp";
    const fs::path modl_reg_object = object_dir / "mod_func.o";
    write_modl_reg_source(modl_reg_cpp, units);
    compile_object_file(modl_reg_cpp, modl_reg_object);
    objects.push_back(modl_reg_object);

    const fs::path library_path = output_dir / "libcorenrnmech.so";
    link_shared_library(objects, library_path);
    std::cout << library_path << '\n';
}

struct NmodlGenerated {
    fs::path generated_cpp;
    fs::path ast_json;
    JsonValue ast;
};

[[nodiscard]] NmodlGenerated run_nmodl_codegen(const fs::path& mod,
                                               const fs::path& generated_dir,
                                               const fs::path& scratch_dir) {
    fs::create_directories(generated_dir);
    fs::create_directories(scratch_dir);
    run_in_dir(mod.parent_path(), {
        shell_quote(MIND_SIM_NMODL_PATH),
        shell_quote(mod.filename()),
        "-o",
        shell_quote(generated_dir),
        "--scratch",
        shell_quote(scratch_dir),
        "--coreneuron",
        "host",
        "--c",
        "passes",
        "--json-ast",
        "--inline",
        "codegen",
        "--force",
    });

    const std::string name = mod.stem().string();
    const fs::path generated_cpp = generated_dir / (name + ".cpp");
    const fs::path ast_json = scratch_dir / (name + ".ast.json");
    if (!fs::is_regular_file(generated_cpp)) {
        throw std::runtime_error("NMODL did not generate " + generated_cpp.string());
    }
    if (!fs::is_regular_file(ast_json)) {
        throw std::runtime_error("NMODL did not generate " + ast_json.string());
    }
    return NmodlGenerated{
        .generated_cpp = generated_cpp,
        .ast_json = ast_json,
        .ast = read_json(ast_json),
    };
}

[[nodiscard]] bool mod_has_mind_block(const fs::path& mod, const fs::path& output_dir) {
    const std::string name = mod.stem().string();
    const fs::path workdir = output_dir / ".mind_sim_ast" / name;
    fs::remove_all(workdir);
    const auto generated = run_nmodl_codegen(mod, workdir / "out", workdir / "scratch");
    return contains_mind_block_ast(generated.ast);
}

void compile_mod(const fs::path& mod, const fs::path& output_dir) {
    const std::string name = mod.stem().string();
    const fs::path workdir = output_dir / ".mind_sim_build" / name;
    fs::remove_all(workdir);
    const fs::path nmodl_dir = workdir / "nmodl";
    const fs::path scratch_dir = workdir / "scratch";
    const auto generated = run_nmodl_codegen(mod, nmodl_dir, scratch_dir);

    auto params = params_from_ast(generated.ast);
    auto states = states_from_ast(generated.ast);
    apply_initial_defaults_from_ast(generated.ast, states);
    const MindSpec spec = parse_mind_block_ast(generated.ast);
    const MechanismDataLayout layout = data_layout_from_ast(generated.ast);
    if (spec.role == "MACRO2MICRO") {
        validate_micro_input_net_receive_ast(generated.ast);
    } else if (spec.role == "MICRO2MACRO") {
        validate_micro_output_net_receive_ast(generated.ast);
    }

    const fs::path wrapper_cpp = workdir / (name + "_mind.cpp");
    const fs::path object_path = workdir / (name + "_mind.o");
    const fs::path library_path = output_dir / ("libmind_sim_mod_" + name + ".so");
    write_text(wrapper_cpp, render_wrapper(generated.generated_cpp, layout, spec, states, params));
    compile_shared_library(wrapper_cpp, object_path, library_path);
}

[[nodiscard]] Args parse_args(int argc, char** argv) {
    Args args;
    if (argc != 2) {
        throw std::runtime_error("usage: mind_nrnivmodl MOD_DIR_OR_FILE");
    }
    args.source = argv[1];
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const fs::path source = fs::absolute(args.source);
        const fs::path output_dir = output_dir_for(source);
        fs::create_directories(output_dir);
        const auto mods = collect_mods(source);
        bool has_mind_mod = false;
        bool has_standard_mod = false;
        for (const auto& mod : mods) {
            if (mod_has_mind_block(mod, output_dir)) {
                has_mind_mod = true;
            } else {
                has_standard_mod = true;
            }
        }
        if (has_mind_mod && has_standard_mod) {
            throw std::runtime_error("do not mix MIND extension MOD files and standard NEURON MOD files in one directory");
        }
        if (has_standard_mod) {
            compile_standard_mods(source, output_dir);
        } else {
            for (const auto& mod : mods) {
                compile_mod(mod, output_dir);
                std::cout << output_dir / ("libmind_sim_mod_" + mod.stem().string() + ".so") << '\n';
            }
        }
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "mind_nrnivmodl: " << exc.what() << '\n';
        return 1;
    }
}
