#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

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

struct MindSpec {
    std::string role;
};

struct Args {
    fs::path source;
};

struct NeuronBlockSpec {
    std::string mechanism;
    bool point_process{false};
    bool artificial_cell{false};
};

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

[[nodiscard]] std::string shell_quote_text(std::string_view text) {
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

[[nodiscard]] bool using_nvhpc_compiler() {
    const std::string compiler = MIND_SIM_CXX_COMPILER;
    return compiler.find("nvc++") != std::string::npos || compiler.find("pgc++") != std::string::npos;
}

[[nodiscard]] bool same_path(const std::string& lhs, const fs::path& rhs) {
    std::error_code ec;
    if (fs::equivalent(fs::path{lhs}, rhs, ec)) {
        return true;
    }
    return fs::path{lhs}.lexically_normal() == rhs.lexically_normal();
}

[[nodiscard]] std::string subprocess_env_prefix() {
    if (!using_nvhpc_compiler()) {
        return {};
    }
    const char* vars_to_unset[] = {
        "CC",
        "CXX",
        "GCC",
        "LD",
        "CFLAGS",
        "CXXFLAGS",
        "CPPFLAGS",
        "LDFLAGS",
        "LIBRARY_PATH",
        "COMPILER_PATH",
        "GCC_EXEC_PREFIX",
    };

    std::ostringstream prefix;
    prefix << "/usr/bin/env";
    for (const char* var: vars_to_unset) {
        prefix << " -u " << var;
    }

    const char* path_env = std::getenv("PATH");
    const char* conda_prefix_env = std::getenv("CONDA_PREFIX");
    if (path_env == nullptr || conda_prefix_env == nullptr) {
        prefix << ' ';
        return prefix.str();
    }

    const fs::path conda_bin = fs::path{conda_prefix_env} / "bin";
    std::vector<std::string> kept;
    std::string_view path{path_env};
    std::size_t begin = 0;
    while (begin <= path.size()) {
        const std::size_t end = path.find(':', begin);
        std::string entry{path.substr(begin, end == std::string_view::npos ? path.size() - begin : end - begin)};
        if (!entry.empty() && !same_path(entry, conda_bin)) {
            kept.push_back(std::move(entry));
        }
        if (end == std::string_view::npos) {
            break;
        }
        begin = end + 1;
    }
    if (kept.empty()) {
        prefix << " PATH='/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' ";
        return prefix.str();
    }

    std::ostringstream joined;
    for (std::size_t i = 0; i < kept.size(); ++i) {
        if (i != 0) {
            joined << ':';
        }
        joined << kept[i];
    }
    prefix << " PATH=" << shell_quote_text(joined.str()) << ' ';
    return prefix.str();
}

void run(const std::vector<std::string>& command) {
    std::ostringstream shell;
    shell << subprocess_env_prefix();
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

void run_in_dir(const fs::path& directory,
                const std::vector<std::string>& command,
                bool use_subprocess_env_prefix = true) {
    std::ostringstream shell;
    shell << "cd " << shell_quote(directory) << " && ";
    if (use_subprocess_env_prefix) {
        shell << subprocess_env_prefix();
    }
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

[[nodiscard]] std::string cpp_identifier(std::string_view value) {
    std::string out;
    out.reserve(value.size() + 1);
    for (char c : value) {
        const auto uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc) != 0 || c == '_') {
            out += c;
        } else {
            out += '_';
        }
    }
    if (out.empty() || std::isdigit(static_cast<unsigned char>(out.front())) != 0) {
        out.insert(out.begin(), '_');
    }
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

[[nodiscard]] bool contains_name(const std::vector<std::string>& names, const std::string& name) {
    return std::find(names.begin(), names.end(), name) != names.end();
}

void append_unique(std::vector<std::string>& names, std::string name) {
    if (!name.empty() && !contains_name(names, name)) {
        names.push_back(std::move(name));
    }
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

[[nodiscard]] MindSpec parse_mind_block_ast(const JsonValue& ast) {
    const auto blocks = node_payloads(ast, "MindBlock");
    if (blocks.empty()) {
        throw std::runtime_error("MOD file is missing a MIND block");
    }
    if (blocks.size() != 1) {
        throw std::runtime_error("MOD file must contain exactly one MIND block");
    }
    const auto names = string_names(*blocks.front());
    if (names.empty()) {
        throw std::runtime_error("MIND block AST has no ROLE");
    }
    MindSpec spec;
    spec.role = upper(names.front());
    if (spec.role != "REGION" && spec.role != "MACRO2MACRO" && spec.role != "MACRO2MICRO" &&
        spec.role != "MICRO2MACRO") {
        throw std::runtime_error("MIND block AST is not structured or has unsupported ROLE " +
                                 spec.role);
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
    if (spec.mechanism.empty()) {
        throw std::runtime_error("NEURON block must contain SUFFIX, POINT_PROCESS, or ARTIFICIAL_CELL");
    }
    return spec;
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
    const std::vector<std::string> expected{"weight"};
    if (args != expected) {
        throw std::runtime_error("MICRO2MACRO requires NET_RECEIVE(weight)");
    }
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

[[nodiscard]] fs::path current_executable_path() {
#if defined(__linux__)
    std::vector<char> buffer(4096);
    while (true) {
        const ssize_t size = readlink("/proc/self/exe", buffer.data(), buffer.size());
        if (size < 0) {
            return {};
        }
        if (static_cast<std::size_t>(size) < buffer.size()) {
            return fs::path(std::string(buffer.data(), static_cast<std::size_t>(size)));
        }
        buffer.resize(buffer.size() * 2);
    }
#else
    return {};
#endif
}

[[nodiscard]] fs::path current_executable_dir() {
    const fs::path exe = current_executable_path();
    if (!exe.empty()) {
        return exe.parent_path();
    }
    return fs::current_path();
}

[[nodiscard]] fs::path packaged_root_dir() {
    const fs::path exe_dir = current_executable_dir();
    if (exe_dir.filename() == "bin") {
        return exe_dir.parent_path();
    }
    return exe_dir;
}

[[nodiscard]] fs::path existing_or_empty(const fs::path& path) {
    std::error_code ec;
    if (fs::exists(path, ec)) {
        return path;
    }
    return {};
}

[[nodiscard]] fs::path existing_file_or_empty(const fs::path& path) {
    std::error_code ec;
    if (fs::is_regular_file(path, ec)) {
        return path;
    }
    return {};
}

[[nodiscard]] fs::path resolved_nmodl_path() {
    if (auto path = existing_file_or_empty(current_executable_dir() / "nmodl"); !path.empty()) {
        return path;
    }
    if (auto path = existing_file_or_empty(fs::path{MIND_SIM_NMODL_PATH}); !path.empty()) {
        return path;
    }
    throw std::runtime_error("could not locate MIND-aware nmodl executable");
}

[[nodiscard]] fs::path resolved_source_include_dir() {
    if (auto path = existing_or_empty(fs::path{MIND_SIM_SOURCE_DIR} / "src"); !path.empty()) {
        return path;
    }
    if (auto path = existing_or_empty(packaged_root_dir() / "include" / "src"); !path.empty()) {
        return path;
    }
    throw std::runtime_error("could not locate MIND_Sim source include directory");
}

[[nodiscard]] fs::path resolved_mechanism_include_dir() {
    if (auto path = existing_or_empty(fs::path{MIND_SIM_MECHANISM_INCLUDE_DIR}); !path.empty()) {
        return path;
    }
    if (auto path = existing_or_empty(packaged_root_dir() / "include" / "src" / "micro" / "sim"); !path.empty()) {
        return path;
    }
    throw std::runtime_error("could not locate MIND_Sim mechanism include directory");
}

[[nodiscard]] fs::path resolved_randoms_include_dir() {
    if (auto path = existing_or_empty(resolved_source_include_dir() / "micro" / "sim" /
                                      "coreneuron" / "utils" / "randoms");
        !path.empty()) {
        return path;
    }
    throw std::runtime_error("could not locate CoreNEURON randoms include directory");
}

[[nodiscard]] fs::path resolved_eigen_include_dir() {
    if (auto path = existing_or_empty(fs::path{MIND_SIM_SOURCE_DIR} / "src" / "mod" /
                                      "neuron" / "external" / "eigen");
        !path.empty()) {
        return path;
    }
    if (auto path = existing_or_empty(packaged_root_dir() / "include" / "src" / "mod" /
                                      "neuron" / "external" / "eigen");
        !path.empty()) {
        return path;
    }
    throw std::runtime_error("could not locate Eigen include directory");
}

void compile_object_file(const fs::path& source_path, const fs::path& object_path) {
    fs::create_directories(object_path.parent_path());
    std::vector<std::string> command{
        shell_quote(MIND_SIM_CXX_COMPILER),
        "-std=c++20",
        "-O2",
        "-fPIC",
        MIND_SIM_MOD_CXX_FLAGS,
        include_arg(resolved_source_include_dir()),
        include_arg(resolved_mechanism_include_dir()),
        include_arg(resolved_randoms_include_dir()),
        include_arg(resolved_eigen_include_dir()),
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
#if defined(MIND_SIM_ENABLE_GPU)
        "-DCORENEURON_ENABLE_GPU",
        "-DR123_USE_INTRIN_H=0",
        "-DEIGEN_DONT_VECTORIZE=1",
#endif
        "-c",
        shell_quote(source_path),
        "-o",
        shell_quote(object_path),
    };
    run(command);
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

struct StandardModUnit {
    fs::path mod;
};

struct MindRuleUnit {
    fs::path mod;
    std::string role;
    std::string descriptor_symbol;
    std::string apply_symbol;
};

[[nodiscard]] const char* apply_context_type(const std::string& role) {
    if (role == "REGION") {
        return "mind_sim::mod::AbiRegionContext";
    }
    if (role == "MACRO2MACRO") {
        return "mind_sim::mod::AbiMacroToMacroContext";
    }
    if (role == "MACRO2MICRO") {
        return "mind_sim::mod::AbiMicroInputContext";
    }
    if (role == "MICRO2MACRO") {
        return "mind_sim::mod::AbiMicroOutputContext";
    }
    throw std::runtime_error("unsupported MIND ROLE " + role);
}

void write_modl_reg_source(const fs::path& path, const std::vector<StandardModUnit>& units) {
    std::ostringstream out;
    out << "#include <cstdio>\n";
    out << "#include <coreneuron/mechanism/membfunc.hpp>\n";
    out << "namespace coreneuron {\n";
    out << "extern int nrnmpi_myid;\n";
    out << "extern int nrn_nobanner_;\n";
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
    }
    out << "}\n";
    out << "}\n";
    write_text(path, out.str());
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
    std::vector<std::string> command{
        shell_quote(resolved_nmodl_path()),
        shell_quote(mod.filename()),
        "-o",
        shell_quote(generated_dir),
        "--scratch",
        shell_quote(scratch_dir),
        "--coreneuron",
#if defined(MIND_SIM_ENABLE_GPU)
        "acc",
        "--oacc",
#else
        "host",
        "--c",
#endif
        "passes",
        "--json-ast",
        "--inline",
        "codegen",
        "--force",
    };
    run_in_dir(mod.parent_path(), command, false);

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

void write_rule_registry_source(const fs::path& path, const std::vector<MindRuleUnit>& rules) {
    std::ostringstream out;
    out << "#include \"mod/abi.hpp\"\n\n";
    out << "#if defined(_WIN32)\n";
    out << "#define MIND_RULE_EXPORT __declspec(dllexport)\n";
    out << "#else\n";
    out << "#define MIND_RULE_EXPORT __attribute__((visibility(\"default\")))\n";
    out << "#endif\n\n";
    for (const auto& rule : rules) {
        out << "extern \"C\" const mind_sim::mod::AbiRuleDescriptor* "
            << rule.descriptor_symbol << "();\n";
        out << "extern \"C\" void " << rule.apply_symbol << "(const "
            << apply_context_type(rule.role) << "*);\n";
    }
    out << "\nnamespace {\n";
    if (!rules.empty()) {
        out << "const mind_sim::mod::AbiRuleEntry kRules[] = {\n";
        for (const auto& rule : rules) {
            out << "    {\n";
            out << "        .descriptor = " << rule.descriptor_symbol << "(),\n";
            out << "        .macro_to_macro_apply = "
                << (rule.role == "MACRO2MACRO" ? rule.apply_symbol : "nullptr") << ",\n";
            out << "        .micro_input_apply = "
                << (rule.role == "MACRO2MICRO" ? rule.apply_symbol : "nullptr") << ",\n";
            out << "        .micro_output_apply = "
                << (rule.role == "MICRO2MACRO" ? rule.apply_symbol : "nullptr") << ",\n";
            out << "        .region_apply = "
                << (rule.role == "REGION" ? rule.apply_symbol : "nullptr") << ",\n";
            out << "    },\n";
        }
        out << "};\n\n";
    }
    out << "const mind_sim::mod::AbiRuleRegistry kRegistry{\n";
    out << "    .abi_version = mind_sim::mod::kModAbiVersion,\n";
    out << "    .rule_count = " << rules.size() << ",\n";
    out << "    .rules = " << (rules.empty() ? "nullptr" : "kRules") << ",\n";
    out << "};\n";
    out << "}  // namespace\n\n";
    out << "extern \"C\" MIND_RULE_EXPORT const mind_sim::mod::AbiRuleRegistry* "
           "mind_rule_registry() {\n";
    out << "    return &kRegistry;\n";
    out << "}\n";
    write_text(path, out.str());
}

void compile_mods(const fs::path& source, const fs::path& output_dir) {
    const auto mods = collect_mods(source);
    const fs::path workdir = output_dir / "corenrn";
    fs::remove_all(workdir);
    const fs::path generated_dir = workdir / "mod2c";
    const fs::path scratch_root = workdir / "scratch";
    const fs::path object_dir = workdir / "build";
    fs::create_directories(generated_dir);
    fs::create_directories(scratch_root);
    fs::create_directories(object_dir);

    std::vector<fs::path> objects;
    std::vector<StandardModUnit> units;
    std::vector<MindRuleUnit> rules;
    objects.reserve(mods.size() + 2);
    units.reserve(mods.size());

    for (const auto& mod : mods) {
        const std::string name = mod.stem().string();
        const auto generated =
            run_nmodl_codegen(mod, generated_dir, scratch_root / cpp_identifier(name));
        units.push_back(StandardModUnit{.mod = mod});

        fs::path source_cpp = generated.generated_cpp;
        if (contains_mind_block_ast(generated.ast)) {
            const MindSpec spec = parse_mind_block_ast(generated.ast);
            if (spec.role == "MACRO2MICRO") {
                validate_micro_input_net_receive_ast(generated.ast);
            } else if (spec.role == "MICRO2MACRO") {
                validate_micro_output_net_receive_ast(generated.ast);
            }

            const auto neuron = parse_neuron_block_ast(generated.ast);
            const std::string safe = cpp_identifier(neuron.mechanism);
            const std::string descriptor_symbol = "mind_rule_descriptor_" + safe;
            const std::string apply_symbol = "mind_rule_apply_" + safe;
            rules.push_back(MindRuleUnit{
                .mod = mod,
                .role = spec.role,
                .descriptor_symbol = descriptor_symbol,
                .apply_symbol = apply_symbol,
            });
        }

        const fs::path object_path = object_dir / (name + ".o");
        compile_object_file(source_cpp, object_path);
        objects.push_back(object_path);
    }

    const fs::path modl_reg_cpp = generated_dir / "mod_func.cpp";
    const fs::path modl_reg_object = object_dir / "mod_func.o";
    write_modl_reg_source(modl_reg_cpp, units);
    compile_object_file(modl_reg_cpp, modl_reg_object);
    objects.push_back(modl_reg_object);

    const fs::path registry_cpp = generated_dir / "mind_rule_registry.cpp";
    const fs::path registry_object = object_dir / "mind_rule_registry.o";
    write_rule_registry_source(registry_cpp, rules);
    compile_object_file(registry_cpp, registry_object);
    objects.push_back(registry_object);

    const fs::path library_path = output_dir / "libcorenrnmech.so";
    link_shared_library(objects, library_path);
    std::cout << library_path << '\n';
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
        compile_mods(source, output_dir);
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "mind_nrnivmodl: " << exc.what() << '\n';
        return 1;
    }
}
