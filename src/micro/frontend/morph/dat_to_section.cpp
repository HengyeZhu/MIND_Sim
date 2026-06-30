// Module 1/4: SWC parsing + sectionify (SWC -> Morph::sections + Morph::datas)

#include "dat_to_section.hpp"

#include "label_utils.hpp"
#include "mutable_section.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <stack>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

namespace mind_micro_morph {

namespace {

std::string swc_label_to_name_local(std::int32_t label) {
    switch (label) {
        case 1:
            return "soma";
        case 2:
            return "axon";
        case 3:
            return "dend";
        case 4:
            return "apic";
        default:
            return std::to_string(label);
    }
}

std::string label_prefix(std::string_view label) {
    if (mind_micro_labels::is_soma_label(label)) {
        return "soma";
    }
    if (mind_micro_labels::is_axon_label(label)) {
        return "axon";
    }
    if (mind_micro_labels::is_dend_label(label)) {
        return "dend";
    }
    if (mind_micro_labels::is_apic_label(label)) {
        return "apic";
    }
    return "sec";
}

void assign_default_section_names(Morph& morph) {
    std::unordered_map<std::string, int> counts;
    for (auto& sec : morph.sections) {
        if (!sec.name.empty()) {
            continue;
        }
        const std::string label_key = sec.label;
        auto& idx = counts[label_key];
        const std::string prefix = label_prefix(label_key);
        std::string name;
        if (prefix == "sec") {
            name = prefix + "_" + label_key + "_" + std::to_string(idx);
        } else {
            name = prefix + "_" + std::to_string(idx);
        }
        sec.name = std::move(name);
        ++idx;
    }
}

}  // namespace

static inline bool is_ascii_space(char c) noexcept {
    // SWC is ASCII text; keep this fast and locale-independent.
    switch (c) {
        case ' ':
        case '\t':
        case '\n':
        case '\r':
        case '\f':
        case '\v':
            return true;
        default:
            return false;
    }
}

static bool is_blank_or_comment(const std::string& line) {
    for (char c : line) {
        if (is_ascii_space(c)) {
            continue;
        }
        return c == '#';
    }
    return true;
}

static inline void skip_spaces(const char*& p, const char* end) noexcept {
    while (p < end && is_ascii_space(*p)) {
        ++p;
    }
}

template <typename Int>
static bool parse_int(const char*& p, const char* end, Int& out) noexcept {
    static_assert(std::is_integral_v<Int>, "parse_int requires an integral type");
    skip_spaces(p, end);
    if (p >= end) {
        return false;
    }
    const auto res = std::from_chars(p, end, out);
    if (res.ec != std::errc{}) {
        return false;
    }
    p = res.ptr;
    return true;
}

static bool parse_float(const char*& p, const char* end, float& out) noexcept {
    skip_spaces(p, end);
    if (p >= end) {
        return false;
    }
    char* parsed_end = nullptr;
    out = std::strtof(p, &parsed_end);
    if (parsed_end == p) {
        return false;
    }
    p = parsed_end;
    return true;
}

// Sectionify implementation notes:
// - SWC input is treated as an arbitrary forest: ids can have gaps and lines can be in any order.
// - We build an id->point map and parent/children adjacency, then do a MorphIO-style
//   DFS pre-order traversal that merges maximal same-label unary chains into Sections.
// - We intentionally do not sort samples by id or allocate id-sized index vectors.
// - We keep only minimal soma-connection behavior (e.g. single-point soma attaches at parentx=0.5)
//   so existing node-geometry/correctness checks keep working on the project reference SWCs.

Morph load_swc_morphology(const std::string& swc_path) {
    std::ifstream file(swc_path);

    std::vector<SwcData> swc_data;
    std::vector<origin_id> swc_origin_ids;
    std::vector<origin_id> swc_parent_origin_ids;
    std::string line;
    std::int32_t line_no = 0;
    while (std::getline(file, line)) {
        ++line_no;
        if (is_blank_or_comment(line)) {
            continue;
        }
        origin_id origin{};
        origin_id parent_origin{};
        std::int32_t label{};
        float x{};
        float y{};
        float z{};
        float radius_um{};

        const char* p = line.c_str();
        const char* const end = p + line.size();
        parse_int(p, end, origin);
        parse_int(p, end, label);
        parse_float(p, end, x);
        parse_float(p, end, y);
        parse_float(p, end, z);
        parse_float(p, end, radius_um);
        parse_int(p, end, parent_origin);
        SwcData data{};
        data.xyz = {static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)};
        data.d_um = static_cast<double>(radius_um) * 2.0;
        data.label = static_cast<std::int8_t>(label);
        swc_data.push_back(data);
        swc_origin_ids.push_back(origin);
        swc_parent_origin_ids.push_back(parent_origin);
    }

    Morph morph{};
    if (swc_data.empty()) {
        return morph;
    }

    const std::size_t num_points = swc_data.size();
    const auto num_points_id = static_cast<swcdata_id>(num_points);

    // Map SWC-origin ids (declared in file) to our internal swc point ids (swcdata_id).
    // Unlike NEURON's Import3d, we do NOT sort by id and we do NOT allocate an
    // id-sized index vector (which can be very slow / memory-heavy for large ids with gaps).
    // This also allows forward references (a child line can appear before its parent line).
    std::unordered_map<origin_id, swcdata_id> origin2swc_id{};
    for (swcdata_id point_id = 0; point_id < num_points_id; ++point_id) {
        const origin_id origin = swc_origin_ids[static_cast<std::size_t>(point_id)];
        const auto [it, inserted] = origin2swc_id.emplace(origin, point_id);
    }

    // Resolve parent indices once up-front (to avoid repeated unordered_map lookups).
    // parent_of[i] == invalid_swcdata_id means "root".
    std::vector<swcdata_id> parent_of(num_points, invalid_swcdata_id);
    std::vector<swcdata_id> roots;

    for (swcdata_id point_id = 0; point_id < num_points_id; ++point_id) {
        const origin_id parent_origin = swc_parent_origin_ids[static_cast<std::size_t>(point_id)];
        if (parent_origin < 0) {
            roots.push_back(point_id);
            continue;
        }
        const auto parent_it = origin2swc_id.find(parent_origin);
        const auto parent_point_id = parent_it->second;
        parent_of[static_cast<std::size_t>(point_id)] = parent_point_id;
    }

    auto parent_swc_id = [&](swcdata_id point_id) -> swcdata_id {
        return parent_of[static_cast<std::size_t>(point_id)];
    };

    std::vector<std::uint32_t> nchild_soma(num_points, 0);
    for (swcdata_id point_id = 0; point_id < num_points_id; ++point_id) {
        if (swc_data[static_cast<std::size_t>(point_id)].label != 1) {
            continue;
        }
        const auto parent_point_id = parent_swc_id(point_id);
        if (parent_point_id != invalid_swcdata_id &&
            swc_data[static_cast<std::size_t>(parent_point_id)].label == 1) {
            ++nchild_soma[static_cast<std::size_t>(parent_point_id)];
        }
    }

    // DFS sectionify (MorphIO-style): build maximal same-label chains until a branch, leaf,
    // or label-change, then recurse on children. Unlike NEURON, we do not require SWC ids
    // to be sorted or parents to appear before children in the file.

    morph.sections.clear();
    morph.datas.clear();
    morph.label_names.clear();
    morph.label_index.clear();
    morph.sections_by_label.clear();

    std::vector<std::vector<swcdata_id>> swc_point_children(num_points);
    for (swcdata_id point_id = 0; point_id < num_points_id; ++point_id) {
        const auto parent_swc_point_id = parent_of[static_cast<std::size_t>(point_id)];
        if (parent_swc_point_id != invalid_swcdata_id) {
            swc_point_children[static_cast<std::size_t>(parent_swc_point_id)].push_back(point_id);
        }
    }

    std::vector<section_id> swc_point_to_section_id(num_points, invalid_section_id);
    std::stack<swcdata_id> dfs_stack;
    for (auto it = roots.rbegin(); it != roots.rend(); ++it) {
        dfs_stack.push(*it);
    }

    std::vector<swcdata_id> chain;

    while (!dfs_stack.empty()) {
        const auto head_swc_point_id = dfs_stack.top();
        dfs_stack.pop();

        const auto parent_swc_point_id = parent_swc_id(head_swc_point_id);
        const bool is_root = (parent_swc_point_id == invalid_swcdata_id);

        chain.clear();
        chain.push_back(head_swc_point_id);
        swcdata_id tail_swc_point_id = head_swc_point_id;

        for (;;) {
            const auto& tail_child_points = swc_point_children[static_cast<std::size_t>(tail_swc_point_id)];
            if (tail_child_points.size() != 1) {
                break;
            }
            const auto next_swc_point_id = tail_child_points.front();
            if (swc_data[static_cast<std::size_t>(next_swc_point_id)].label !=
                swc_data[static_cast<std::size_t>(tail_swc_point_id)].label) {
                break;
            }
            tail_swc_point_id = next_swc_point_id;
            chain.push_back(tail_swc_point_id);
        }

        Section sec{};
        sec.head_swc_id = head_swc_point_id;
        sec.label = swc_label_to_name_local(swc_data[static_cast<std::size_t>(head_swc_point_id)].label);
        sec.offset = morph.datas.size();
        sec.rallbranch = 1.0;
        // `nseg` is assigned later by an explicit nseg policy stage (fixed/target_length).
        sec.nseg = 0;

        if (is_root) {
            sec.parent_sec_id = invalid_section_id;
            sec.parentx = 1.0;
            sec.wire_first = false;
            sec.parent_swc_id = invalid_swcdata_id;
            for (const auto chain_swc_point_id : chain) {
                morph.datas.push_back(chain_swc_point_id);
            }
        } else {
            sec.parent_swc_id = parent_swc_point_id;
            const section_id parent_section_id = swc_point_to_section_id[static_cast<std::size_t>(parent_swc_point_id)];
            sec.parent_sec_id = parent_section_id;
            sec.parentx = 1.0;
            sec.wire_first = false;

            const bool parent_section_is_soma = mind_micro_labels::is_soma_label(morph.sections[parent_section_id].label);
            const bool is_non_soma = !mind_micro_labels::is_soma_label(sec.label);

            if (morph.sections[parent_section_id].parent_sec_id == invalid_section_id) {
                const bool parent_section_is_single_point_soma = (morph.sections[parent_section_id].count == 1);
                if (parent_section_is_soma && is_non_soma && parent_section_is_single_point_soma) {
                    sec.parentx = 0.5;
                    if (chain.size() > 1) {
                        sec.wire_first = true;
                    }
                } else if (parent_swc_point_id == morph.sections[parent_section_id].head_swc_id) {
                    sec.parentx = 0.0;
                    if (is_non_soma && nchild_soma[static_cast<std::size_t>(parent_swc_point_id)] > 1) {
                        sec.wire_first = true;
                    }
                }
            }
            if (!sec.wire_first) {
                const auto& parent_pt = swc_data[static_cast<std::size_t>(parent_swc_point_id)];
                const auto& head_pt = swc_data[static_cast<std::size_t>(head_swc_point_id)];
                if (parent_pt.xyz != head_pt.xyz) {
                    morph.datas.push_back(parent_swc_point_id);
                }
            }
            for (const auto chain_swc_point_id : chain) {
                morph.datas.push_back(chain_swc_point_id);
            }
        }

        sec.count = morph.datas.size() - sec.offset;
        const auto new_section_id = static_cast<section_id>(morph.sections.size());
        morph.sections.push_back(std::move(sec));

        // Maintain a label->sections lookup table during the initial build, so
        // channel insertion planning ("for label in ...") can avoid an extra pass.
        const auto label_u = ensure_section_label_index(morph, morph.sections[new_section_id].label);
        morph.sections_by_label[label_u].push_back(new_section_id);

        for (const auto chain_swc_point_id : chain) {
            swc_point_to_section_id[static_cast<std::size_t>(chain_swc_point_id)] = new_section_id;
        }

        const auto& tail_child_points = swc_point_children[static_cast<std::size_t>(tail_swc_point_id)];
        for (auto it = tail_child_points.rbegin(); it != tail_child_points.rend(); ++it) {
            dfs_stack.push(*it);
        }
    }


    auto add_child_sorted = [&](section_id parent_section_id, section_id child_section_id) {
        auto& child_section_ids = morph.sections[parent_section_id].children;
        const double child_parentx = morph.sections[child_section_id].parentx;
        auto it = child_section_ids.begin();
        for (; it != child_section_ids.end(); ++it) {
            const double existing_parentx = morph.sections[*it].parentx;
            if (child_parentx <= existing_parentx) {
                child_section_ids.insert(it, child_section_id);
                return;
            }
        }
        child_section_ids.push_back(child_section_id);
    };

    for (section_id child_section_id = 0; child_section_id < morph.sections.size(); ++child_section_id) {
        const auto parent_section_id = morph.sections[child_section_id].parent_sec_id;
        if (parent_section_id != invalid_section_id) {
            add_child_sorted(parent_section_id, child_section_id);
        }
    }

    assign_default_section_names(morph);
    morph.swc = std::move(swc_data);

    return morph;
}

std::vector<section_id> sections_with_label(const Morph& morph, std::string_view label) {
    if (label.empty()) {
        return {};
    }
    const auto it = morph.label_index.find(std::string(label));
    if (it == morph.label_index.end()) {
        return {};
    }
    const auto label_u = it->second;
    if (label_u >= morph.sections_by_label.size()) {
        return {};
    }
    return morph.sections_by_label[label_u];
}

}  // namespace mind_micro_morph

namespace mind_micro_morph {
namespace asc_detail {
namespace {

enum class Tok {
    lparen,
    rparen,
    pipe,
    lt,
    gt,
    comma,
    number,
    symbol,
    string,
    end,
};

struct Token {
    Tok kind{Tok::end};
    std::string text{};
    int line{1};
    int column{1};
    float number_value{0.0f};
};

inline bool is_ascii_space(char c) noexcept {
    switch (c) {
        case ' ':
        case '\t':
        case '\n':
        case '\r':
        case '\f':
        case '\v':
            return true;
        default:
            return false;
    }
}

inline bool is_symbol_delim(char c) noexcept {
    return is_ascii_space(c) || c == ';' || c == '(' || c == ')' || c == '|' || c == '<' || c == '>' ||
           c == ',' || c == '"';
}

inline char ascii_upper(char c) noexcept {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
}

inline bool equals_no_space_ascii_case(std::string_view s, std::string_view needle_upper_no_space) noexcept {
    std::size_t j = 0;
    for (char c : s) {
        if (is_ascii_space(c)) {
            continue;
        }
        if (j >= needle_upper_no_space.size()) {
            return false;
        }
        if (ascii_upper(c) != needle_upper_no_space[j]) {
            return false;
        }
        ++j;
    }
    return j == needle_upper_no_space.size();
}

inline bool is_number_start(const std::string& text, std::size_t i) {
    const char c = text[i];
    if (std::isdigit(static_cast<unsigned char>(c))) {
        return true;
    }
    if (c == '.') {
        return (i + 1 < text.size()) && std::isdigit(static_cast<unsigned char>(text[i + 1]));
    }
    if (c == '+' || c == '-') {
        if (i + 1 >= text.size()) {
            return false;
        }
        const char n1 = text[i + 1];
        if (std::isdigit(static_cast<unsigned char>(n1))) {
            return true;
        }
        return (n1 == '.' && (i + 2 < text.size()) && std::isdigit(static_cast<unsigned char>(text[i + 2])));
    }
    return false;
}

std::vector<Token> tokenize_asc(const std::string& text) {
    std::vector<Token> tokens;

    std::size_t i = 0;
    int line = 1;
    int col = 1;

    auto advance_char = [&](char c) {
        if (c == '\n') {
            ++line;
            col = 1;
        } else {
            ++col;
        }
    };

    while (i < text.size()) {
        const char c = text[i];

        if (c == ';') {
            while (i < text.size() && text[i] != '\n') {
                ++i;
                ++col;
            }
            continue;
        }

        if (is_ascii_space(c)) {
            advance_char(c);
            ++i;
            continue;
        }

        const int tok_line = line;
        const int tok_col = col;

        switch (c) {
            case '(':
                tokens.push_back(Token{Tok::lparen, "(", tok_line, tok_col});
                ++i;
                ++col;
                continue;
            case ')':
                tokens.push_back(Token{Tok::rparen, ")", tok_line, tok_col});
                ++i;
                ++col;
                continue;
            case '|':
                tokens.push_back(Token{Tok::pipe, "|", tok_line, tok_col});
                ++i;
                ++col;
                continue;
            case '<':
                tokens.push_back(Token{Tok::lt, "<", tok_line, tok_col});
                ++i;
                ++col;
                continue;
            case '>':
                tokens.push_back(Token{Tok::gt, ">", tok_line, tok_col});
                ++i;
                ++col;
                continue;
            case ',':
                tokens.push_back(Token{Tok::comma, ",", tok_line, tok_col});
                ++i;
                ++col;
                continue;
            case '"': {
                ++i;
                ++col;
                std::size_t start = i;
                while (i < text.size() && text[i] != '"') {
                    advance_char(text[i]);
                    ++i;
                }
                if (i >= text.size()) {
                    throw std::runtime_error("ASC parse error: unterminated string literal");
                }
                const auto len = i - start;
                tokens.push_back(Token{Tok::string, text.substr(start, len), tok_line, tok_col});
                ++i;
                ++col;
                continue;
            }
            default:
                break;
        }

        if (is_number_start(text, i)) {
            const char* begin_ptr = text.c_str() + i;
            char* end_ptr = nullptr;
            const float parsed = std::strtof(begin_ptr, &end_ptr);
            if (end_ptr != begin_ptr) {
                const auto n = static_cast<std::size_t>(end_ptr - begin_ptr);
                tokens.push_back(Token{Tok::number, {}, tok_line, tok_col, parsed});
                i += n;
                col += static_cast<int>(n);
                continue;
            }
        }

        std::size_t start = i;
        while (i < text.size() && !is_symbol_delim(text[i])) {
            ++i;
            ++col;
        }
        tokens.push_back(Token{Tok::symbol, text.substr(start, i - start), tok_line, tok_col});
    }

    tokens.push_back(Token{Tok::end, "", line, col});
    return tokens;
}

struct AscPoint {
    float x_um{};
    float y_um{};
    float z_um{};
    float diam_um{};
};

struct AscBranch {
    std::vector<AscPoint> points{};
    std::vector<AscBranch> children{};
};

enum class AscTreeType {
    unknown,
    soma,
    axon,
    dend,
    apic,
};

struct AscTree {
    AscTreeType type{AscTreeType::unknown};
    std::string label{};
    AscBranch root{};
};

AscTreeType classify_label(std::string_view label) {
    if (equals_no_space_ascii_case(label, "CELLBODY")) {
        return AscTreeType::soma;
    }
    if (equals_no_space_ascii_case(label, "AXON")) {
        return AscTreeType::axon;
    }
    if (equals_no_space_ascii_case(label, "DENDRITE")) {
        return AscTreeType::dend;
    }
    if (equals_no_space_ascii_case(label, "APICAL")) {
        return AscTreeType::apic;
    }
    return AscTreeType::unknown;
}

bool is_branch_end_symbol(std::string_view s) {
    return equals_no_space_ascii_case(s, "GENERATED") || equals_no_space_ascii_case(s, "HIGH") ||
           equals_no_space_ascii_case(s, "INCOMPLETE") || equals_no_space_ascii_case(s, "LOW") ||
           equals_no_space_ascii_case(s, "NORMAL") || equals_no_space_ascii_case(s, "MIDPOINT") ||
           equals_no_space_ascii_case(s, "ORIGIN");
}

class AscParser {
  public:
    explicit AscParser(std::vector<Token> tokens)
        : tokens_(std::move(tokens)) {}

    std::vector<AscTree> parse() {
        std::vector<AscTree> trees;
        while (!at_end()) {
            if (current().kind != Tok::lparen) {
                consume();
                continue;
            }
            AscTree tree = parse_root_expression();
            trees.push_back(std::move(tree));
        }
        return trees;
    }

  private:
    std::vector<Token> tokens_{};
    std::size_t pos_{0};

    const Token& current() const {
        return tokens_[pos_];
    }

    const Token& peek(std::size_t n = 1) const {
        const auto idx = pos_ + n;
        if (idx >= tokens_.size()) {
            return tokens_.back();
        }
        return tokens_[idx];
    }

    bool at_end() const {
        return current().kind == Tok::end;
    }

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("ASC parse error at line " + std::to_string(current().line) + ":" +
                                 std::to_string(current().column) + ": " + msg);
    }

    void consume() {
        if (!at_end()) {
            ++pos_;
        }
    }

    void expect(Tok kind, const char* what) {
        if (current().kind != kind) {
            fail(std::string("expected ") + what + ", got '" + current().text + "'");
        }
    }

    void skip_balanced_sexp() {
        expect(Tok::lparen, "'('");
        int depth = 0;
        while (!at_end()) {
            if (current().kind == Tok::lparen) {
                ++depth;
            } else if (current().kind == Tok::rparen) {
                --depth;
            }
            consume();
            if (depth == 0) {
                return;
            }
        }
        fail("unexpected EOF while skipping parenthesized expression");
    }

    void skip_spine() {
        // Spine payload is enclosed in < ... >.
        // We only need section topology/pt3d, so this can be ignored.
        if (current().kind != Tok::lt) {
            return;
        }
        consume();
        while (!at_end() && current().kind != Tok::gt) {
            consume();
        }
        if (current().kind == Tok::gt) {
            consume();
        }
    }

    float parse_float_token() {
        expect(Tok::number, "numeric token");
        const float v = current().number_value;
        consume();
        return v;
    }

    AscPoint parse_point() {
        expect(Tok::lparen, "'(' before point");
        consume();

        AscPoint p{};
        p.x_um = parse_float_token();
        p.y_um = parse_float_token();
        p.z_um = parse_float_token();

        // Marker payload may omit diameter. For neurites, diameter is expected.
        if (current().kind == Tok::number) {
            p.diam_um = parse_float_token();
        } else {
            p.diam_um = 0.0f;
        }

        while (current().kind == Tok::symbol || current().kind == Tok::string || current().kind == Tok::comma) {
            consume();
        }

        expect(Tok::rparen, "')' after point");
        consume();
        return p;
    }

    std::vector<AscBranch> parse_children_group() {
        expect(Tok::lparen, "'(' starting branch split");
        consume();

        std::vector<AscBranch> children;
        while (!at_end()) {
            if (current().kind == Tok::rparen) {
                consume();
                break;
            }

            if (current().kind == Tok::pipe) {
                consume();
                continue;
            }

            AscBranch child{};
            parse_branch(child);
            if (!child.points.empty() || !child.children.empty()) {
                children.push_back(std::move(child));
            }

            if (current().kind == Tok::pipe) {
                consume();
                continue;
            }
            if (current().kind == Tok::lparen) {
                continue;
            }
            if (current().kind == Tok::rparen) {
                consume();
                break;
            }

            if (at_end()) {
                break;
            }
            consume();
        }
        return children;
    }

    void parse_branch(AscBranch& out) {
        while (!at_end()) {
            if (current().kind == Tok::pipe || current().kind == Tok::rparen) {
                return;
            }

            if (current().kind == Tok::symbol && is_branch_end_symbol(current().text)) {
                consume();
                continue;
            }

            if (current().kind == Tok::lt) {
                skip_spine();
                continue;
            }

            if (current().kind == Tok::lparen) {
                const auto pk = peek();
                if (pk.kind == Tok::number) {
                    out.points.push_back(parse_point());
                    continue;
                }
                if (pk.kind == Tok::lparen) {
                    auto kids = parse_children_group();
                    out.children.insert(out.children.end(),
                                        std::make_move_iterator(kids.begin()),
                                        std::make_move_iterator(kids.end()));
                    continue;
                }

                // Marker/meta expressions are ignored for morphology topology.
                skip_balanced_sexp();
                continue;
            }

            consume();
        }
    }

    void parse_root_header(AscTree& tree) {
        while (!at_end()) {
            if (current().kind == Tok::rparen) {
                return;
            }

            if (current().kind == Tok::string) {
                tree.label = current().text;
                const auto maybe_type = classify_label(tree.label);
                if (maybe_type != AscTreeType::unknown) {
                    tree.type = maybe_type;
                }
                consume();
                continue;
            }

            if (current().kind == Tok::symbol) {
                const auto maybe_type = classify_label(current().text);
                if (maybe_type != AscTreeType::unknown) {
                    tree.type = maybe_type;
                    tree.label = current().text;
                }
                consume();
                continue;
            }

            if (current().kind == Tok::lparen) {
                const auto pk = peek();
                if (pk.kind == Tok::number) {
                    return;
                }

                if (pk.kind == Tok::symbol) {
                    const auto maybe_type = classify_label(pk.text);
                    if (maybe_type != AscTreeType::unknown) {
                        tree.type = maybe_type;
                        tree.label = pk.text;
                        consume();  // '('
                        consume();  // label symbol
                        while (!at_end() && current().kind != Tok::rparen) {
                            consume();
                        }
                        if (current().kind == Tok::rparen) {
                            consume();
                        }
                        continue;
                    }
                }

                skip_balanced_sexp();
                continue;
            }

            consume();
        }
    }

    AscTree parse_root_expression() {
        expect(Tok::lparen, "'(' starting top-level expression");
        consume();

        AscTree tree{};
        parse_root_header(tree);

        if (current().kind != Tok::rparen) {
            parse_branch(tree.root);
        }

        expect(Tok::rparen, "')' ending top-level expression");
        consume();

        return tree;
    }
};

bool same_xyz(const AscPoint& a, const AscPoint& b) {
    // Match Import3d_Neurolucida3::need_extra() equality checks.
    return a.x_um == b.x_um && a.y_um == b.y_um && a.z_um == b.z_um;
}

struct Vec3d {
    double x{};
    double y{};
    double z{};
};

double point_distance(const Vec3d& a, const Vec3d& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    const double dz = a.z - b.z;
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

double dot(const Vec3d& a, const Vec3d& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

double norm(const Vec3d& v) {
    return std::sqrt(dot(v, v));
}

Vec3d normalize(Vec3d v) {
    const double m = norm(v);
    if (m <= 1e-12 || !std::isfinite(m)) {
        return Vec3d{0.0, 0.0, 0.0};
    }
    const double inv = 1.0 / m;
    v.x *= inv;
    v.y *= inv;
    v.z *= inv;
    return v;
}

Vec3d add(const Vec3d& a, const Vec3d& b) {
    return Vec3d{a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3d mul(const Vec3d& v, double s) {
    return Vec3d{v.x * s, v.y * s, v.z * s};
}

std::vector<Vec3d> contour_uniform_samples(const std::vector<AscPoint>& contour_points, std::size_t sample_count) {
    std::vector<Vec3d> points;
    for (const auto& p : contour_points) {
        points.push_back(
            Vec3d{static_cast<double>(p.x_um), static_cast<double>(p.y_um), static_cast<double>(p.z_um)});
    }

    if (points.empty() || sample_count == 0) {
        return {};
    }
    if (points.size() == 1) {
        return std::vector<Vec3d>(sample_count, points.front());
    }

    // Match Import3d_Section::contourcenter:
    // build cumulative distance over the original contour point sequence as an open polyline
    // (i.e. no explicit final segment from last point back to first).
    const auto n = points.size();
    std::vector<double> perim(n, 0.0);
    for (std::size_t i = 1; i < n; ++i) {
        perim[i] = perim[i - 1] + point_distance(points[i - 1], points[i]);
    }
    const double total = perim.back();
    if (!(total > 1e-12) || !std::isfinite(total)) {
        return std::vector<Vec3d>(sample_count, points.front());
    }

    std::vector<Vec3d> sampled(sample_count);
    std::size_t edge = 0;
    const double step = total / static_cast<double>(sample_count - 1);
    for (std::size_t k = 0; k < sample_count; ++k) {
        double target = step * static_cast<double>(k);
        if (k + 1 == sample_count) {
            target = total;
        }
        while (edge + 1 < perim.size() && perim[edge + 1] < target) {
            ++edge;
        }
        if (edge + 1 >= n) {
            edge = n - 2;
        }
        const std::size_t i0 = edge;
        const std::size_t i1 = edge + 1;
        const double seg0 = perim[edge];
        const double seg1 = perim[edge + 1];
        const double seg = seg1 - seg0;
        const double t = (seg > 1e-12) ? std::clamp((target - seg0) / seg, 0.0, 1.0) : 0.0;
        sampled[k] = Vec3d{
            points[i0].x + (points[i1].x - points[i0].x) * t,
            points[i0].y + (points[i1].y - points[i0].y) * t,
            points[i0].z + (points[i1].z - points[i0].z) * t,
        };
    }
    return sampled;
}

struct EigenSymm3 {
    std::array<double, 3> eval{};
    std::array<std::array<double, 3>, 3> evec{};
};

EigenSymm3 jacobi_eigen_symm3(std::array<std::array<double, 3>, 3> a) {
    std::array<std::array<double, 3>, 3> v = {{
        {{1.0, 0.0, 0.0}},
        {{0.0, 1.0, 0.0}},
        {{0.0, 0.0, 1.0}},
    }};

    for (int iter = 0; iter < 64; ++iter) {
        int p = 0;
        int q = 1;
        double max_off = std::fabs(a[0][1]);
        if (std::fabs(a[0][2]) > max_off) {
            p = 0;
            q = 2;
            max_off = std::fabs(a[0][2]);
        }
        if (std::fabs(a[1][2]) > max_off) {
            p = 1;
            q = 2;
            max_off = std::fabs(a[1][2]);
        }
        if (max_off < 1e-12) {
            break;
        }

        const double app = a[p][p];
        const double aqq = a[q][q];
        const double apq = a[p][q];
        const double theta = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double c = std::cos(theta);
        const double s = std::sin(theta);

        a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        for (int r = 0; r < 3; ++r) {
            if (r == p || r == q) {
                continue;
            }
            const double arp = a[r][p];
            const double arq = a[r][q];
            a[r][p] = c * arp - s * arq;
            a[p][r] = a[r][p];
            a[r][q] = s * arp + c * arq;
            a[q][r] = a[r][q];
        }

        for (int r = 0; r < 3; ++r) {
            const double vrp = v[r][p];
            const double vrq = v[r][q];
            v[r][p] = c * vrp - s * vrq;
            v[r][q] = s * vrp + c * vrq;
        }
    }

    EigenSymm3 out{};
    out.eval = {a[0][0], a[1][1], a[2][2]};
    out.evec = v;
    return out;
}

Vec3d eigen_col(const EigenSymm3& e, int col) {
    return Vec3d{e.evec[0][col], e.evec[1][col], e.evec[2][col]};
}

void keep_strictly_increasing(std::vector<double>& x, std::vector<double>& y) {
    if (x.empty() || y.empty() || x.size() != y.size()) {
        x.clear();
        y.clear();
        return;
    }
    std::vector<double> out_x;
    std::vector<double> out_y;
    out_x.push_back(x.front());
    out_y.push_back(y.front());
    for (std::size_t i = 1; i < x.size(); ++i) {
        if (x[i] > out_x.back()) {
            out_x.push_back(x[i]);
            out_y.push_back(y[i]);
        }
    }
    x.swap(out_x);
    y.swap(out_y);
}

std::vector<double> linspace(double a, double b, std::size_t n) {
    std::vector<double> out(n);
    if (n == 0) {
        return out;
    }
    if (n == 1) {
        out[0] = a;
        return out;
    }
    const double step = (b - a) / static_cast<double>(n - 1);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = a + step * static_cast<double>(i);
    }
    return out;
}

std::vector<double> interpolate_linear(const std::vector<double>& x_new,
                                       const std::vector<double>& x,
                                       const std::vector<double>& y) {
    std::vector<double> out(x_new.size(), 0.0);
    if (x.empty() || y.empty() || x.size() != y.size()) {
        return out;
    }
    if (x.size() == 1) {
        std::fill(out.begin(), out.end(), y.front());
        return out;
    }

    for (std::size_t i = 0; i < x_new.size(); ++i) {
        const double xv = x_new[i];
        if (xv <= x.front()) {
            out[i] = y.front();
            continue;
        }
        if (xv >= x.back()) {
            out[i] = y.back();
            continue;
        }
        const auto it = std::lower_bound(x.begin(), x.end(), xv);
        const std::size_t hi = static_cast<std::size_t>(std::distance(x.begin(), it));
        const std::size_t lo = hi - 1;
        const double x0 = x[lo];
        const double x1 = x[hi];
        const double t = (x1 > x0) ? (xv - x0) / (x1 - x0) : 0.0;
        out[i] = y[lo] + (y[hi] - y[lo]) * t;
    }
    return out;
}

std::vector<Pt3dPoint> contour2centroid_soma_points(const std::vector<AscPoint>& contour_points) {
    if (contour_points.size() < 3) {
        return {};
    }

    const auto sampled = contour_uniform_samples(contour_points, 101);
    if (sampled.size() < 4) {
        return {};
    }

    Vec3d mean{};
    for (const auto& p : sampled) {
        mean.x += p.x;
        mean.y += p.y;
        mean.z += p.z;
    }
    const double inv_n = 1.0 / static_cast<double>(sampled.size());
    mean.x *= inv_n;
    mean.y *= inv_n;
    mean.z *= inv_n;

    std::array<std::array<double, 3>, 3> m = {{
        {{0.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0}},
        {{0.0, 0.0, 0.0}},
    }};

    std::vector<Vec3d> centered;
    for (const auto& p : sampled) {
        const Vec3d q{p.x - mean.x, p.y - mean.y, p.z - mean.z};
        centered.push_back(q);
        m[0][0] += q.x * q.x;
        m[0][1] += q.x * q.y;
        m[0][2] += q.x * q.z;
        m[1][1] += q.y * q.y;
        m[1][2] += q.y * q.z;
        m[2][2] += q.z * q.z;
    }
    m[1][0] = m[0][1];
    m[2][0] = m[0][2];
    m[2][1] = m[1][2];

    const auto eig = jacobi_eigen_symm3(m);
    const int imax = static_cast<int>(
        std::distance(eig.eval.begin(), std::max_element(eig.eval.begin(), eig.eval.end())));
    const int imin = static_cast<int>(
        std::distance(eig.eval.begin(), std::min_element(eig.eval.begin(), eig.eval.end())));
    const int imid = 3 - imax - imin;

    Vec3d major = normalize(eigen_col(eig, imax));
    if (norm(major) < 1e-12) {
        return {};
    }
    const double major_abs[3] = {std::fabs(major.x), std::fabs(major.y), std::fabs(major.z)};
    int major_axis = 0;
    if (major_abs[1] > major_abs[major_axis]) {
        major_axis = 1;
    }
    if (major_abs[2] > major_abs[major_axis]) {
        major_axis = 2;
    }
    const double orient = (major_axis == 0) ? major.x : (major_axis == 1 ? major.y : major.z);
    if (orient < 0.0) {
        major = mul(major, -1.0);
    }

    Vec3d minor = eigen_col(eig, imid);
    minor.z = 0.0;
    const double major_mag = norm(major);
    double minor_mag = norm(minor);
    if (minor_mag / (major_mag + 1e-100) < 1e-6) {
        minor = Vec3d{-major.y, major.x, 0.0};
        minor_mag = norm(minor);
        if (minor_mag < 1e-12) {
            minor = Vec3d{1.0, 0.0, 0.0};
            minor_mag = 1.0;
        }
    }
    minor = mul(minor, 1.0 / minor_mag);

    std::vector<double> d;
    std::vector<double> rad;
    for (const auto& q : centered) {
        d.push_back(dot(q, major));
        rad.push_back(dot(q, minor));
    }

    const auto it_max = std::max_element(d.begin(), d.end());
    const std::size_t imax_d = static_cast<std::size_t>(std::distance(d.begin(), it_max));
    std::rotate(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(imax_d), d.end());
    std::rotate(rad.begin(), rad.begin() + static_cast<std::ptrdiff_t>(imax_d), rad.end());

    const auto it_min = std::min_element(d.begin(), d.end());
    const std::size_t imin_d = static_cast<std::size_t>(std::distance(d.begin(), it_min));
    if (imin_d == 0 || imin_d + 1 >= d.size()) {
        return {};
    }

    std::vector<double> side2(d.begin() + static_cast<std::ptrdiff_t>(imin_d), d.end());
    std::vector<double> rad2(rad.begin() + static_cast<std::ptrdiff_t>(imin_d), rad.end());
    d = std::vector<double>(d.begin(), d.begin() + static_cast<std::ptrdiff_t>(imin_d));
    rad = std::vector<double>(rad.begin(), rad.begin() + static_cast<std::ptrdiff_t>(imin_d));
    std::reverse(d.begin(), d.end());
    std::reverse(rad.begin(), rad.end());

    keep_strictly_increasing(d, rad);
    keep_strictly_increasing(side2, rad2);
    if (d.size() < 2 || side2.size() < 2) {
        return {};
    }

    std::vector<double> merged = d;
    merged.insert(merged.end(), side2.begin(), side2.end());
    std::sort(merged.begin(), merged.end());
    if (merged.size() < 4) {
        return {};
    }

    const double lo = merged[1];
    const double hi = merged[merged.size() - 2];
    if (!(hi > lo) || !std::isfinite(lo) || !std::isfinite(hi)) {
        return {};
    }

    const auto major_pos = linspace(lo, hi, 21);
    auto rad1_i = interpolate_linear(major_pos, d, rad);
    auto rad2_i = interpolate_linear(major_pos, side2, rad2);

    if (rad1_i.size() != major_pos.size() || rad2_i.size() != major_pos.size()) {
        return {};
    }

    std::vector<double> diam(major_pos.size(), 0.0);
    for (std::size_t i = 0; i < major_pos.size(); ++i) {
        const double dd = std::fabs(rad1_i[i] - rad2_i[i]);
        diam[i] = dd;
    }
    if (diam.size() >= 2) {
        diam.front() = 0.5 * (diam[0] + diam[1]);
        diam.back() = 0.5 * (diam[diam.size() - 1] + diam[diam.size() - 2]);
    }

    std::vector<Pt3dPoint> soma_pts;
    for (std::size_t i = 0; i < major_pos.size(); ++i) {
        const Vec3d pt = add(mean, mul(major, major_pos[i]));
        double d_um = diam[i];
        if (!(d_um > 1e-6) || !std::isfinite(d_um)) {
            d_um = 1.0;
        }
        soma_pts.push_back(Pt3dPoint{
            pt.x,
            pt.y,
            pt.z,
            d_um,
        });
    }
    return soma_pts;
}

section_id create_soma_section(Morph& morph, const std::vector<AscPoint>& contour_points) {
    if (contour_points.empty()) {
        return invalid_section_id;
    }

    auto soma_pts = contour2centroid_soma_points(contour_points);
    if (soma_pts.size() < 2) {
        return invalid_section_id;
    }

    return append_section_with_pt3d(
        morph,
        "soma_0",
        invalid_section_id,
        1.0,
        "soma",
        std::span<const Pt3dPoint>(soma_pts.data(), soma_pts.size()));
}

std::optional<AscPoint> first_point_in_branch(const AscBranch& branch) {
    if (!branch.points.empty()) {
        return branch.points.front();
    }
    for (const auto& child : branch.children) {
        if (auto p = first_point_in_branch(child)) {
            return p;
        }
    }
    return std::nullopt;
}

struct LabelCounters {
    int axon{0};
    int dend{0};
    int apic{0};
};

std::string next_name_for_label(std::string_view label, LabelCounters& counters) {
    if (label == "axon") {
        return "axon_" + std::to_string(counters.axon++);
    }
    if (label == "dend") {
        return "dend_" + std::to_string(counters.dend++);
    }
    if (label == "apic") {
        return "apic_" + std::to_string(counters.apic++);
    }
    return "sec_0";
}

void emit_branch_sections(Morph& morph,
                          const AscBranch& branch,
                          section_id parent_section_id,
                          const std::optional<AscPoint>& parent_terminal,
                          std::string_view label,
                          LabelCounters& counters,
                          double parentx) {
    // Match NEURON Import3d branch semantics:
    // each parsed branch maps to one section (no unary-chain merge).
    const std::vector<AscPoint>* section_points = &branch.points;
    std::vector<AscPoint> owned_section_points;
    std::optional<AscPoint> current_terminal = parent_terminal;
    if (!branch.points.empty()) {
        current_terminal = branch.points.back();
    }

    // Match NEURON Import3d need_extra/newsec behavior:
    // child section prepends parent terminal point when first point differs.
    if (parent_section_id != invalid_section_id && parent_terminal.has_value() && !section_points->empty() &&
        !same_xyz(*parent_terminal, section_points->front())) {
        owned_section_points = *section_points;
        AscPoint duplicated_parent = *parent_terminal;
        const float child_first_diam = owned_section_points.front().diam_um;
        duplicated_parent.diam_um = child_first_diam;
        owned_section_points.insert(owned_section_points.begin(), duplicated_parent);
        section_points = &owned_section_points;
    }

    section_id current_parent = parent_section_id;
    bool created_section = false;

    if (!section_points->empty()) {
        if (section_points->size() == 1) {
            if (section_points != &owned_section_points) {
                owned_section_points = *section_points;
                section_points = &owned_section_points;
            }
            // Keep NEURON-compatible geometry for one-point sections.
            AscPoint p = owned_section_points.front();
            float diam = std::fabs(p.diam_um);
            if (!(static_cast<double>(diam) > 1e-6) || !std::isfinite(static_cast<double>(diam))) {
                diam = 1.0f;
            }
            const float half = 0.5f * diam;
            owned_section_points = {
                AscPoint{p.x_um - half, p.y_um, p.z_um, diam},
                AscPoint{p.x_um, p.y_um, p.z_um, diam},
                AscPoint{p.x_um + half, p.y_um, p.z_um, diam},
            };
            section_points = &owned_section_points;
        }

        std::vector<Pt3dPoint> pt3d;
        for (const auto& p : *section_points) {
            double d = std::fabs(p.diam_um);
            if (!(d > 1e-6) || !std::isfinite(d)) {
                d = 1.0;
            }
            pt3d.push_back(Pt3dPoint{p.x_um, p.y_um, p.z_um, d});
        }

        const std::string sec_name = next_name_for_label(label, counters);
        const auto new_section_id = append_section_with_pt3d(morph,
                                                             sec_name,
                                                             current_parent,
                                                             current_parent == invalid_section_id ? 1.0 : parentx,
                                                             std::string(label),
                                                             std::span<const Pt3dPoint>(pt3d.data(), pt3d.size()));

        current_parent = new_section_id;
        created_section = true;
    }

    const double child_parentx = created_section ? 1.0 : parentx;
    for (const auto& child : branch.children) {
        emit_branch_sections(morph, child, current_parent, current_terminal, label, counters, child_parentx);
    }
}

std::optional<AscPoint> find_first_neurite_point(const std::vector<AscTree>& trees) {
    for (const auto& t : trees) {
        if (t.type != AscTreeType::axon && t.type != AscTreeType::dend && t.type != AscTreeType::apic) {
            continue;
        }
        if (auto p = first_point_in_branch(t.root)) {
            return p;
        }
    }
    return std::nullopt;
}

}  // namespace

Morph load_asc_morphology(const std::string& asc_path) {
    std::ifstream file(asc_path);
    if (!file) {
        throw std::runtime_error("failed to open ASC morphology file: " + asc_path);
    }

    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (text.empty()) {
        return Morph{};
    }

    auto tokens = tokenize_asc(text);
    AscParser parser(std::move(tokens));
    auto trees = parser.parse();

    Morph morph{};

    std::vector<AscPoint> soma_contour;
    std::size_t soma_count = 0;
    for (const auto& tree : trees) {
        if (tree.type != AscTreeType::soma) {
            continue;
        }
        if (tree.root.points.empty()) {
            continue;
        }
        ++soma_count;
        if (soma_count == 1) {
            soma_contour = tree.root.points;
        }
    }

    if (soma_count > 1) {
        throw std::runtime_error("ASC morphology has multiple CellBody contours; only one is supported");
    }

    section_id soma_id = invalid_section_id;
    if (!soma_contour.empty()) {
        soma_id = create_soma_section(morph, soma_contour);
    } else if (const auto p = find_first_neurite_point(trees)) {
        soma_id = create_soma_section(morph, std::vector<AscPoint>{*p});
    }

    LabelCounters counters{};

    for (const auto& tree : trees) {
        std::string_view label;
        switch (tree.type) {
            case AscTreeType::axon:
                label = "axon";
                break;
            case AscTreeType::dend:
                label = "dend";
                break;
            case AscTreeType::apic:
                label = "apic";
                break;
            default:
                continue;
        }

        if (tree.root.points.empty() && tree.root.children.empty()) {
            continue;
        }

        emit_branch_sections(
            morph,
            tree.root,
            soma_id,
            std::nullopt,
            label,
            counters,
            (soma_id == invalid_section_id) ? 1.0 : 0.5);
    }

    return morph;
}

}  // namespace asc_detail

Morph load_asc_morphology(const std::string& asc_path) {
    return asc_detail::load_asc_morphology(asc_path);
}

}  // namespace mind_micro_morph
