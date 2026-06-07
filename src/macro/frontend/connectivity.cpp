#include "macro/frontend/connectivity.hpp"

#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace mind_sim::macro::frontend {

namespace {

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> cells;
    std::string cell;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char ch = line[i];
        if (quoted) {
            if (ch == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cell.push_back('"');
                    ++i;
                } else {
                    quoted = false;
                }
            } else {
                cell.push_back(ch);
            }
        } else if (ch == '"') {
            quoted = true;
        } else if (ch == ',') {
            cells.push_back(trim(std::move(cell)));
            cell.clear();
        } else {
            cell.push_back(ch);
        }
    }
    if (quoted) {
        throw std::runtime_error("connectivity CSV has an unterminated quoted field");
    }
    cells.push_back(trim(std::move(cell)));
    return cells;
}

bool row_has_value(const std::vector<std::string>& row) {
    for (const auto& cell: row) {
        if (!cell.empty()) {
            return true;
        }
    }
    return false;
}

std::vector<std::vector<std::string>> read_csv_rows(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open connectivity CSV: " + path);
    }
    std::vector<std::vector<std::string>> rows;
    std::string line;
    while (std::getline(input, line)) {
        auto row = parse_csv_line(line);
        if (row_has_value(row)) {
            rows.push_back(std::move(row));
        }
    }
    if (rows.empty()) {
        throw std::runtime_error("connectivity CSV is empty: " + path);
    }
    return rows;
}

std::size_t matrix_marker(const std::vector<std::vector<std::string>>& rows,
                          const std::string& name) {
    for (std::size_t index = 0; index < rows.size(); ++index) {
        const auto& row = rows[index];
        if (row.size() >= 2 && row[0] == "matrix" && row[1] == name) {
            return index;
        }
    }
    throw std::runtime_error("connectivity CSV is missing matrix," + name);
}

double parse_double(const std::string& text, const std::string& context) {
    std::size_t consumed = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("connectivity CSV has a non-numeric " + context + ": " + text);
    }
    if (consumed != text.size()) {
        throw std::runtime_error("connectivity CSV has trailing text in " + context + ": " + text);
    }
    if (!std::isfinite(value)) {
        throw std::runtime_error("connectivity CSV value must be finite in " + context);
    }
    return value;
}

struct MatrixSection {
    std::vector<std::string> labels{};
    std::vector<std::vector<double>> values{};
};

MatrixSection parse_matrix_section(const std::vector<std::vector<std::string>>& rows,
                                   std::size_t begin,
                                   std::size_t end,
                                   const std::string& name) {
    if (begin >= end) {
        throw std::runtime_error("connectivity CSV matrix," + name + " is empty");
    }
    const auto& header = rows[begin];
    if (header.size() < 2) {
        throw std::runtime_error("connectivity CSV matrix," + name + " has no ROI labels");
    }
    MatrixSection section;
    section.labels.assign(header.begin() + 1, header.end());
    for (const auto& label: section.labels) {
        if (label.empty()) {
            throw std::runtime_error("connectivity CSV matrix," + name + " has an empty ROI label");
        }
    }

    section.values.reserve(end - begin - 1);
    std::vector<std::string> row_labels;
    row_labels.reserve(end - begin - 1);
    for (std::size_t row_index = begin + 1; row_index < end; ++row_index) {
        const auto& row = rows[row_index];
        if (row.size() != section.labels.size() + 1) {
            throw std::runtime_error(
                "connectivity CSV matrix," + name + " rows must match the ROI count");
        }
        row_labels.push_back(row[0]);
        std::vector<double> values;
        values.reserve(section.labels.size());
        for (std::size_t column = 1; column < row.size(); ++column) {
            values.push_back(parse_double(row[column], name));
        }
        section.values.push_back(std::move(values));
    }
    if (row_labels != section.labels) {
        throw std::runtime_error(
            "connectivity CSV matrix," + name + " row labels must match column labels");
    }
    return section;
}

void validate_connectivity_csv(const std::vector<std::string>& labels,
                               const std::vector<std::vector<double>>& weights,
                               const std::vector<std::vector<double>>& delays) {
    for (std::size_t target = 0; target < labels.size(); ++target) {
        for (std::size_t source = 0; source < labels.size(); ++source) {
            const double weight = weights[target][source];
            const double delay = delays[target][source];
            if (target == source && (weight != 0.0 || delay != 0.0)) {
                throw std::runtime_error(
                    "connectivity CSV diagonal entries must be zero: " +
                    labels[target]);
            }
            if (weight <= 0.0 && delay != 0.0) {
                throw std::runtime_error(
                    "connectivity CSV delays_ms must be zero when weight <= 0: target=" +
                    labels[target] + ", source=" + labels[source]);
            }
        }
    }
}

}  // namespace

Connectivity::Connectivity(std::vector<std::string> labels,
                           std::vector<std::vector<double>> weights,
                           std::vector<std::vector<double>> delays)
    : labels_(std::move(labels)) {
    if (labels_.empty()) {
        throw std::runtime_error("Connectivity requires at least one ROI");
    }
    rois_.reserve(labels_.size());
    for (int index = 0; index < static_cast<int>(labels_.size()); ++index) {
        auto& label = labels_[static_cast<std::size_t>(index)];
        if (label.empty()) {
            throw std::runtime_error("ROI label must be non-empty");
        }
        const auto [_, inserted] = label_to_index_.emplace(label, index);
        if (!inserted) {
            throw std::runtime_error("ROI labels must be unique");
        }
        rois_.push_back(ROI{.index = index, .label = label});
    }
    weights_ = flatten_square_matrix(weights, "weights", false);
    delays_ = flatten_square_matrix(delays, "delays", true);
}

Connectivity Connectivity::from_csv(const std::string& path) {
    auto rows = read_csv_rows(path);
    const auto weights_marker = matrix_marker(rows, "weights");
    const auto delays_marker = matrix_marker(rows, "delays_ms");
    if (delays_marker <= weights_marker) {
        throw std::runtime_error(
            "connectivity CSV matrix,delays_ms must follow matrix,weights");
    }
    auto weights = parse_matrix_section(rows, weights_marker + 1, delays_marker, "weights");
    auto delays = parse_matrix_section(rows, delays_marker + 1, rows.size(), "delays_ms");
    if (delays.labels != weights.labels) {
        throw std::runtime_error("connectivity CSV weights and delays_ms labels must match");
    }
    validate_connectivity_csv(weights.labels, weights.values, delays.values);
    return Connectivity(std::move(weights.labels), std::move(weights.values), std::move(delays.values));
}

int Connectivity::roi_count() const noexcept {
    return static_cast<int>(rois_.size());
}

const std::vector<ROI>& Connectivity::rois() const noexcept {
    return rois_;
}

const std::vector<std::string>& Connectivity::labels() const noexcept {
    return labels_;
}

const std::vector<double>& Connectivity::weights() const noexcept {
    return weights_;
}

const std::vector<double>& Connectivity::delays() const noexcept {
    return delays_;
}

double Connectivity::weight_at(int target_roi, int source_roi) const {
    return weights_[matrix_offset(target_roi, source_roi)];
}

double Connectivity::delay_at(int target_roi, int source_roi) const {
    return delays_[matrix_offset(target_roi, source_roi)];
}

double Connectivity::min_positive_delay() const noexcept {
    double value = std::numeric_limits<double>::infinity();
    for (double delay: delays_) {
        if (delay > 0.0 && delay < value) {
            value = delay;
        }
    }
    return std::isfinite(value) ? value : 0.0;
}

int Connectivity::roi_index(const std::string& label) const {
    const auto iter = label_to_index_.find(label);
    if (iter == label_to_index_.end()) {
        throw std::runtime_error("unknown ROI: " + label);
    }
    return iter->second;
}

std::vector<double> Connectivity::flatten_square_matrix(
    const std::vector<std::vector<double>>& matrix,
    const std::string& name,
    bool require_non_negative) const {
    const auto n = labels_.size();
    if (matrix.size() != n) {
        throw std::runtime_error(name + " row count must match ROI count");
    }
    std::vector<double> out;
    out.reserve(n * n);
    for (const auto& row: matrix) {
        if (row.size() != n) {
            throw std::runtime_error(name + " must be square");
        }
        for (double value: row) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(name + " must contain finite values");
            }
            if (require_non_negative && value < 0.0) {
                throw std::runtime_error(name + " must be non-negative");
            }
            out.push_back(value);
        }
    }
    return out;
}

std::size_t Connectivity::matrix_offset(int target_roi, int source_roi) const {
    const int n = roi_count();
    if (target_roi < 0 || target_roi >= n || source_roi < 0 || source_roi >= n) {
        throw std::runtime_error("connectivity matrix index out of range");
    }
    return static_cast<std::size_t>(target_roi * n + source_roi);
}

}  // namespace mind_sim::macro::frontend
