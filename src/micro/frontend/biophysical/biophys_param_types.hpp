#pragma once

#include <cstddef>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace mind_micro_biophysical {

struct SegmentParamBatch {
    std::vector<int> section_indices{};
    std::vector<std::size_t> value_offsets{};
    std::vector<double> values{};
};

using ParamValue = std::variant<double, SegmentParamBatch>;
using ParamList = std::vector<std::pair<std::string, ParamValue>>;

}  // namespace mind_micro_biophysical
