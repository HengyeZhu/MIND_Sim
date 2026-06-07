#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mind_micro_biophysical {

enum class MechanismKind : std::uint8_t {
    Density = 0,
    EventTarget = 1,
    ArtificialCell = 2,
};

enum class MechanismFieldRole : std::uint8_t {
    Parameter = 0,
    Assigned = 1,
    State = 2,
    Range = 3,
    Pointer = 4,
};

struct MechanismField {
    std::string name{};
    MechanismFieldRole role{MechanismFieldRole::Parameter};
    int array_size{1};
    double default_value{0.0};
    bool is_global{false};
};

struct MechanismMetadata {
    int id{-1};
    int runtime_type{-1};
    std::string name{};
    MechanismKind kind{MechanismKind::Density};
    bool has_net_receive{false};
    int net_receive_weight_count{1};
    std::vector<MechanismField> fields{};
    std::vector<int> field_data_offsets{};
    std::unordered_map<std::string, int> field_index_by_name{};
};

class MechanismCatalog {
  public:
    MechanismCatalog();

    [[nodiscard]] bool contains(const std::string& name) const noexcept;
    [[nodiscard]] const MechanismMetadata& require(const std::string& name) const;
    [[nodiscard]] const MechanismMetadata& require(int metadata_id) const;
    [[nodiscard]] int metadata_id(const std::string& name) const;
    [[nodiscard]] const std::vector<MechanismMetadata>& mechanisms() const noexcept {
        return mechanisms_;
    }

    int register_mechanism(MechanismMetadata metadata);
    int sync_registered_mechanisms();
    int load_mech_path(const std::filesystem::path& path);

  private:
    std::vector<MechanismMetadata> mechanisms_{};
    std::unordered_map<std::string, int> index_by_name_{};
};

[[nodiscard]] const char* mechanism_kind_name(MechanismKind kind) noexcept;
[[nodiscard]] std::string mechanism_metadata_summary(const MechanismMetadata& metadata);

}  // namespace mind_micro_biophysical
