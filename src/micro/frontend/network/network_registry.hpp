#pragma once

#include "biophysical/biophys_param_types.hpp"
#include "morph/cell_location.hpp"

#include <cmath>
#include <cstddef>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace mind_micro_network {

enum class EventTargetKind {
    EventTarget = 0,
    ArtificialCell = 1,
};

struct EventTargetRegistration {
    int id{0};
    int mech_group_index{-1};
    int group_slot{-1};
};

struct EventTargetMechanismGroup {
    EventTargetKind kind{EventTargetKind::EventTarget};
    std::string mech{};
    int net_receive_weight_count{0};
    std::vector<int> event_target_ids{};
    std::vector<int> gids{};
    std::vector<int> section_indices{};
    std::vector<double> locs{};
};

enum class NetConSourceKind {
    RealCell = 0,
    EventTarget = 1,
    SpikeInput = 2,
};

struct NetConRegistration {
    int id{0};
    int edge_index{-1};
};

struct SpikeInputRegistration {
    int id{0};
    int source_slot{-1};
    int runtime_index{-1};
};

struct RealNetConSourceKey {
    int gid{-1};
    int section_index{-1};
    double loc{0.0};

    [[nodiscard]] bool operator==(const RealNetConSourceKey& other) const noexcept {
        return gid == other.gid && section_index == other.section_index && loc == other.loc;
    }

    [[nodiscard]] bool operator<(const RealNetConSourceKey& other) const noexcept {
        if (gid != other.gid) {
            return gid < other.gid;
        }
        if (section_index != other.section_index) {
            return section_index < other.section_index;
        }
        return loc < other.loc;
    }
};

struct RealNetConSourceGroupRegistration {
    RealNetConSourceKey source{};
    double threshold{10.0};
    std::vector<int> target_event_target_ids{};
    std::vector<double> delays{};
    std::vector<int> weight_offsets{};
    std::vector<int> weight_counts{};
    std::vector<double> weights{};
};

struct EventTargetNetConSourceGroupRegistration {
    int source_event_target_id{-1};
    double threshold{10.0};
    std::vector<int> target_event_target_ids{};
    std::vector<double> delays{};
    std::vector<int> weight_offsets{};
    std::vector<int> weight_counts{};
    std::vector<double> weights{};
};

struct EventSourceSlot {
    NetConSourceKind source_kind{NetConSourceKind::RealCell};
    RealNetConSourceKey real_source{};
    int source_event_target_id{-1};
    int spike_input_id{-1};
    double threshold{10.0};
    int fanout_count{0};
};

struct EventTargetSlot {
    int event_target_id{-1};
};

struct EventEdge {
    int source_slot{-1};
    int target_slot{-1};
    double delay{0.0};
    int weight_offset{0};
    int weight_count{0};
};

struct ResolvedTransferEndpoint {
    double* cpu_ptr{nullptr};
    double* gpu_ptr{nullptr};
};

enum class TransferEndpointKind {
    Location = 0,
    LocationVoltage = 1,
    EventTarget = 2,
};

struct TransferEndpointDescriptor {
    TransferEndpointKind kind{TransferEndpointKind::Location};
    int gid{-1};
    int section_index{-1};
    double x{NAN};
    int event_target_id{-1};
    std::string mech{};
    std::string var{};
    int array_index{-1};
};

struct TransferSourceDecl {
    int sid{-1};
    TransferEndpointDescriptor endpoint{};
};

struct TransferTargetDecl {
    int sid{-1};
    TransferEndpointDescriptor endpoint{};
};

class NetworkRegistry final {
public:
    void clear();
    int clear_registered_transfers();
    [[nodiscard]] bool has_registered_transfers() const {
        return !transfer_sources_.empty() || !transfer_targets_.empty();
    }

    [[nodiscard]] int register_event_target(int gid,
                                             int section_index,
                                             double loc,
                                             const std::string& mech,
                                             int net_receive_weight_count);
    [[nodiscard]] int register_artificial_cell(const std::string& mech,
                                               int net_receive_weight_count);
    [[nodiscard]] const EventTargetRegistration& event_target(int id) const;
    [[nodiscard]] EventTargetRegistration& event_target(int id);
    [[nodiscard]] EventTargetKind event_target_kind(int id) const;
    [[nodiscard]] int event_target_gid(int id) const;
    [[nodiscard]] int event_target_section_index(int id) const;
    [[nodiscard]] double event_target_loc(int id) const;
    [[nodiscard]] const std::string& event_target_mech(int id) const;
    [[nodiscard]] int event_target_net_receive_weight_count(int id) const;
    [[nodiscard]] const std::vector<EventTargetRegistration>& event_targets() const {
        return event_targets_;
    }
    [[nodiscard]] const std::vector<EventTargetMechanismGroup>& event_target_mechanism_groups() const {
        return event_target_mechanism_groups_;
    }

    [[nodiscard]] int register_real_netcon(int source_gid,
                                           int source_section_index,
                                           double source_loc,
                                           int target_event_target_id,
                                           double weight,
                                           double delay,
                                           std::optional<double> threshold = std::nullopt);
    [[nodiscard]] std::vector<int> register_real_netcon_source_group(int source_gid,
                                                                     int source_section_index,
                                                                     double source_loc,
                                                                     std::vector<int> target_event_target_ids,
                                                                     std::vector<double> delays,
                                                                     std::vector<int> weight_offsets,
                                                                     std::vector<int> weight_counts,
                                                                     std::vector<double> weights,
                                                                     double threshold);
    [[nodiscard]] std::vector<std::vector<int>> register_real_netcon_source_groups(
        std::vector<RealNetConSourceGroupRegistration> groups);
    [[nodiscard]] int register_event_target_netcon(int source_event_target_id,
                                                    int target_event_target_id,
                                                    double weight,
                                                    double delay,
                                                    std::optional<double> threshold = std::nullopt);
    [[nodiscard]] std::vector<int> register_event_target_netcon_source_group(
        int source_event_target_id,
        std::vector<int> target_event_target_ids,
        std::vector<double> delays,
        std::vector<int> weight_offsets,
        std::vector<int> weight_counts,
        std::vector<double> weights,
        double threshold);
    [[nodiscard]] std::vector<std::vector<int>> register_event_target_netcon_source_groups(
        std::vector<EventTargetNetConSourceGroupRegistration> groups);
    [[nodiscard]] int register_spike_input_source();
    [[nodiscard]] const SpikeInputRegistration& spike_input(int id) const;
    [[nodiscard]] SpikeInputRegistration& spike_input(int id);
    [[nodiscard]] int register_spike_input_netcon(int spike_input_id,
                                                  int target_event_target_id,
                                                  double weight,
                                                  double delay);
    [[nodiscard]] int spike_input_runtime_index(int id) const;
    void set_spike_input_runtime_index(int id, int runtime_index);
    [[nodiscard]] const std::vector<SpikeInputRegistration>& spike_inputs() const {
        return spike_inputs_;
    }
    [[nodiscard]] const NetConRegistration& netcon(int connection_id) const;
    [[nodiscard]] NetConRegistration& netcon(int connection_id);
    [[nodiscard]] int get_netcon_runtime_index(int connection_id) const;
    [[nodiscard]] const std::vector<NetConRegistration>& netcons() const { return netcons_; }
    [[nodiscard]] const std::vector<EventSourceSlot>& event_source_slots() const { return event_source_slots_; }
    [[nodiscard]] const std::vector<EventTargetSlot>& event_target_slots() const { return event_target_slots_; }
    [[nodiscard]] const std::vector<EventEdge>& event_edges() const { return event_edges_; }
    [[nodiscard]] const std::vector<double>& event_edge_weights() const { return event_edge_weights_; }
    [[nodiscard]] std::size_t netcon_weight_count(int connection_id) const;
    [[nodiscard]] double get_netcon_weight(int connection_id, int array_index) const;
    void set_netcon_weight(int connection_id, int array_index, double value);
    [[nodiscard]] double get_netcon_threshold(int connection_id) const;
    void set_netcon_threshold(int connection_id, double value);
    [[nodiscard]] double get_netcon_delay(int connection_id) const;
    void set_netcon_delay(int connection_id, double value);
    [[nodiscard]] int get_netcon_target_event_target_id(int connection_id) const;
    [[nodiscard]] int get_netcon_source_event_target_id(int connection_id) const;
    [[nodiscard]] RealNetConSourceKey get_real_netcon_source_key(int connection_id) const;
    int register_gid_source(int gid,
                            int source_gid,
                            int source_section_index,
                            double source_loc,
                            std::optional<double> threshold = std::nullopt);
    [[nodiscard]] int register_gid_connect(int gid,
                                           int target_event_target_id,
                                           double weight,
                                           double delay);

    [[nodiscard]] int register_transfer_source(TransferEndpointDescriptor endpoint, int sid = -1);
    int register_transfer_target(int sid, TransferEndpointDescriptor endpoint);

private:
    [[nodiscard]] int intern_event_target_mechanism_group_(EventTargetKind kind,
                                                           const std::string& mech,
                                                           int net_receive_weight_count);
    [[nodiscard]] const EventTargetMechanismGroup& event_target_group_(int event_target_id) const;
    [[nodiscard]] EventTargetMechanismGroup& event_target_group_(int event_target_id);
    [[nodiscard]] int intern_real_event_source_slot_(const RealNetConSourceKey& source_key,
                                                     std::optional<double> threshold,
                                                     bool strict_threshold_conflict);
    [[nodiscard]] int intern_event_target_event_source_slot_(int source_event_target_id,
                                                              std::optional<double> threshold,
                                                              bool strict_threshold_conflict);
    [[nodiscard]] int intern_spike_input_source_slot_(int spike_input_id);
    [[nodiscard]] int append_existing_event_target_target_(int event_target_id);
    [[nodiscard]] int append_event_edge_single_weight_(int source_slot,
                                                       int target_slot,
                                                       double delay,
                                                       double weight,
                                                       int weight_count);
    [[nodiscard]] const EventSourceSlot& event_source_slot_for_connection_(int connection_id) const;
    [[nodiscard]] EventSourceSlot& event_source_slot_for_connection_(int connection_id);
    [[nodiscard]] const EventEdge& event_edge_for_connection_(int connection_id) const;
    [[nodiscard]] EventEdge& event_edge_for_connection_(int connection_id);
    std::vector<EventTargetRegistration> event_targets_{};
    std::vector<EventTargetMechanismGroup> event_target_mechanism_groups_{};
    std::unordered_map<std::string, int> located_event_target_mechanism_group_index_by_name_{};
    std::unordered_map<std::string, int> artificial_event_target_mechanism_group_index_by_name_{};
    std::string cached_located_event_target_mech_{};
    int cached_located_event_target_group_index_{-1};
    std::string cached_artificial_event_target_mech_{};
    int cached_artificial_event_target_group_index_{-1};

    std::vector<NetConRegistration> netcons_{};
    std::vector<EventSourceSlot> event_source_slots_{};
    std::map<RealNetConSourceKey, int> event_real_source_slot_index_by_key_{};
    bool has_cached_event_real_source_slot_{false};
    RealNetConSourceKey cached_event_real_source_key_{};
    int cached_event_real_source_slot_index_{-1};
    std::unordered_map<int, int> event_event_target_source_slot_index_by_id_{};
    int cached_event_event_target_source_id_{-1};
    int cached_event_event_target_source_slot_index_{-1};
    std::vector<SpikeInputRegistration> spike_inputs_{};
    std::vector<EventTargetSlot> event_target_slots_{};
    std::vector<int> event_target_slot_by_event_target_id_{};
    std::vector<EventEdge> event_edges_{};
    std::vector<double> event_edge_weights_{};
    std::vector<int> gid_source_slot_by_gid_{};

    std::vector<TransferSourceDecl> transfer_sources_{};
    std::unordered_map<int, int> transfer_source_index_by_sid_{};
    std::vector<TransferTargetDecl> transfer_targets_{};
    int next_transfer_sid_counter_{0};
};

}  // namespace mind_micro_network
