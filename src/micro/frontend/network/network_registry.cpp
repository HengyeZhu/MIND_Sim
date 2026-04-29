#include "network/network_registry.hpp"

#include <algorithm>
#include <iterator>
#include <limits>
#include <span>
#include <string_view>
#include <stdexcept>
#include <utility>

namespace mind_micro_network {

namespace {

void validate_weight_delay(double weight, double delay) {
    if (!std::isfinite(weight) || !std::isfinite(delay) || delay < 0.0) {
        throw std::runtime_error("connection weight/delay must be finite and delay must be non-negative");
    }
}

void validate_threshold(const std::optional<double>& threshold) {
    if (threshold.has_value() && !std::isfinite(*threshold)) {
        throw std::runtime_error("threshold must be finite");
    }
}

int require_netcon_weight_count(const NetworkRegistry& registry, int target_event_target_id) {
    if (target_event_target_id < 0) {
        return 1;
    }
    const int weight_count = registry.event_target_net_receive_weight_count(target_event_target_id);
    if (weight_count < 1) {
        throw std::runtime_error("NetCon target mechanism '" +
                                 registry.event_target_mech(target_event_target_id) +
                                 "' has no registered NET_RECEIVE weight vector");
    }
    return weight_count;
}

int normalize_weight_index(int array_index, std::size_t count) {
    int resolved = array_index;
    if (resolved < 0) {
        resolved += static_cast<int>(count);
    }
    if (resolved < 0 || static_cast<std::size_t>(resolved) >= count) {
        throw std::out_of_range(
            "NetCon weight index out of range: index=" + std::to_string(array_index) +
            ", count=" + std::to_string(count));
    }
    return resolved;
}

std::size_t total_weight_count(std::span<const int> weight_counts) {
    std::size_t total = 0;
    for (const int count : weight_counts) {
        total += static_cast<std::size_t>(count);
    }
    return total;
}

void validate_flat_netcon_group_payload(const NetworkRegistry& registry,
                                        std::span<const int> target_event_target_ids,
                                        std::span<const double> delays,
                                        std::span<const int> weight_offsets,
                                        std::span<const int> weight_counts,
                                        std::span<const double> weights,
                                        const char* what) {
    const std::size_t count = target_event_target_ids.size();
    if (delays.size() != count || weight_offsets.size() != count || weight_counts.size() != count) {
        throw std::runtime_error(std::string(what) + " size mismatch");
    }
    for (std::size_t i = 0; i < count; ++i) {
        validate_weight_delay(0.0, delays[i]);
        const int weight_offset = weight_offsets[i];
        const int weight_count = weight_counts[i];
        if (weight_offset < 0 || weight_count < 1) {
            throw std::runtime_error(std::string(what) + " weight offset/count must be non-negative");
        }
        const std::size_t begin = static_cast<std::size_t>(weight_offset);
        const std::size_t end = begin + static_cast<std::size_t>(weight_count);
        if (begin > weights.size() || end > weights.size()) {
            throw std::runtime_error(std::string(what) + " weight range is out of bounds");
        }
        for (std::size_t j = begin; j < end; ++j) {
            if (!std::isfinite(weights[j])) {
                throw std::runtime_error(std::string(what) + " weights must be finite");
            }
        }

        const int target_event_target_id = target_event_target_ids[i];
        const int expected_weight_count =
            target_event_target_id >= 0 ? require_netcon_weight_count(registry, target_event_target_id) : 1;
        if (weight_count != expected_weight_count) {
            throw std::runtime_error(
                std::string(what) + " weight_count mismatch at slot=" + std::to_string(i) +
                " expected=" + std::to_string(expected_weight_count) +
                " got=" + std::to_string(weight_count));
        }
    }
}


}  // namespace

int NetworkRegistry::intern_event_target_mechanism_group_(EventTargetKind kind,
                                                          const std::string& mech,
                                                          int net_receive_weight_count) {
    if (net_receive_weight_count < 0) {
        throw std::runtime_error("NET_RECEIVE weight count must be non-negative for mechanism '" + mech + "'");
    }
    auto* cached_mech = &cached_located_event_target_mech_;
    auto* cached_index = &cached_located_event_target_group_index_;
    auto& index_by_name = kind == EventTargetKind::ArtificialCell
                              ? artificial_event_target_mechanism_group_index_by_name_
                              : located_event_target_mechanism_group_index_by_name_;
    if (kind == EventTargetKind::ArtificialCell) {
        cached_mech = &cached_artificial_event_target_mech_;
        cached_index = &cached_artificial_event_target_group_index_;
    }
    if (*cached_index >= 0 && *cached_mech == mech) {
        const auto& group = event_target_mechanism_groups_[static_cast<std::size_t>(*cached_index)];
        if (group.net_receive_weight_count != net_receive_weight_count) {
            throw std::runtime_error("inconsistent NET_RECEIVE weight count for mechanism '" + mech + "'");
        }
        return *cached_index;
    }
    const auto it = index_by_name.find(mech);
    if (it != index_by_name.end()) {
        const auto& group = event_target_mechanism_groups_[static_cast<std::size_t>(it->second)];
        if (group.net_receive_weight_count != net_receive_weight_count) {
            throw std::runtime_error("inconsistent NET_RECEIVE weight count for mechanism '" + mech + "'");
        }
        *cached_mech = mech;
        *cached_index = it->second;
        return it->second;
    }
    const int mech_group_index = static_cast<int>(event_target_mechanism_groups_.size());
    EventTargetMechanismGroup group{};
    group.kind = kind;
    group.mech = mech;
    group.net_receive_weight_count = net_receive_weight_count;
    event_target_mechanism_groups_.push_back(std::move(group));
    index_by_name.emplace(mech, mech_group_index);
    *cached_mech = mech;
    *cached_index = mech_group_index;
    return mech_group_index;
}

void NetworkRegistry::clear() {
    *this = NetworkRegistry{};
}

int NetworkRegistry::clear_registered_transfers() {
    transfer_sources_.clear();
    transfer_source_index_by_sid_.clear();
    transfer_targets_.clear();
    next_transfer_sid_counter_ = 0;
    return 0;
}

int NetworkRegistry::register_event_target(int gid,
                                            int section_index,
                                            double loc,
                                            const std::string& mech,
                                            int net_receive_weight_count) {
    if (section_index < 0) {
        throw std::runtime_error("event-target section index must be non-negative");
    }
    if (mech.empty()) {
        throw std::runtime_error("event-target mechanism name is empty");
    }
    if (!std::isfinite(loc) || loc < 0.0 || loc > 1.0) {
        throw std::runtime_error("event-target location must be finite and in [0, 1]");
    }

    const int mech_group_index =
        intern_event_target_mechanism_group_(EventTargetKind::EventTarget, mech, net_receive_weight_count);
    auto& group = event_target_mechanism_groups_[static_cast<std::size_t>(mech_group_index)];
    const int id = static_cast<int>(event_targets_.size());
    EventTargetRegistration registration{};
    registration.id = id;
    registration.mech_group_index = mech_group_index;
    registration.group_slot = static_cast<int>(group.event_target_ids.size());
    event_targets_.push_back(std::move(registration));
    group.event_target_ids.push_back(id);
    group.gids.push_back(gid);
    group.section_indices.push_back(section_index);
    group.locs.push_back(loc);
    const int target_slot = static_cast<int>(event_target_slots_.size());
    event_target_slots_.push_back(EventTargetSlot{.event_target_id = id});
    event_target_slot_by_event_target_id_.push_back(target_slot);
    return id;
}

int NetworkRegistry::register_artificial_cell(const std::string& mech, int net_receive_weight_count) {
    const int mech_group_index =
        intern_event_target_mechanism_group_(EventTargetKind::ArtificialCell, mech, net_receive_weight_count);
    auto& group = event_target_mechanism_groups_[static_cast<std::size_t>(mech_group_index)];
    const int id = static_cast<int>(event_targets_.size());
    EventTargetRegistration registration{};
    registration.id = id;
    registration.mech_group_index = mech_group_index;
    registration.group_slot = static_cast<int>(group.event_target_ids.size());
    event_targets_.push_back(std::move(registration));
    group.event_target_ids.push_back(id);
    const int target_slot = static_cast<int>(event_target_slots_.size());
    event_target_slots_.push_back(EventTargetSlot{.event_target_id = id});
    event_target_slot_by_event_target_id_.push_back(target_slot);
    return id;
}

const EventTargetRegistration& NetworkRegistry::event_target(int id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= event_targets_.size()) {
        throw std::runtime_error("unknown event-target id=" + std::to_string(id));
    }
    return event_targets_[static_cast<std::size_t>(id)];
}

EventTargetRegistration& NetworkRegistry::event_target(int id) {
    if (id < 0 || static_cast<std::size_t>(id) >= event_targets_.size()) {
        throw std::runtime_error("unknown event-target id=" + std::to_string(id));
    }
    return event_targets_[static_cast<std::size_t>(id)];
}

EventTargetKind NetworkRegistry::event_target_kind(int id) const {
    return event_target_group_(id).kind;
}

int NetworkRegistry::event_target_gid(int id) const {
    const auto& registration = event_target(id);
    const auto& group = event_target_group_(id);
    if (group.kind == EventTargetKind::ArtificialCell) {
        return -1;
    }
    return group.gids[static_cast<std::size_t>(registration.group_slot)];
}

int NetworkRegistry::event_target_section_index(int id) const {
    const auto& registration = event_target(id);
    const auto& group = event_target_group_(id);
    if (group.kind == EventTargetKind::ArtificialCell) {
        return -1;
    }
    return group.section_indices[static_cast<std::size_t>(registration.group_slot)];
}

double NetworkRegistry::event_target_loc(int id) const {
    const auto& registration = event_target(id);
    const auto& group = event_target_group_(id);
    if (group.kind == EventTargetKind::ArtificialCell) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return group.locs[static_cast<std::size_t>(registration.group_slot)];
}

const EventTargetMechanismGroup& NetworkRegistry::event_target_group_(int event_target_id) const {
    const auto& registration = event_target(event_target_id);
    return event_target_mechanism_groups_[static_cast<std::size_t>(registration.mech_group_index)];
}

EventTargetMechanismGroup& NetworkRegistry::event_target_group_(int event_target_id) {
    auto& registration = event_target(event_target_id);
    return event_target_mechanism_groups_[static_cast<std::size_t>(registration.mech_group_index)];
}

const std::string& NetworkRegistry::event_target_mech(int id) const {
    return event_target_group_(id).mech;
}

int NetworkRegistry::event_target_net_receive_weight_count(int id) const {
    return event_target_group_(id).net_receive_weight_count;
}

int NetworkRegistry::intern_real_event_source_slot_(const RealNetConSourceKey& source_key,
                                                    std::optional<double> threshold,
                                                    bool strict_threshold_conflict) {
    validate_threshold(threshold);
    auto apply_threshold = [&](EventSourceSlot& slot, const std::string& source_name) {
        if (!threshold.has_value()) {
            return;
        }
        if (strict_threshold_conflict && slot.threshold != *threshold) {
            throw std::runtime_error(
                "conflicting threshold for " + source_name + " source gid=" + std::to_string(source_key.gid));
        }
        slot.threshold = *threshold;
    };

    if (has_cached_event_real_source_slot_ && cached_event_real_source_key_ == source_key &&
        cached_event_real_source_slot_index_ >= 0) {
        auto& slot = event_source_slots_[static_cast<std::size_t>(cached_event_real_source_slot_index_)];
        apply_threshold(slot, "real");
        return cached_event_real_source_slot_index_;
    }
    const auto it = event_real_source_slot_index_by_key_.find(source_key);
    if (it != event_real_source_slot_index_by_key_.end()) {
        auto& slot = event_source_slots_[static_cast<std::size_t>(it->second)];
        apply_threshold(slot, "real");
        has_cached_event_real_source_slot_ = true;
        cached_event_real_source_key_ = source_key;
        cached_event_real_source_slot_index_ = it->second;
        return it->second;
    }

    const int source_slot = static_cast<int>(event_source_slots_.size());
    EventSourceSlot slot{};
    slot.source_kind = NetConSourceKind::RealCell;
    slot.real_source = source_key;
    slot.threshold = threshold.value_or(10.0);
    event_source_slots_.push_back(std::move(slot));
    event_real_source_slot_index_by_key_.emplace(source_key, source_slot);
    has_cached_event_real_source_slot_ = true;
    cached_event_real_source_key_ = source_key;
    cached_event_real_source_slot_index_ = source_slot;
    return source_slot;
}

int NetworkRegistry::intern_event_target_event_source_slot_(int source_event_target_id,
                                                             std::optional<double> threshold,
                                                             bool strict_threshold_conflict) {
    validate_threshold(threshold);
    auto apply_threshold = [&](EventSourceSlot& slot) {
        if (!threshold.has_value()) {
            return;
        }
        if (strict_threshold_conflict && slot.threshold != *threshold) {
            throw std::runtime_error(
                "conflicting threshold for event-target source id=" +
                std::to_string(source_event_target_id));
        }
        slot.threshold = *threshold;
    };

    if (cached_event_event_target_source_id_ == source_event_target_id &&
        cached_event_event_target_source_slot_index_ >= 0) {
        auto& slot =
            event_source_slots_[static_cast<std::size_t>(cached_event_event_target_source_slot_index_)];
        apply_threshold(slot);
        return cached_event_event_target_source_slot_index_;
    }
    const auto it = event_event_target_source_slot_index_by_id_.find(source_event_target_id);
    if (it != event_event_target_source_slot_index_by_id_.end()) {
        auto& slot = event_source_slots_[static_cast<std::size_t>(it->second)];
        apply_threshold(slot);
        cached_event_event_target_source_id_ = source_event_target_id;
        cached_event_event_target_source_slot_index_ = it->second;
        return it->second;
    }

    const int source_slot = static_cast<int>(event_source_slots_.size());
    EventSourceSlot slot{};
    slot.source_kind = NetConSourceKind::EventTarget;
    slot.source_event_target_id = source_event_target_id;
    slot.threshold = threshold.value_or(10.0);
    event_source_slots_.push_back(std::move(slot));
    event_event_target_source_slot_index_by_id_.emplace(source_event_target_id, source_slot);
    cached_event_event_target_source_id_ = source_event_target_id;
    cached_event_event_target_source_slot_index_ = source_slot;
    return source_slot;
}

int NetworkRegistry::intern_spike_input_source_slot_(int spike_input_id) {
    const int source_slot = static_cast<int>(event_source_slots_.size());
    EventSourceSlot slot{};
    slot.source_kind = NetConSourceKind::SpikeInput;
    slot.spike_input_id = spike_input_id;
    event_source_slots_.push_back(slot);
    return source_slot;
}

int NetworkRegistry::append_existing_event_target_target_(int event_target_id) {
    if (event_target_id < 0) {
        return -1;
    }
    (void)event_target(event_target_id);
    const auto target_index = static_cast<std::size_t>(event_target_id);
    if (target_index < event_target_slot_by_event_target_id_.size()) {
        const int existing = event_target_slot_by_event_target_id_[target_index];
        if (existing >= 0) {
            return existing;
        }
    }
    const int target_slot = static_cast<int>(event_target_slots_.size());
    EventTargetSlot target{};
    target.event_target_id = event_target_id;
    event_target_slots_.push_back(target);
    if (event_target_slot_by_event_target_id_.size() <= target_index) {
        event_target_slot_by_event_target_id_.resize(target_index + 1, -1);
    }
    event_target_slot_by_event_target_id_[target_index] = target_slot;
    return target_slot;
}

int NetworkRegistry::append_event_edge_single_weight_(int source_slot,
                                                      int target_slot,
                                                      double delay,
                                                      double weight,
                                                      int weight_count) {
    const int edge_index = static_cast<int>(event_edges_.size());
    const int weight_offset = static_cast<int>(event_edge_weights_.size());
    event_edges_.push_back(EventEdge{
        .source_slot = source_slot,
        .target_slot = target_slot,
        .delay = delay,
        .weight_offset = weight_offset,
        .weight_count = weight_count,
    });
    event_edge_weights_.resize(event_edge_weights_.size() + static_cast<std::size_t>(weight_count), 0.0);
    event_edge_weights_[static_cast<std::size_t>(weight_offset)] = weight;
    ++event_source_slots_[static_cast<std::size_t>(source_slot)].fanout_count;
    return edge_index;
}

int NetworkRegistry::register_real_netcon(int source_gid,
                                          int source_section_index,
                                          double source_loc,
                                          int target_event_target_id,
                                          double weight,
                                          double delay,
                                          std::optional<double> threshold) {
    if (!std::isfinite(source_loc) || source_loc < 0.0 || source_loc > 1.0) {
        throw std::runtime_error("pre location must be finite and in [0, 1]");
    }
    if (source_section_index < 0) {
        throw std::runtime_error("pre section index must be non-negative");
    }
    validate_weight_delay(weight, delay);
    validate_threshold(threshold);

    const RealNetConSourceKey source_key{
        .gid = source_gid,
        .section_index = source_section_index,
        .loc = source_loc,
    };
    const int source_slot = intern_real_event_source_slot_(source_key, threshold, false);
    const int target_slot = append_existing_event_target_target_(target_event_target_id);
    const int weight_count =
        target_event_target_id >= 0 ? require_netcon_weight_count(*this, target_event_target_id) : 1;
    const int edge_index =
        append_event_edge_single_weight_(source_slot, target_slot, delay, weight, weight_count);

    NetConRegistration registration{};
    registration.id = static_cast<int>(netcons_.size());
    registration.edge_index = edge_index;
    netcons_.push_back(registration);
    return registration.id;
}

std::vector<int> NetworkRegistry::register_real_netcon_source_group(
    int source_gid,
    int source_section_index,
    double source_loc,
    std::vector<int> target_event_target_ids,
    std::vector<double> delays,
    std::vector<int> weight_offsets,
    std::vector<int> weight_counts,
    std::vector<double> weights,
    double threshold) {
    std::vector<RealNetConSourceGroupRegistration> groups{};
    groups.push_back(RealNetConSourceGroupRegistration{
        .source = RealNetConSourceKey{
            .gid = source_gid,
            .section_index = source_section_index,
            .loc = source_loc,
        },
        .threshold = threshold,
        .target_event_target_ids = std::move(target_event_target_ids),
        .delays = std::move(delays),
        .weight_offsets = std::move(weight_offsets),
        .weight_counts = std::move(weight_counts),
        .weights = std::move(weights),
    });
    auto out = register_real_netcon_source_groups(std::move(groups));
    return out.empty() ? std::vector<int>{} : std::move(out.front());
}

std::vector<std::vector<int>> NetworkRegistry::register_real_netcon_source_groups(
    std::vector<RealNetConSourceGroupRegistration> groups) {
    std::vector<std::vector<int>> out(groups.size());
    std::vector<int> source_slots(groups.size(), -1);
    std::vector<std::vector<int>> target_slots(groups.size());
    std::size_t total_edges = 0;
    std::size_t total_weights = 0;

    for (std::size_t group_index = 0; group_index < groups.size(); ++group_index) {
        auto& group = groups[group_index];
        if (!std::isfinite(group.source.loc) || group.source.loc < 0.0 || group.source.loc > 1.0) {
            throw std::runtime_error("pre location must be finite and in [0, 1]");
        }
        if (group.source.section_index < 0) {
            throw std::runtime_error("pre section index must be non-negative");
        }
        if (!std::isfinite(group.threshold)) {
            throw std::runtime_error("threshold must be finite");
        }

        validate_flat_netcon_group_payload(
            *this,
            group.target_event_target_ids,
            group.delays,
            group.weight_offsets,
            group.weight_counts,
            group.weights,
            "real NetCon source group");

        source_slots[group_index] = intern_real_event_source_slot_(group.source, group.threshold, false);
        target_slots[group_index].resize(group.target_event_target_ids.size());
        for (std::size_t i = 0; i < group.target_event_target_ids.size(); ++i) {
            target_slots[group_index][i] = append_existing_event_target_target_(group.target_event_target_ids[i]);
        }

        total_edges += group.target_event_target_ids.size();
        total_weights += total_weight_count(group.weight_counts);
        out[group_index].resize(group.target_event_target_ids.size());
    }

    const std::size_t old_edge_count = event_edges_.size();
    const std::size_t old_weight_count = event_edge_weights_.size();
    const std::size_t old_netcon_count = netcons_.size();
    event_edges_.resize(old_edge_count + total_edges);
    event_edge_weights_.resize(old_weight_count + total_weights);
    netcons_.resize(old_netcon_count + total_edges);

    std::size_t edge_cursor = old_edge_count;
    std::size_t weight_cursor = old_weight_count;
    std::size_t netcon_cursor = old_netcon_count;
    for (std::size_t group_index = 0; group_index < groups.size(); ++group_index) {
        auto& group = groups[group_index];
        const int source_slot = source_slots[group_index];
        auto& group_out = out[group_index];
        for (std::size_t i = 0; i < group.target_event_target_ids.size(); ++i) {
            const int weight_offset = group.weight_offsets[i];
            const int weight_count = group.weight_counts[i];
            const std::size_t begin = static_cast<std::size_t>(weight_offset);
            const std::size_t edge_index = edge_cursor++;
            const std::size_t netcon_index = netcon_cursor++;
            const std::size_t weight_index = weight_cursor;

            event_edges_[edge_index] = EventEdge{
                .source_slot = source_slot,
                .target_slot = target_slots[group_index][i],
                .delay = group.delays[i],
                .weight_offset = static_cast<int>(weight_index),
                .weight_count = weight_count,
            };
            std::copy_n(
                group.weights.data() + begin,
                static_cast<std::size_t>(weight_count),
                event_edge_weights_.begin() + static_cast<std::ptrdiff_t>(weight_index));
            weight_cursor += static_cast<std::size_t>(weight_count);

            netcons_[netcon_index] = NetConRegistration{
                .id = static_cast<int>(netcon_index),
                .edge_index = static_cast<int>(edge_index),
            };
            group_out[i] = static_cast<int>(netcon_index);
            ++event_source_slots_[static_cast<std::size_t>(source_slot)].fanout_count;
        }
    }
    return out;
}

int NetworkRegistry::register_event_target_netcon(int source_event_target_id,
                                                   int target_event_target_id,
                                                   double weight,
                                                   double delay,
                                                   std::optional<double> threshold) {
    if (source_event_target_id < 0) {
        throw std::runtime_error("pre event-target id must be non-negative");
    }
    validate_weight_delay(weight, delay);
    validate_threshold(threshold);

    (void)event_target(source_event_target_id);
    const int source_slot =
        intern_event_target_event_source_slot_(source_event_target_id, threshold, false);
    const int target_slot = append_existing_event_target_target_(target_event_target_id);
    const int weight_count =
        target_event_target_id >= 0 ? require_netcon_weight_count(*this, target_event_target_id) : 1;
    const int edge_index =
        append_event_edge_single_weight_(source_slot, target_slot, delay, weight, weight_count);

    NetConRegistration registration{};
    registration.id = static_cast<int>(netcons_.size());
    registration.edge_index = edge_index;
    netcons_.push_back(registration);
    return registration.id;
}

std::vector<int> NetworkRegistry::register_event_target_netcon_source_group(
    int source_event_target_id,
    std::vector<int> target_event_target_ids,
    std::vector<double> delays,
    std::vector<int> weight_offsets,
    std::vector<int> weight_counts,
    std::vector<double> weights,
    double threshold) {
    std::vector<EventTargetNetConSourceGroupRegistration> groups{};
    groups.push_back(EventTargetNetConSourceGroupRegistration{
        .source_event_target_id = source_event_target_id,
        .threshold = threshold,
        .target_event_target_ids = std::move(target_event_target_ids),
        .delays = std::move(delays),
        .weight_offsets = std::move(weight_offsets),
        .weight_counts = std::move(weight_counts),
        .weights = std::move(weights),
    });
    auto out = register_event_target_netcon_source_groups(std::move(groups));
    return out.empty() ? std::vector<int>{} : std::move(out.front());
}

std::vector<std::vector<int>> NetworkRegistry::register_event_target_netcon_source_groups(
    std::vector<EventTargetNetConSourceGroupRegistration> groups) {
    std::vector<std::vector<int>> out(groups.size());
    std::vector<int> source_slots(groups.size(), -1);
    std::vector<std::vector<int>> target_slots(groups.size());
    std::size_t total_edges = 0;
    std::size_t total_weights = 0;

    for (std::size_t group_index = 0; group_index < groups.size(); ++group_index) {
        auto& group = groups[group_index];
        if (group.source_event_target_id < 0) {
            throw std::runtime_error("pre event-target id must be non-negative");
        }
        if (!std::isfinite(group.threshold)) {
            throw std::runtime_error("threshold must be finite");
        }

        (void)event_target(group.source_event_target_id);
        validate_flat_netcon_group_payload(
            *this,
            group.target_event_target_ids,
            group.delays,
            group.weight_offsets,
            group.weight_counts,
            group.weights,
            "event-target NetCon source group");

        source_slots[group_index] =
            intern_event_target_event_source_slot_(group.source_event_target_id, group.threshold, false);
        target_slots[group_index].resize(group.target_event_target_ids.size());
        for (std::size_t i = 0; i < group.target_event_target_ids.size(); ++i) {
            target_slots[group_index][i] = append_existing_event_target_target_(group.target_event_target_ids[i]);
        }

        total_edges += group.target_event_target_ids.size();
        total_weights += total_weight_count(group.weight_counts);
        out[group_index].resize(group.target_event_target_ids.size());
    }

    const std::size_t old_edge_count = event_edges_.size();
    const std::size_t old_weight_count = event_edge_weights_.size();
    const std::size_t old_netcon_count = netcons_.size();
    event_edges_.resize(old_edge_count + total_edges);
    event_edge_weights_.resize(old_weight_count + total_weights);
    netcons_.resize(old_netcon_count + total_edges);

    std::size_t edge_cursor = old_edge_count;
    std::size_t weight_cursor = old_weight_count;
    std::size_t netcon_cursor = old_netcon_count;
    for (std::size_t group_index = 0; group_index < groups.size(); ++group_index) {
        auto& group = groups[group_index];
        const int source_slot = source_slots[group_index];
        auto& group_out = out[group_index];
        for (std::size_t i = 0; i < group.target_event_target_ids.size(); ++i) {
            const int weight_offset = group.weight_offsets[i];
            const int weight_count = group.weight_counts[i];
            const std::size_t begin = static_cast<std::size_t>(weight_offset);
            const std::size_t edge_index = edge_cursor++;
            const std::size_t netcon_index = netcon_cursor++;
            const std::size_t weight_index = weight_cursor;

            event_edges_[edge_index] = EventEdge{
                .source_slot = source_slot,
                .target_slot = target_slots[group_index][i],
                .delay = group.delays[i],
                .weight_offset = static_cast<int>(weight_index),
                .weight_count = weight_count,
            };
            std::copy_n(
                group.weights.data() + begin,
                static_cast<std::size_t>(weight_count),
                event_edge_weights_.begin() + static_cast<std::ptrdiff_t>(weight_index));
            weight_cursor += static_cast<std::size_t>(weight_count);

            netcons_[netcon_index] = NetConRegistration{
                .id = static_cast<int>(netcon_index),
                .edge_index = static_cast<int>(edge_index),
            };
            group_out[i] = static_cast<int>(netcon_index);
            ++event_source_slots_[static_cast<std::size_t>(source_slot)].fanout_count;
        }
    }
    return out;
}

int NetworkRegistry::register_spike_input_source() {
    const int id = static_cast<int>(spike_inputs_.size());
    const int source_slot = intern_spike_input_source_slot_(id);
    spike_inputs_.push_back(SpikeInputRegistration{
        .id = id,
        .source_slot = source_slot,
        .runtime_index = -1,
    });
    return id;
}

const SpikeInputRegistration& NetworkRegistry::spike_input(int id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= spike_inputs_.size()) {
        throw std::runtime_error("unknown spike input id=" + std::to_string(id));
    }
    return spike_inputs_[static_cast<std::size_t>(id)];
}

SpikeInputRegistration& NetworkRegistry::spike_input(int id) {
    if (id < 0 || static_cast<std::size_t>(id) >= spike_inputs_.size()) {
        throw std::runtime_error("unknown spike input id=" + std::to_string(id));
    }
    return spike_inputs_[static_cast<std::size_t>(id)];
}

int NetworkRegistry::register_spike_input_netcon(int spike_input_id,
                                                 int target_event_target_id,
                                                 double weight,
                                                 double delay) {
    validate_weight_delay(weight, delay);
    const int source_slot = spike_input(spike_input_id).source_slot;
    const int target_slot = append_existing_event_target_target_(target_event_target_id);
    const int weight_count =
        target_event_target_id >= 0 ? require_netcon_weight_count(*this, target_event_target_id) : 1;
    const int edge_index =
        append_event_edge_single_weight_(source_slot, target_slot, delay, weight, weight_count);

    NetConRegistration registration{};
    registration.id = static_cast<int>(netcons_.size());
    registration.edge_index = edge_index;
    netcons_.push_back(registration);
    return registration.id;
}

int NetworkRegistry::spike_input_runtime_index(int id) const {
    return spike_input(id).runtime_index;
}

void NetworkRegistry::set_spike_input_runtime_index(int id, int runtime_index) {
    spike_input(id).runtime_index = runtime_index;
}

const NetConRegistration& NetworkRegistry::netcon(int connection_id) const {
    if (connection_id < 0 || static_cast<std::size_t>(connection_id) >= netcons_.size()) {
        throw std::runtime_error("unknown connection id=" + std::to_string(connection_id));
    }
    return netcons_[static_cast<std::size_t>(connection_id)];
}

NetConRegistration& NetworkRegistry::netcon(int connection_id) {
    if (connection_id < 0 || static_cast<std::size_t>(connection_id) >= netcons_.size()) {
        throw std::runtime_error("unknown connection id=" + std::to_string(connection_id));
    }
    return netcons_[static_cast<std::size_t>(connection_id)];
}

int NetworkRegistry::get_netcon_runtime_index(int connection_id) const {
    return netcon(connection_id).edge_index;
}

const EventEdge& NetworkRegistry::event_edge_for_connection_(int connection_id) const {
    const auto& registration = netcon(connection_id);
    return event_edges_[static_cast<std::size_t>(registration.edge_index)];
}

EventEdge& NetworkRegistry::event_edge_for_connection_(int connection_id) {
    auto& registration = netcon(connection_id);
    return event_edges_[static_cast<std::size_t>(registration.edge_index)];
}

const EventSourceSlot& NetworkRegistry::event_source_slot_for_connection_(int connection_id) const {
    const auto& edge = event_edge_for_connection_(connection_id);
    return event_source_slots_[static_cast<std::size_t>(edge.source_slot)];
}

EventSourceSlot& NetworkRegistry::event_source_slot_for_connection_(int connection_id) {
    auto& edge = event_edge_for_connection_(connection_id);
    return event_source_slots_[static_cast<std::size_t>(edge.source_slot)];
}

std::size_t NetworkRegistry::netcon_weight_count(int connection_id) const {
    return static_cast<std::size_t>(event_edge_for_connection_(connection_id).weight_count);
}

double NetworkRegistry::get_netcon_weight(int connection_id, int array_index) const {
    const auto& edge = event_edge_for_connection_(connection_id);
    const int resolved_index = normalize_weight_index(array_index, static_cast<std::size_t>(edge.weight_count));
    return event_edge_weights_[static_cast<std::size_t>(edge.weight_offset + resolved_index)];
}

void NetworkRegistry::set_netcon_weight(int connection_id, int array_index, double value) {
    if (!std::isfinite(value)) {
        throw std::runtime_error("connection weight must be finite");
    }
    const auto& edge = event_edge_for_connection_(connection_id);
    const int resolved_index = normalize_weight_index(array_index, static_cast<std::size_t>(edge.weight_count));
    event_edge_weights_[static_cast<std::size_t>(edge.weight_offset + resolved_index)] = value;
}

double NetworkRegistry::get_netcon_threshold(int connection_id) const {
    return event_source_slot_for_connection_(connection_id).threshold;
}

void NetworkRegistry::set_netcon_threshold(int connection_id, double value) {
    if (!std::isfinite(value)) {
        throw std::runtime_error("connection threshold must be finite");
    }
    event_source_slot_for_connection_(connection_id).threshold = value;
}

double NetworkRegistry::get_netcon_delay(int connection_id) const {
    return event_edge_for_connection_(connection_id).delay;
}

void NetworkRegistry::set_netcon_delay(int connection_id, double value) {
    if (!std::isfinite(value) || value < 0.0) {
        throw std::runtime_error("connection delay must be finite and non-negative");
    }
    event_edge_for_connection_(connection_id).delay = value;
}

int NetworkRegistry::get_netcon_target_event_target_id(int connection_id) const {
    const auto& edge = event_edge_for_connection_(connection_id);
    if (edge.target_slot < 0) {
        return -1;
    }
    const auto& target = event_target_slots_[static_cast<std::size_t>(edge.target_slot)];
    return target.event_target_id;
}

int NetworkRegistry::get_netcon_source_event_target_id(int connection_id) const {
    const auto& source = event_source_slot_for_connection_(connection_id);
    return source.source_kind == NetConSourceKind::EventTarget ? source.source_event_target_id : -1;
}

RealNetConSourceKey NetworkRegistry::get_real_netcon_source_key(int connection_id) const {
    const auto& source = event_source_slot_for_connection_(connection_id);
    if (source.source_kind != NetConSourceKind::RealCell) {
        throw std::runtime_error(
            "connection id=" + std::to_string(connection_id) + " is not a real-cell source");
    }
    return source.real_source;
}

int NetworkRegistry::register_gid_source(int gid,
                                         int source_gid,
                                         int source_section_index,
                                         double source_loc,
                                         std::optional<double> threshold) {
    if (gid < 0) {
        throw std::runtime_error("gid source id must be non-negative");
    }
    if (source_section_index < 0) {
        throw std::runtime_error("gid source section index must be non-negative");
    }
    if (!std::isfinite(source_loc) || source_loc < 0.0 || source_loc > 1.0) {
        throw std::runtime_error("gid source loc must be finite and in [0, 1]");
    }
    validate_threshold(threshold);

    const RealNetConSourceKey source_key{
        .gid = source_gid,
        .section_index = source_section_index,
        .loc = source_loc,
    };
    const int source_slot = intern_real_event_source_slot_(source_key, threshold, false);
    const auto gid_index = static_cast<std::size_t>(gid);
    if (gid_source_slot_by_gid_.size() <= gid_index) {
        gid_source_slot_by_gid_.resize(gid_index + 1, -1);
    }
    if (gid_source_slot_by_gid_[gid_index] >= 0) {
        throw std::runtime_error("gid source gid=" + std::to_string(gid) + " is already registered");
    }
    gid_source_slot_by_gid_[gid_index] = source_slot;
    return 0;
}

int NetworkRegistry::register_gid_connect(int gid,
                                          int target_event_target_id,
                                          double weight,
                                          double delay) {
    validate_weight_delay(weight, delay);
    if (gid < 0 || static_cast<std::size_t>(gid) >= gid_source_slot_by_gid_.size() ||
        gid_source_slot_by_gid_[static_cast<std::size_t>(gid)] < 0) {
        throw std::runtime_error("gid source gid=" + std::to_string(gid) + " is not registered");
    }
    const int source_slot = gid_source_slot_by_gid_[static_cast<std::size_t>(gid)];
    const int target_slot = append_existing_event_target_target_(target_event_target_id);
    const int weight_count =
        target_event_target_id >= 0 ? require_netcon_weight_count(*this, target_event_target_id) : 1;
    const int edge_index =
        append_event_edge_single_weight_(source_slot, target_slot, delay, weight, weight_count);

    NetConRegistration registration{};
    registration.id = static_cast<int>(netcons_.size());
    registration.edge_index = edge_index;
    netcons_.push_back(registration);
    return registration.id;
}

int NetworkRegistry::register_transfer_source(TransferEndpointDescriptor endpoint, int sid) {
    if (sid >= 0) {
        if (transfer_source_index_by_sid_.contains(sid)) {
            throw std::runtime_error("source_var sid already exists");
        }
        next_transfer_sid_counter_ = std::max(next_transfer_sid_counter_, sid + 1);
    } else {
        while (transfer_source_index_by_sid_.contains(next_transfer_sid_counter_)) {
            ++next_transfer_sid_counter_;
        }
        sid = next_transfer_sid_counter_++;
    }
    const int source_index = static_cast<int>(transfer_sources_.size());
    transfer_sources_.push_back(TransferSourceDecl{
        .sid = sid,
        .endpoint = std::move(endpoint),
    });
    transfer_source_index_by_sid_[sid] = source_index;
    return sid;
}

int NetworkRegistry::register_transfer_target(int sid, TransferEndpointDescriptor endpoint) {
    if (sid < 0) {
        throw std::runtime_error("target_var: sid must be non-negative");
    }
    next_transfer_sid_counter_ = std::max(next_transfer_sid_counter_, sid + 1);
    transfer_targets_.push_back(TransferTargetDecl{
        .sid = sid,
        .endpoint = std::move(endpoint),
    });
    return 0;
}


}  // namespace mind_micro_network
