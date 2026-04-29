#pragma once

#include <cstddef>
#include <vector>

namespace mind_sim::micro::sim {

struct MicroEventTable {
    std::vector<double> time{};
    std::vector<int> index{};

    [[nodiscard]] std::size_t size() const noexcept {
        return time.size();
    }

    void clear() {
        time.clear();
        index.clear();
    }

    void resize(std::size_t count) {
        time.resize(count);
        index.resize(count);
    }
};

struct MicroSpikeTableView {
    const double* time{nullptr};
    const int* gid{nullptr};
    std::size_t count{0};

    [[nodiscard]] std::size_t size() const noexcept {
        return count;
    }
};

struct MicroSpikeTable {
    std::vector<double> time{};
    std::vector<int> gid{};

    [[nodiscard]] std::size_t size() const noexcept {
        return time.size();
    }

    void clear() {
        time.clear();
        gid.clear();
    }

    void resize(std::size_t count) {
        time.resize(count);
        gid.resize(count);
    }

    void append(double spike_time, int spike_gid) {
        time.push_back(spike_time);
        gid.push_back(spike_gid);
    }

    [[nodiscard]] MicroSpikeTableView view(std::size_t offset, std::size_t count) const noexcept {
        if (count == 0) {
            return MicroSpikeTableView{};
        }
        return MicroSpikeTableView{
            .time = time.data() + offset,
            .gid = gid.data() + offset,
            .count = count,
        };
    }

    void append_view(const MicroSpikeTableView& view) {
        if (view.count == 0) {
            return;
        }
        time.insert(time.end(), view.time, view.time + view.count);
        gid.insert(gid.end(), view.gid, view.gid + view.count);
    }
};

}  // namespace mind_sim::micro::sim
