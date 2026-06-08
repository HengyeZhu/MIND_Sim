#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
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

    void sort_by_time_gid() {
        std::vector<std::size_t> order(time.size());
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::stable_sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
            if (time[lhs] == time[rhs]) {
                return gid[lhs] < gid[rhs];
            }
            return time[lhs] < time[rhs];
        });
        std::vector<double> sorted_time;
        std::vector<int> sorted_gid;
        sorted_time.reserve(time.size());
        sorted_gid.reserve(gid.size());
        for (std::size_t index: order) {
            sorted_time.push_back(time[index]);
            sorted_gid.push_back(gid[index]);
        }
        time.swap(sorted_time);
        gid.swap(sorted_gid);
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
