#pragma once

#include "coreneuron/network/netcon.hpp"
#include "coreneuron/permute/data_layout.hpp"
#include "coreneuron/io/mem_layout_util.hpp"
#include "coreneuron/mechanism/mechanism.hpp"
#include "coreneuron/sim/multicore.hpp"
#include "coreneuron/utils/randoms/nrnran123.h"
#include "micro/sim/device.hpp"
#include "micro/sim/mechanism_runtime.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace mind_sim::micro::sim {

struct Random123Deleter {
    void operator()(coreneuron::nrnran123_State* state) const noexcept {
        if (state != nullptr) {
            coreneuron::nrnran123_deletestream(state);
        }
    }
};

struct MechanismRuntimeInfo {
    int type{-1};
    int metadata_id{-1};
    std::string name{};
    bool is_event_target{false};
};

struct CoreInputEventTarget {
    int thread_index{0};
    int input_presyn_index{0};
};

struct CoreNetConRef {
    int thread_index{0};
    int netcon_index{0};
};

struct CoreNetReceiveBufferStorage {
    coreneuron::NetReceiveBuffer_t buffer{};

    CoreNetReceiveBufferStorage() = default;
    CoreNetReceiveBufferStorage(const CoreNetReceiveBufferStorage&) = delete;
    CoreNetReceiveBufferStorage& operator=(const CoreNetReceiveBufferStorage&) = delete;

    CoreNetReceiveBufferStorage(CoreNetReceiveBufferStorage&& other) noexcept
        : buffer(other.buffer) {
        other.buffer = coreneuron::NetReceiveBuffer_t{};
    }

    CoreNetReceiveBufferStorage& operator=(CoreNetReceiveBufferStorage&& other) noexcept {
        if (this != &other) {
            clear();
            buffer = other.buffer;
            other.buffer = coreneuron::NetReceiveBuffer_t{};
        }
        return *this;
    }

    ~CoreNetReceiveBufferStorage() {
        clear();
    }

    [[nodiscard]] bool active() const noexcept {
        return buffer._size > 0;
    }

    void allocate(int capacity, int pnt_offset) {
        clear();
        buffer._size = capacity;
        buffer._pnt_offset = pnt_offset;
        if (capacity <= 0) {
            return;
        }
        buffer._displ = static_cast<int*>(
            coreneuron::ecalloc_align(static_cast<std::size_t>(capacity) + 1, sizeof(int)));
        buffer._nrb_index = static_cast<int*>(
            coreneuron::ecalloc_align(static_cast<std::size_t>(capacity), sizeof(int)));
        buffer._pnt_index = static_cast<int*>(
            coreneuron::ecalloc_align(static_cast<std::size_t>(capacity), sizeof(int)));
        buffer._weight_index = static_cast<int*>(
            coreneuron::ecalloc_align(static_cast<std::size_t>(capacity), sizeof(int)));
        buffer._nrb_t = static_cast<double*>(
            coreneuron::ecalloc_align(static_cast<std::size_t>(capacity), sizeof(double)));
        buffer._nrb_flag = static_cast<double*>(
            coreneuron::ecalloc_align(static_cast<std::size_t>(capacity), sizeof(double)));
    }

    void clear() noexcept {
        ::free_memory(buffer._displ);
        ::free_memory(buffer._nrb_index);
        ::free_memory(buffer._pnt_index);
        ::free_memory(buffer._weight_index);
        ::free_memory(buffer._nrb_t);
        ::free_memory(buffer._nrb_flag);
        buffer = coreneuron::NetReceiveBuffer_t{};
    }
};

struct CoreMembList {
    coreneuron::Memb_list ml{};
    int type{-1};
    int metadata_id{-1};
    std::string name{};
    bool is_event_target{false};
    std::vector<int> nodeindices{};
    std::vector<coreneuron::Datum> pdata{};
    std::vector<coreneuron::ThreadDatum> thread{};
    bool has_net_receive_buffer{false};
    CoreNetReceiveBufferStorage net_receive_buffer{};
    std::unique_ptr<coreneuron::NetSendBuffer_t> net_send_buffer{};
    std::vector<int> permute_storage{};
    std::size_t thread_data_offset{std::numeric_limits<std::size_t>::max()};

    CoreMembList() = default;
    CoreMembList(const CoreMembList&) = delete;
    CoreMembList& operator=(const CoreMembList&) = delete;

    CoreMembList(CoreMembList&& other) noexcept {
        move_from(std::move(other));
    }

    CoreMembList& operator=(CoreMembList&& other) noexcept {
        if (this != &other) {
            move_from(std::move(other));
        }
        return *this;
    }

    void move_from(CoreMembList&& other) noexcept {
        type = other.type;
        metadata_id = other.metadata_id;
        name = std::move(other.name);
        is_event_target = other.is_event_target;
        nodeindices = std::move(other.nodeindices);
        pdata = std::move(other.pdata);
        thread = std::move(other.thread);
        has_net_receive_buffer = other.has_net_receive_buffer;
        net_receive_buffer = std::move(other.net_receive_buffer);
        net_send_buffer = std::move(other.net_send_buffer);
        permute_storage = std::move(other.permute_storage);
        thread_data_offset = other.thread_data_offset;
        ml = coreneuron::Memb_list{};
        bind();
        other.ml = coreneuron::Memb_list{};
        other.has_net_receive_buffer = false;
    }

    void bind(double* thread_data_base = nullptr) noexcept {
        ml.nodeindices = nodeindices.empty() ? nullptr : nodeindices.data();
        ml._permute = permute_storage.empty() ? nullptr : permute_storage.data();
        ml.data = nullptr;
        if (thread_data_base != nullptr &&
            thread_data_offset != std::numeric_limits<std::size_t>::max()) {
            ml.data = thread_data_base + thread_data_offset;
        }
        ml.pdata = pdata.empty() ? nullptr : pdata.data();
        ml._thread = thread.empty() ? nullptr : thread.data();
        ml.nodecount = static_cast<int>(nodeindices.size());
        if (ml._nodecount_padded < ml.nodecount) {
            ml._nodecount_padded = ((ml.nodecount + 7) / 8) * 8;
        }
        ml._net_receive_buffer =
            has_net_receive_buffer && net_receive_buffer.active() ? &net_receive_buffer.buffer
                                                                  : nullptr;
        ml._net_send_buffer = net_send_buffer.get();
    }

    void allocate_net_receive_buffer(int capacity, int pnt_offset = 0) {
        if (capacity < 0) {
            throw std::runtime_error("negative NET_RECEIVE buffer capacity");
        }
        has_net_receive_buffer = true;
        net_receive_buffer.allocate(capacity, pnt_offset);
    }

    void allocate_net_send_buffer(int capacity) {
        if (capacity < 0) {
            throw std::runtime_error("negative NET_SEND buffer capacity");
        }
        net_send_buffer = std::make_unique<coreneuron::NetSendBuffer_t>(std::max(1, capacity));
    }
};

struct CoreNeuronThread: coreneuron::NrnThread {
    std::vector<coreneuron::NrnThreadMembList> tml_storage{};
    std::vector<CoreMembList> memb_lists{};
    std::vector<coreneuron::Memb_list*> ml_ptrs{};
    std::vector<coreneuron::Point_process> pntproc_storage{};
    std::vector<int> pntproc_event_target_ids{};
    std::vector<coreneuron::PreSyn> presyn_storage{};
    std::vector<int> presyn_source_gids{};
    std::vector<coreneuron::PreSynHelper> presyn_helpers{};
    std::vector<coreneuron::InputPreSyn> input_presyns{};
    std::vector<int> input_presyn_source_gids{};
    std::vector<coreneuron::NetCon> netcon_storage{};
    std::vector<int> netcon_source_gids{};
    std::vector<int> netcon_weight_counts{};
    std::vector<double> weight_storage{};
    std::vector<int> net_send_buffer{};
    std::vector<int> pnt2presyn_ix_storage{};
    std::vector<std::size_t> pnt2presyn_ix_offsets{};
    std::vector<int*> pnt2presyn_ix_ptrs{};
    std::vector<coreneuron::NrnThreadBAList> ba_storage{};
    std::vector<int> watch_types_storage{};

    std::vector<int> v_parent_index{};
    std::vector<double> shadow_rhs{};
    std::vector<double> shadow_d{};
    std::vector<double> data_storage{};
    std::size_t node_data_stride{0};
    std::vector<int> node_permutation{};
    std::vector<void*> vdata_storage{};
    std::vector<std::unique_ptr<coreneuron::nrnran123_State, Random123Deleter>> random123_storage{};

    CoreNeuronThread() = default;
    CoreNeuronThread(const CoreNeuronThread&) = delete;
    CoreNeuronThread& operator=(const CoreNeuronThread&) = delete;
    CoreNeuronThread(CoreNeuronThread&&) noexcept = default;
    CoreNeuronThread& operator=(CoreNeuronThread&&) noexcept = default;

    static constexpr std::size_t node_data_slot_count = 7;

    [[nodiscard]] std::size_t mechanism_data_begin() const noexcept {
        return node_data_stride * node_data_slot_count;
    }

    void allocate_node_data(std::size_t node_count) {
        node_data_stride = static_cast<std::size_t>(
            coreneuron::nrn_soa_padded_size(static_cast<int>(node_count), SOA_LAYOUT));
        data_storage.assign(mechanism_data_begin(), 0.0);
    }

    void truncate_mechanism_data() {
        data_storage.resize(mechanism_data_begin());
    }

    [[nodiscard]] std::span<double> actual_rhs() noexcept {
        return {data_storage.data(), node_data_stride};
    }
    [[nodiscard]] std::span<const double> actual_rhs() const noexcept {
        return {data_storage.data(), node_data_stride};
    }
    [[nodiscard]] std::span<double> actual_d() noexcept {
        return {data_storage.data() + node_data_stride, node_data_stride};
    }
    [[nodiscard]] std::span<const double> actual_d() const noexcept {
        return {data_storage.data() + node_data_stride, node_data_stride};
    }
    [[nodiscard]] std::span<double> actual_a() noexcept {
        return {data_storage.data() + 2 * node_data_stride, node_data_stride};
    }
    [[nodiscard]] std::span<const double> actual_a() const noexcept {
        return {data_storage.data() + 2 * node_data_stride, node_data_stride};
    }
    [[nodiscard]] std::span<double> actual_b() noexcept {
        return {data_storage.data() + 3 * node_data_stride, node_data_stride};
    }
    [[nodiscard]] std::span<const double> actual_b() const noexcept {
        return {data_storage.data() + 3 * node_data_stride, node_data_stride};
    }
    [[nodiscard]] std::span<double> actual_v() noexcept {
        return {data_storage.data() + 4 * node_data_stride, node_data_stride};
    }
    [[nodiscard]] std::span<const double> actual_v() const noexcept {
        return {data_storage.data() + 4 * node_data_stride, node_data_stride};
    }
    [[nodiscard]] std::span<double> actual_area() noexcept {
        return {data_storage.data() + 5 * node_data_stride, node_data_stride};
    }
    [[nodiscard]] std::span<const double> actual_area() const noexcept {
        return {data_storage.data() + 5 * node_data_stride, node_data_stride};
    }
    [[nodiscard]] std::span<double> actual_diam() noexcept {
        return {data_storage.data() + 6 * node_data_stride, node_data_stride};
    }
    [[nodiscard]] std::span<const double> actual_diam() const noexcept {
        return {data_storage.data() + 6 * node_data_stride, node_data_stride};
    }

    void bind_into(coreneuron::NrnThread& target, int mechanism_capacity) {
        double* const thread_data_base = data_storage.empty() ? nullptr : data_storage.data();
        for (auto& ml: memb_lists) {
            ml.bind(thread_data_base);
        }

        ml_ptrs.assign(static_cast<std::size_t>(mechanism_capacity), nullptr);
        for (std::size_t i = 0; i < memb_lists.size(); ++i) {
            const int type = memb_lists[i].type;
            ml_ptrs[static_cast<std::size_t>(type)] = &memb_lists[i].ml;
        }

        for (std::size_t i = 0; i < tml_storage.size(); ++i) {
            auto& item = tml_storage[i];
            item.next = (i + 1 < tml_storage.size()) ? &tml_storage[i + 1] : nullptr;
            item.ml = ml_ptrs[static_cast<std::size_t>(item.index)];
        }

        pnt2presyn_ix_ptrs.resize(pnt2presyn_ix_offsets.empty() ? 0 : pnt2presyn_ix_offsets.size() - 1);
        for (std::size_t i = 0; i < pnt2presyn_ix_ptrs.size(); ++i) {
            const auto begin = pnt2presyn_ix_offsets[i];
            const auto end = pnt2presyn_ix_offsets[i + 1];
            pnt2presyn_ix_ptrs[i] = begin == end ? nullptr : pnt2presyn_ix_storage.data() + begin;
        }

        target._t = _t;
        target._dt = _dt;
        target.cj = cj;
        target.ncell = ncell;
        target.end = end;
        target.id = id;
        target._stop_stepping = _stop_stepping;
        target.n_vecplay = n_vecplay;
        target._nidata = _nidata;
        target._idata = _idata;
        target._vecplay = _vecplay;
        target._sp13mat = _sp13mat;
        target._ecell_memb_list = _ecell_memb_list;
        target._ctime = _ctime;
        for (int i = 0; i < BEFORE_AFTER_SIZE; ++i) {
            target.tbl[i] = tbl[i];
        }
        target.shadow_rhs_cnt = shadow_rhs_cnt;
        target.compute_gpu = compute_gpu;
        target.stream_id = stream_id;
        target._net_send_buffer_cnt = _net_send_buffer_cnt;
        target._watch_types = watch_types_storage.empty() ? nullptr : watch_types_storage.data();
        target.mapping = mapping;
        target.trajec_requests = trajec_requests;
        target._fornetcon_perm_indices_size = _fornetcon_perm_indices_size;
        target._fornetcon_perm_indices = _fornetcon_perm_indices;
        target._fornetcon_weight_perm_size = _fornetcon_weight_perm_size;
        target._fornetcon_weight_perm = _fornetcon_weight_perm;
        target._pnt_offset = _pnt_offset;

        target.tml = tml_storage.empty() ? nullptr : tml_storage.data();
        target._ml_list = ml_ptrs.empty() ? nullptr : ml_ptrs.data();
        target.pntprocs = pntproc_storage.empty() ? nullptr : pntproc_storage.data();
        target.presyns = presyn_storage.empty() ? nullptr : presyn_storage.data();
        target.presyns_helper = presyn_helpers.empty() ? nullptr : presyn_helpers.data();
        target.pnt2presyn_ix = pnt2presyn_ix_ptrs.empty() ? nullptr : pnt2presyn_ix_ptrs.data();
        target.netcons = netcon_storage.empty() ? nullptr : netcon_storage.data();
        target.weights = weight_storage.empty() ? nullptr : weight_storage.data();
        target._net_send_buffer = net_send_buffer.empty() ? nullptr : net_send_buffer.data();
        target._net_send_buffer_size = static_cast<int>(net_send_buffer.size());

        target._v_parent_index = v_parent_index.empty() ? nullptr : v_parent_index.data();
        target._permute = node_permutation.empty() ? nullptr : node_permutation.data();
        target._actual_rhs = node_data_stride == 0 ? nullptr : thread_data_base;
        target._actual_d = node_data_stride == 0 ? nullptr : thread_data_base + node_data_stride;
        target._actual_a = node_data_stride == 0 ? nullptr : thread_data_base + 2 * node_data_stride;
        target._actual_b = node_data_stride == 0 ? nullptr : thread_data_base + 3 * node_data_stride;
        target._actual_v = node_data_stride == 0 ? nullptr : thread_data_base + 4 * node_data_stride;
        target._actual_area = node_data_stride == 0 ? nullptr : thread_data_base + 5 * node_data_stride;
        target._actual_diam = node_data_stride == 0 ? nullptr : thread_data_base + 6 * node_data_stride;
        target._shadow_rhs = shadow_rhs.empty() ? nullptr : shadow_rhs.data();
        target._shadow_d = shadow_d.empty() ? nullptr : shadow_d.data();
        target._data = data_storage.empty() ? nullptr : data_storage.data();
        target._ndata = data_storage.size();
        target._vdata = vdata_storage.empty() ? nullptr : vdata_storage.data();
        target._nvdata = vdata_storage.size();

        target.n_pntproc = static_cast<int>(pntproc_storage.size());
        target.n_weight = static_cast<int>(weight_storage.size());
        target.n_netcon = static_cast<int>(netcon_storage.size());
        target.n_input_presyn = static_cast<int>(input_presyns.size());
        target.n_presyn = static_cast<int>(presyn_storage.size());
        target.n_real_output = n_real_output;
    }

    void bind(int mechanism_capacity) {
        bind_into(*this, mechanism_capacity);
    }

    void sync_state_from(const coreneuron::NrnThread& source) {
        _t = source._t;
        _dt = source._dt;
        cj = source.cj;
        _stop_stepping = source._stop_stepping;
        shadow_rhs_cnt = source.shadow_rhs_cnt;
        compute_gpu = source.compute_gpu;
        stream_id = source.stream_id;
        _net_send_buffer_cnt = source._net_send_buffer_cnt;
        trajec_requests = source.trajec_requests;
    }
};

struct CoreNeuronData {
    std::unordered_map<std::string, int> mechanism_type{};
    std::vector<MechanismRuntimeInfo> mechanisms{};
    std::vector<CoreNeuronThread> threads{};
    std::vector<coreneuron::NrnThread> runtime_threads{};
    std::vector<CoreInputEventTarget> input_event_targets{};
    std::vector<CoreNetConRef> netcon_presyn_order{};
    MicroDeviceConfig device_config{};
    double dt{0.025};
    double celsius{6.3};
    int secondorder{0};
    bool gpu_device_runtime_active{false};
    int gpu_runtime_ref_count{0};
    double effective_mindelay{10.0};

    CoreNeuronData() = default;
    CoreNeuronData(const CoreNeuronData&) = delete;
    CoreNeuronData& operator=(const CoreNeuronData&) = delete;
    CoreNeuronData(CoreNeuronData&&) noexcept = delete;
    CoreNeuronData& operator=(CoreNeuronData&&) noexcept = delete;
    ~CoreNeuronData();

    [[nodiscard]] bool empty() const noexcept {
        return threads.empty();
    }

    [[nodiscard]] int mechanism_capacity() const noexcept {
        int capacity = core_mechanism_capacity();
        for (const auto& mechanism: mechanisms) {
            capacity = std::max(capacity, mechanism.type + 1);
        }
        return capacity;
    }

    void bind() {
        const int capacity = mechanism_capacity();
        runtime_threads.resize(threads.size());
        for (std::size_t i = 0; i < threads.size(); ++i) {
            auto& thread = threads[i];
            thread.bind(capacity);
            thread.bind_into(runtime_threads[i], capacity);
        }
    }

    [[nodiscard]] coreneuron::NrnThread* nrn_threads() noexcept {
        return runtime_threads.empty() ? nullptr : runtime_threads.data();
    }

    [[nodiscard]] const coreneuron::NrnThread* nrn_threads() const noexcept {
        return runtime_threads.empty() ? nullptr : runtime_threads.data();
    }

    void sync_threads_from_runtime() {
        const auto count = std::min(threads.size(), runtime_threads.size());
        for (std::size_t i = 0; i < count; ++i) {
            threads[i].sync_state_from(runtime_threads[i]);
        }
    }
};

}  // namespace mind_sim::micro::sim
