#pragma once

#include "micro/sim/core_neuron_data.hpp"
#include "micro/sim/types.hpp"

#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace mind_sim::micro::sim {

struct MicroWindowToken {
    double start_time{0.0};
    double stop_time{0.0};
    std::size_t spike_begin{0};
    std::size_t spike_end{0};
    bool submitted{false};
    bool ran_synchronously{false};
};

class MicroRuntime {
  public:
    explicit MicroRuntime(CoreNeuronData& core_data);
    MicroRuntime(const MicroRuntime&) = delete;
    MicroRuntime& operator=(const MicroRuntime&) = delete;
    MicroRuntime(MicroRuntime&&) noexcept = delete;
    MicroRuntime& operator=(MicroRuntime&&) noexcept = delete;
    ~MicroRuntime();

    void finitialize(double voltage);

    [[nodiscard]] mind_sim::micro::sim::MicroEventTable& scheduled_events() noexcept;

    [[nodiscard]] bool uses_gpu() const noexcept;
    [[nodiscard]] MicroWindowToken submit_window(double start_time, double stop_time);
    mind_sim::micro::sim::MicroSpikeTableView finish_window(MicroWindowToken& token);
    mind_sim::micro::sim::MicroSpikeTableView advance_window(double start_time, double stop_time);

  private:
    void bind_core_globals();
    void ensure_core_globals_bound();
    void ensure_registered_mechanisms();
    void require_registered_mechanisms() const;
    void initialize_private_mechanism_storage();
    void ensure_private_mechanism_storage_initialized();
    void require_private_mechanism_storage_initialized() const;
    void ensure_device_runtime_ready();
    void release_runtime_ref() noexcept;
    [[nodiscard]] MicroWindowToken prepare_window(double start_time, double stop_time);
    void run_prepared_window(double stop_time);
    void ensure_window_worker_started();
    void window_worker_loop();
    void submit_worker_window(double stop_time);
    void wait_worker_window();
    void stop_window_worker() noexcept;
    mind_sim::micro::sim::MicroSpikeTableView spike_view_from(std::size_t spike_begin,
                                                              std::size_t spike_end) const;

    CoreNeuronData* core_data_{nullptr};
    mind_sim::micro::sim::MicroEventTable scheduled_events_{};
    bool core_globals_bound_{false};
    bool mechanisms_checked_{false};
    bool private_mechanism_storage_checked_{false};
    bool runtime_ref_held_{false};
    bool window_active_{false};
    std::thread window_worker_{};
    std::mutex window_worker_mutex_{};
    std::condition_variable window_worker_cv_{};
    bool window_worker_stop_{false};
    bool window_worker_has_job_{false};
    bool window_worker_job_done_{true};
    double window_worker_stop_time_{0.0};
    std::size_t window_worker_spike_end_{0};
    std::exception_ptr window_worker_exception_{nullptr};
};

}  // namespace mind_sim::micro::sim
