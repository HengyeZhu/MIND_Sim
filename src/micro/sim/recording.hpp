#pragma once

#include "micro/sim/core_neuron_data.hpp"

#include <cstddef>
#include <vector>

namespace mind_sim::micro::sim {

struct CoreThreadRecording {
    std::vector<void*> vpr{};
    std::vector<double*> gather{};
    std::vector<double*> varrays{};
    coreneuron::TrajectoryRequests request{};

    [[nodiscard]] bool active() const noexcept {
        return !gather.empty();
    }
};

class CoreRecordingPlan {
  public:
    explicit CoreRecordingPlan(CoreNeuronData& core_data);
    CoreRecordingPlan(const CoreRecordingPlan&) = delete;
    CoreRecordingPlan& operator=(const CoreRecordingPlan&) = delete;
    CoreRecordingPlan(CoreRecordingPlan&&) = delete;
    CoreRecordingPlan& operator=(CoreRecordingPlan&&) = delete;
    ~CoreRecordingPlan();

    void add_target(double* value, double* samples);
    void prepare(int sample_count);
    void activate();
    void update_host();
    void deactivate() noexcept;

    [[nodiscard]] bool active() const noexcept;
    [[nodiscard]] int recorded_sample_count() const;
    void validate_recorded_sample_count(int expected) const;

  private:
    [[nodiscard]] bool has_targets() const noexcept;

    CoreNeuronData* core_data_{nullptr};
    std::vector<CoreThreadRecording> threads_{};
    bool prepared_{false};
    bool active_{false};
    bool device_request_active_{false};
};

[[nodiscard]] int thread_index_for_data_pointer(const CoreNeuronData& core_data,
                                                const double* ptr);

}  // namespace mind_sim::micro::sim
