NEURON {
    ARTIFICIAL_CELL ca3_input_to_spikes
    THREADSAFE
    RANGE ca3_input, rate, start_time, stop_time
    RANGE base_hz, gain_hz, max_rate_hz, threshold, slope
    RANDOM rng
}

MIND {
    ROLE MACRO2MICRO
    TARGET_INPUT ca3_input
}

PARAMETER {
    base_hz = 1.0
    gain_hz = 45.0
    max_rate_hz = 120.0
    threshold = -0.35
    slope = 4.0
}

ASSIGNED {
    ca3_input
    rate
    start_time
    stop_time
    window_ms
    lambda
    spike_count
    count
    event_time
}

PROCEDURE update_rate() {
    rate = base_hz + gain_hz / (1.0 + exp(-slope * (ca3_input - threshold)))
    if (rate > max_rate_hz) {
        rate = max_rate_hz
    }
    if (rate < 0.0) {
        rate = 0.0
    }
}

FUNCTION poisson_count(mean) {
    LOCAL limit, product
    poisson_count = 0.0
    if (mean > 0.0) {
        limit = exp(-mean)
        product = 1.0
        poisson_count = -1.0
        while (product > limit) {
            poisson_count = poisson_count + 1.0
            product = product * random_uniform(rng)
        }
    }
}

NET_RECEIVE(weight) {
    if (flag == 0.0) {
        update_rate()
        if (rate > 0.0) {
            window_ms = stop_time - t
            lambda = rate * window_ms / 1000.0
            spike_count = poisson_count(lambda)
            count = 0.0
            while (count < spike_count) {
                event_time = t + window_ms * random_uniform(rng)
                net_send(event_time - t, 1.0)
                count = count + 1.0
            }
        }
    }
    if (flag == 1.0) {
        net_event(t)
    }
}
