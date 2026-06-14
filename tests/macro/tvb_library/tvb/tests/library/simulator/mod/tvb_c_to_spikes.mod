NEURON {
    ARTIFICIAL_CELL tvb_c_to_spikes
    THREADSAFE
    RANGE c
}

MIND {
    ROLE MACRO2MICRO
    READ_SOURCE c
}

ASSIGNED {
    c
}

NET_RECEIVE(weight) {
}
