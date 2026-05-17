#include "python_api/bindings/bindings.hpp"

NB_MODULE(_native, m) {
    m.doc() = "MIND_Sim native C++ frontend and simulator bindings";
    mind_sim::python_api::bindings::bind_rules(m);
    mind_sim::python_api::bindings::bind_io(m);
    mind_sim::python_api::bindings::bind_micro(m);
    mind_sim::python_api::bindings::bind_macro(m);
}
