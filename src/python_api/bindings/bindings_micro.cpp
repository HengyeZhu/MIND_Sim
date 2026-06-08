#include "python_api/bindings/bindings.hpp"

#include <algorithm>

namespace mind_sim::python_api::bindings {

std::vector<Sim*>& default_micro_registry() {
    static std::vector<Sim*> registry;
    return registry;
}

Sim::Sim() {
    register_default_micro(this);
}

Sim::~Sim() {
    unregister_default_micro(this);
}

void register_default_micro(Sim* sim) {
    if (sim == nullptr) {
        return;
    }
    auto& registry = default_micro_registry();
    if (std::find(registry.begin(), registry.end(), sim) == registry.end()) {
        registry.push_back(sim);
    }
}

void unregister_default_micro(Sim* sim) {
    auto& registry = default_micro_registry();
    registry.erase(std::remove(registry.begin(), registry.end(), sim), registry.end());
}

Sim& default_micro() {
    auto& registry = default_micro_registry();
    if (registry.empty()) {
        throw std::runtime_error("ROI.use_micro() requires one ms.Sim() to exist");
    }
    if (registry.size() != 1) {
        throw std::runtime_error("ROI.use_micro() requires exactly one ms.Sim()");
    }
    return *registry.front();
}

void bind_micro(nb::module_& m) {
    nb::class_<SectionSpec>(m, "section")
        .def(nb::init<const std::string&, const std::string&>(), nb::arg("name"), nb::arg("label"))
        .def_rw("name", &SectionSpec::name)
        .def(
            "connect",
            [](SectionSpec& child, nb::handle parent, double parentx) {
                if (parentx < 0.0 || parentx > 1.0) {
                    throw std::runtime_error("section.connect parentx must be in [0, 1]");
                }
                child.parent_name = parent.is_none() ? std::string{} : section_name_from_arg(parent);
                child.parentx = parentx;
            },
            nb::arg("parent").none(),
            nb::arg("parentx") = 1.0)
        .def_prop_ro("parent",
                     [](const SectionSpec& sec) -> nb::object {
                         return sec.parent_name.empty() ? nb::none() : nb::str(sec.parent_name.c_str());
                     })
        .def_prop_ro("parentx", [](const SectionSpec& sec) { return sec.parentx; })
        .def_prop_ro("label", [](const SectionSpec& sec) { return sec.label; })
        .def_rw("nseg", &SectionSpec::nseg)
        .def_rw("L_um", &SectionSpec::L_um)
        .def_rw("diam_um", &SectionSpec::diam_um)
        .def_prop_rw("pt3d",
                     [](const SectionSpec& sec) { return pt3d_to_list(sec.pt3d); },
                     [](SectionSpec& sec, nb::handle value) { sec.pt3d = pt3d_from_obj(value); })
        .def("__repr__", [](const SectionSpec& sec) { return "<mind_sim.section name='" + sec.name + "'>"; });

    nb::class_<SectionListIter>(m, "_section_list_iterator")
        .def("__iter__", [](SectionListIter& it) -> SectionListIter& { return it; })
        .def("__next__", &SectionListIter::next_item, nb::rv_policy::reference_internal);

    nb::class_<SectionList>(m, "section_list")
        .def(nb::init<>())
        .def("__len__", [](const SectionList& list) { return list.sections->size(); })
        .def("__getitem__",
             [](SectionList& list, int index) -> SectionSpec& {
                 return list.sections->at(
                     static_cast<std::size_t>(normalize_py_index(index, list.sections->size(), "section_list")));
             },
             nb::rv_policy::reference_internal)
        .def("__iter__",
             [](const SectionList& list) {
                 return SectionListIter{list.sections, 0};
             })
        .def("append", [](SectionList& list, const SectionSpec& sec) { list.sections->push_back(sec); })
        .def("extend",
             [](SectionList& list, nb::handle items) {
                 auto parsed = parse_sections(items);
                 list.sections->insert(list.sections->end(), parsed.begin(), parsed.end());
             })
        .def("delete_subtree",
             [](SectionList& list, nb::handle sec) -> SectionList& {
                 *list.sections = mind_micro_frontend::delete_subtree_specs(*list.sections, section_name_from_arg(sec));
                 return list;
             },
             nb::rv_policy::reference_internal)
        .def("delete_label",
             [](SectionList& list, const std::string& label) -> SectionList& {
                 *list.sections = mind_micro_frontend::delete_label_specs(*list.sections, label);
                 return list;
             },
             nb::rv_policy::reference_internal)
        .def("to_list",
             [](const SectionList& list) {
                 nb::list out;
                 for (const auto& sec : *list.sections) {
                     out.append(sec);
                 }
                 return out;
             })
        .def("__repr__",
             [](const SectionList& list) {
                 return "<mind_sim.section_list size=" + std::to_string(list.sections->size()) + ">";
             });

    nb::class_<mind_micro_morph::SectionDistanceLayout>(m, "section_distance_layout")
        .def("distance",
             [](const mind_micro_morph::SectionDistanceLayout& layout,
                const std::string& a_sec,
                double a_x,
                const std::string& b_sec,
                double b_x) {
                 return mind_micro_morph::distance_between_locs_um(layout, a_sec, a_x, b_sec, b_x);
             })
        .def("diam",
             [](const mind_micro_morph::SectionDistanceLayout& layout, const std::string& sec, double x) {
                 mind_micro_morph::SectionLocationResolveInfo info;
                 if (!mind_micro_morph::resolve_section_location(layout, sec, x, &info)) {
                     throw std::runtime_error("section location could not be resolved");
                 }
                 return info.diam_um;
             });

    m.def("load_swc_sections", &mind_micro_frontend::load_swc_sections, nb::arg("swc_file"));
    m.def("load_asc_sections", &mind_micro_frontend::load_asc_sections, nb::arg("asc_file"));
    m.def("load_swc_section_list",
          [](const std::string& swc_file) {
              return make_section_list(mind_micro_frontend::load_swc_sections(swc_file));
          },
          nb::arg("swc_file"));
    m.def("load_asc_section_list",
          [](const std::string& asc_file) {
              return make_section_list(mind_micro_frontend::load_asc_sections(asc_file));
          },
          nb::arg("asc_file"));
    m.def("build_section_distance_layout",
          [](nb::handle sections) {
              mind_micro_frontend::SectionNseg nseg;
              auto morph = mind_micro_frontend::build_morph_from_sections(parse_sections(sections), nseg);
              mind_micro_frontend::apply_section_nseg(morph, nseg);
              return mind_micro_morph::build_section_distance_layout(morph);
          },
          nb::arg("sections"));

    nb::class_<PopulationIter>(m, "_population_iterator")
        .def("__iter__", [](PopulationIter& it) -> PopulationIter& { return it; })
        .def("__next__", &PopulationIter::next_item);

    nb::class_<PopulationView>(m, "population")
        .def_ro("name", &PopulationView::name)
        .def_ro("gid_begin", &PopulationView::gid_begin)
        .def_ro("gid_end", &PopulationView::gid_end)
        .def("__len__", &PopulationView::size)
        .def("__getitem__",
             [](const PopulationView& pop, int index) {
                 return pop.at(static_cast<std::size_t>(normalize_py_index(index, pop.size(), "population")));
             })
        .def("__iter__", [](const PopulationView& pop) { return PopulationIter{pop, 0}; })
        .def("__repr__", [](const PopulationView& pop) { return "<mind_sim.population name='" + pop.name + "'>"; });

    nb::class_<CellView>(m, "cell")
        .def_ro("gid", &CellView::gid)
        .def_prop_rw("v_init", &CellView::get_v_init, &CellView::set_v_init)
        .def("group", &CellView::group, nb::arg("label"))
        .def("__repr__", [](const CellView& cell) { return "<mind_sim.cell gid=" + std::to_string(cell.gid) + ">"; });

    nb::class_<SectionGroupIter>(m, "_section_group_iterator")
        .def("__iter__", [](SectionGroupIter& it) -> SectionGroupIter& { return it; })
        .def("__next__", &SectionGroupIter::next_item);

    nb::class_<SegmentValueBatchView>(m, "segment_values")
        .def("__len__", &SegmentValueBatchView::size)
        .def("__repr__",
             [](const SegmentValueBatchView& batch) {
                 return "<mind_sim.segment_values size=" + std::to_string(batch.size()) + ">";
             });

    nb::class_<SectionGroupView>(m, "section_group")
        .def("__len__", &SectionGroupView::size)
        .def("__getitem__",
             [](const SectionGroupView& group, int index) {
                 return group.at(static_cast<std::size_t>(normalize_py_index(index, group.size(), "section_group")));
             })
        .def("__iter__", [](const SectionGroupView& group) { return SectionGroupIter{group, 0}; })
        .def("segment_values", &SectionGroupView::segment_values, nb::arg("section_indices"), nb::arg("values"))
        .def("insert",
             [](const SectionGroupView& group, const std::string& mech, const nb::kwargs& kwargs) {
                 group.insert(mech, parse_kwargs(kwargs));
             })
        .def("ion_style",
             nb::overload_cast<const std::string&>(&SectionGroupView::ion_style, nb::const_),
             nb::arg("ion_mechanism"))
        .def("ion_style",
             nb::overload_cast<const std::string&, int, int, int, int, int>(
                 &SectionGroupView::ion_style,
                 nb::const_),
             nb::arg("ion_mechanism"),
             nb::arg("c_style"),
             nb::arg("e_style"),
             nb::arg("einit"),
             nb::arg("eadvance"),
             nb::arg("cinit"))
        .def("__getattr__",
             [](const SectionGroupView& group, const std::string& key) {
                 if (key == "v_init") {
                     return group.get_v_init();
                 }
                 if (key == "cm") {
                     return group.get_cm();
                 }
                 if (key == "Ra") {
                     return group.get_Ra();
                 }
                 return group.get_ion_range(key);
             })
        .def("__setattr__",
             [](SectionGroupView& group, const std::string& key, nb::handle value) {
                 if (key == "v_init") {
                     group.set_v_init(nb::cast<double>(value));
                     return;
                 }
                 if (key == "cm") {
                     group.set_cm(nb::cast<double>(value));
                     return;
                 }
                 if (key == "Ra") {
                     group.set_Ra(nb::cast<double>(value));
                     return;
                 }
                 group.set_ion_range(key, parse_param_value(value));
             })
        .def_prop_rw("v_init", &SectionGroupView::get_v_init, &SectionGroupView::set_v_init)
        .def_prop_rw("cm", &SectionGroupView::get_cm, &SectionGroupView::set_cm)
        .def_prop_rw("Ra", &SectionGroupView::get_Ra, &SectionGroupView::set_Ra)
        .def("__repr__",
             [](const SectionGroupView& group) {
                 return "<mind_sim.section_group label='" + group.label + "' size=" +
                        std::to_string(group.size()) + ">";
             });

    nb::class_<SectionView>(m, "section_view")
        .def_prop_ro("gid", [](const SectionView& sec) { return sec.gid; })
        .def_prop_ro("section_index", [](const SectionView& sec) { return sec.section_index; })
        .def_prop_ro("label", &SectionView::label)
        .def("loc", &SectionView::loc, nb::arg("loc"))
        .def("__call__", &SectionView::loc, nb::arg("loc"))
        .def("insert",
             [](const SectionView& sec, const std::string& mech, const nb::kwargs& kwargs) {
                 return sec.insert(mech, parse_kwargs(kwargs));
             })
        .def("ref",
             &SectionView::ref,
             nb::arg("var"),
             nb::arg("mech") = "global",
             nb::arg("array_index") = -1)
        .def_prop_ro("_ref_v", &SectionView::ref_v)
        .def("__repr__",
             [](const SectionView& sec) {
                 return "<mind_sim.section_view gid=" + std::to_string(sec.gid) +
                        " section_index=" + std::to_string(sec.section_index) + ">";
             });

    nb::class_<VariableRefView>(m, "variable_ref")
        .def("value", &VariableRefView::value)
        .def("__repr__", [](const VariableRefView&) { return "<mind_sim.variable_ref>"; });

    nb::class_<PointProcessView>(m, "point_process")
        .def_ro("insert_id", &PointProcessView::insert_id)
        .def("ref", &PointProcessView::ref, nb::arg("var"), nb::arg("array_index") = -1)
        .def("__getattr__",
             [](const PointProcessView& mechanism, const std::string& key) {
                 return mechanism.get_var(key);
             })
        .def("__setattr__",
             [](PointProcessView& mechanism, const std::string& key, nb::handle value) {
                 mechanism.set_var(key, nb::cast<double>(value));
             })
        .def("__repr__",
             [](const PointProcessView& mechanism) {
                 return "<mind_sim.point_process insert_id=" + std::to_string(mechanism.insert_id) + ">";
             });

    nb::class_<ArtificialCellView>(m, "artificial_cell")
        .def_ro("insert_id", &ArtificialCellView::insert_id)
        .def("ref", &ArtificialCellView::ref, nb::arg("var"), nb::arg("array_index") = -1)
        .def("__getattr__",
             [](const ArtificialCellView& mechanism, const std::string& key) {
                 return mechanism.get_var(key);
             })
        .def("__setattr__",
             [](ArtificialCellView& mechanism, const std::string& key, nb::handle value) {
                 mechanism.set_var(key, nb::cast<double>(value));
             })
        .def("__repr__",
             [](const ArtificialCellView& mechanism) {
                 return "<mind_sim.artificial_cell insert_id=" + std::to_string(mechanism.insert_id) + ">";
             });

    nb::class_<NetConWeightView>(m, "netcon_weight")
        .def("__len__", &NetConWeightView::size)
        .def("__getitem__",
             [](const NetConWeightView& w, int index) {
                 return w.get(normalize_py_index(index, w.size(), "netcon_weight"));
             })
        .def("__setitem__",
             [](NetConWeightView& w, int index, double value) {
                 w.set(normalize_py_index(index, w.size(), "netcon_weight"), value);
             });

    nb::class_<NetConView>(m, "netcon")
        .def_prop_ro("id", &NetConView::id)
        .def_prop_ro("runtime_index", &NetConView::runtime_index)
        .def_prop_ro("target_event_target_id", &NetConView::target_event_target_id)
        .def_prop_ro("source_event_target_id", &NetConView::source_event_target_id)
        .def_prop_ro("weight", &NetConView::weight_view)
        .def("wcnt", &NetConView::wcnt)
        .def_prop_rw("delay", &NetConView::get_delay, &NetConView::set_delay)
        .def_prop_rw("threshold", &NetConView::get_threshold, &NetConView::set_threshold)
        .def("__repr__",
             [](const NetConView& nc) {
                 return "<mind_sim.netcon id=" + std::to_string(nc.connection_id) + ">";
             });

    nb::class_<NetworkView>(m, "network")
        .def("register_spike_source",
             &NetworkView::register_spike_source,
             nb::arg("sid"),
             nb::arg("source"),
             nb::arg("threshold") = nb::none())
        .def("sid_connect",
             &NetworkView::sid_connect,
             nb::arg("sid"),
             nb::arg("post"),
             nb::arg("weight"),
             nb::arg("delay"))
        .def("event_connect",
             [](const NetworkView& network,
                const ArtificialCellView& source,
                const PointProcessView& post,
                double weight,
                double delay) {
                 return network.event_connect(source, post, weight, delay);
             },
             nb::arg("source"),
             nb::arg("post"),
             nb::arg("weight"),
             nb::arg("delay"))
        .def("event_connect",
             [](const NetworkView& network,
                const PointProcessView& source,
                const PointProcessView& post,
                double weight,
                double delay) {
                 return network.event_connect(source, post, weight, delay);
             },
             nb::arg("source"),
             nb::arg("post"),
             nb::arg("weight"),
             nb::arg("delay"))
        .def("__repr__", [](const NetworkView&) { return "<mind_sim.network>"; });

    nb::class_<TimeRefView>(m, "_time_ref")
        .def("__repr__", [](const TimeRefView&) { return "<mind_sim._time_ref t>"; });

    nb::class_<VectorView>(m, "Vector")
        .def(nb::init<>())
        .def("record", nb::overload_cast<const VariableRefView&>(&VectorView::record), nb::arg("ref"))
        .def("record", nb::overload_cast<const TimeRefView&>(&VectorView::record), nb::arg("ref"))
        .def("to_python", &VectorView::to_python)
        .def("save_h5",
             [](const VectorView& vec, const std::string& path, const std::string& name) {
                 mind_sim::io::save_vector_h5(vec.buffer ? vec.buffer->samples : std::vector<double>{},
                                                path,
                                                name);
             },
             nb::arg("path"),
             nb::arg("name") = "values")
        .def("size", &VectorView::size)
        .def("__len__", &VectorView::size)
        .def("__getitem__",
             [](const VectorView& vec, int index) {
                 return vec.get(index);
             })
        .def("__repr__",
             [](const VectorView& vec) {
                 return "<mind_sim.Vector size=" + std::to_string(vec.size()) + ">";
             });

    nb::class_<Sim>(m, "Sim")
        .def(nb::init<>())
        .def_rw("name", &Sim::name)
        .def("set_device", &Sim::set_device)
        .def("set_dt", &Sim::set_dt)
        .def("get_dt", &Sim::get_dt)
        .def("set_num_threads", &Sim::set_num_threads)
        .def("get_num_threads", &Sim::get_num_threads)
        .def("load_mech", &Sim::load_mech)
        .def("ion_register", &Sim::ion_register, nb::arg("ion"), nb::arg("charge"))
        .def("ion_charge", &Sim::ion_charge, nb::arg("ion_mechanism"))
        .def("get_loaded_mech_paths", &Sim::get_loaded_mech_paths)
        .def("__getattr__",
             [](const Sim& sim, const std::string& key) {
                 if (key == "name") {
                     return nb::cast(sim.name);
                 }
                 return nb::cast(sim.get_global(key));
             })
        .def("__setattr__",
             [](Sim& sim, const std::string& key, nb::handle value) {
                 if (key == "name") {
                     sim.name = nb::cast<std::string>(value);
                     return;
                 }
                 if (key == "celsius") {
                     sim.set_celsius(nb::cast<double>(value));
                     return;
                 }
                 sim.set_global(key, nb::cast<double>(value));
             })
        .def("insert",
             [](Sim& sim, const std::string& mech, const nb::kwargs& kwargs) {
                 return sim.insert(mech, parse_kwargs(kwargs));
             })
        .def("build_morphology", &Sim::build_morphology, nb::arg("morph_templates"), nb::rv_policy::reference_internal)
        .def("__len__", [](const Sim& sim) { return sim.model.population_count(); })
        .def("population", &sim_population, nb::arg("name"), nb::keep_alive<0, 1>())
        .def("populations", &sim_populations)
        .def("network", &sim_network, nb::keep_alive<0, 1>())
        .def("build_microcircuit", &Sim::build_microcircuit)
        .def("finitialize", &Sim::finitialize, nb::arg("v_init"))
        .def("run", &Sim::run, nb::arg("tstop"), nb::call_guard<nb::gil_scoped_release>())
        .def("continue_run",
             &Sim::continue_run,
             nb::arg("runtime"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("fadvance", &Sim::fadvance, nb::call_guard<nb::gil_scoped_release>())
        .def("get_t", &Sim::get_t)
        .def_prop_rw("celsius", &Sim::get_celsius, &Sim::set_celsius)
        .def_prop_ro("_ref_t", &sim_time_ref);

}

}  // namespace mind_sim::python_api::bindings
