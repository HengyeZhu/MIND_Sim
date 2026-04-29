/*
# =============================================================================
# Copyright (c) 2016 - 2022 Blue Brain Project/EPFL
#
# See top-level LICENSE file for details.
# =============================================================================.
*/
#pragma once

#include <cassert>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace neuron::mechanism {

enum class field_role {
    parameter,
    assigned,
    state,
    range,
};

template <typename T>
struct field {
    using type = T;
    field(std::string name_)
        : name{std::move(name_)} {}
    field(std::string name_, int array_size_)
        : name{std::move(name_)}
        , array_size{array_size_} {}
    field(std::string name_, field_role role_)
        : role{role_}
        , name{std::move(name_)} {}
    field(std::string name_, field_role role_, int array_size_)
        : array_size{array_size_}
        , role{role_}
        , name{std::move(name_)} {}
    field(std::string name_, std::string semantics_)
        : name{std::move(name_)}
        , semantics{std::move(semantics_)} {}
    field(std::string name_, std::string semantics_, std::string target_)
        : name{std::move(name_)}
        , semantics{std::move(semantics_)}
        , target{std::move(target_)} {}
    field(std::string name_,
          std::string semantics_,
          std::string target_,
          int ion_conc_style_,
          int ion_rev_style_,
          bool ion_write_interior_ = false,
          bool ion_write_exterior_ = false)
        : name{std::move(name_)}
        , semantics{std::move(semantics_)}
        , target{std::move(target_)}
        , ion_conc_style{ion_conc_style_}
        , ion_rev_style{ion_rev_style_}
        , ion_write_interior{ion_write_interior_}
        , ion_write_exterior{ion_write_exterior_} {}
    int array_size{1};
    field_role role{field_role::range};
    std::string name{}, semantics{}, target{};
    int ion_conc_style{0};
    int ion_rev_style{0};
    bool ion_write_interior{false};
    bool ion_write_exterior{false};
};

namespace detail {
struct data_field_info {
    std::string name{};
    int array_size{1};
    field_role role{field_role::range};
};

struct dparam_field_info {
    std::string name{};
    std::string semantic{};
    std::string target{};
    int ion_conc_style{0};
    int ion_rev_style{0};
    bool ion_write_interior{false};
    bool ion_write_exterior{false};
};

void register_data_fields(int mech_type,
                          std::vector<data_field_info> const& param_info,
                          std::vector<dparam_field_info> const& dparam_info);
}  // namespace detail

template <typename... Fields>
static void register_data_fields(int mech_type, Fields const&... fields) {
    std::vector<detail::data_field_info> param_info{};
    std::vector<detail::dparam_field_info> dparam_info{};
    auto const process = [&](auto const& field) {
        using field_t = std::decay_t<decltype(field)>;
        using data_t = typename field_t::type;
        if constexpr (std::is_same_v<data_t, double>) {
            assert(field.semantics.empty());
            assert(field.target.empty());
            param_info.push_back(detail::data_field_info{
                .name = field.name,
                .array_size = field.array_size,
                .role = field.role,
            });
        } else {
            static_assert(std::is_same_v<data_t, int> || std::is_pointer_v<data_t>,
                          "only pointers, doubles and ints are supported");
            assert(field.array_size == 1);
            dparam_info.push_back(detail::dparam_field_info{
                .name = field.name,
                .semantic = field.semantics,
                .target = field.target,
                .ion_conc_style = field.ion_conc_style,
                .ion_rev_style = field.ion_rev_style,
                .ion_write_interior = field.ion_write_interior,
                .ion_write_exterior = field.ion_write_exterior,
            });
        }
    };
    static_cast<void>(std::initializer_list<int>{(static_cast<void>(process(fields)), 0)...});
    detail::register_data_fields(mech_type, param_info, dparam_info);
}

}  // namespace neuron::mechanism
