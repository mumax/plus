#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "datatypes.hpp"

namespace pybind11 {
namespace detail {

// int3 caster by a small wrapper around the array<int, 3> caster
template <>
struct type_caster<int3> {
  typedef std::array<int, 3> i3;

 public:
  PYBIND11_TYPE_CASTER(int3, _("int3"));

  bool load(handle src, bool convert) {
    if (!i3_caster.load(src, convert)) {
      return false;
    }
    auto arr = (i3)i3_caster;
    value = {arr[0], arr[1], arr[2]};
    return true;
  }

  static handle cast(int3 src, return_value_policy policy, handle parent) {
    i3 arr{src.x, src.y, src.z};
    return type_caster<i3>::cast(arr, policy, parent);
  }

 private:
  type_caster<i3> i3_caster;
};

template <>
struct type_caster<real3> {
  typedef std::array<real, 3> r3;

 public:
  PYBIND11_TYPE_CASTER(real3, _("real3"));

  bool load(handle src, bool convert) {
    if (!r3_caster.load(src, convert)) {
      return false;
    }
    auto arr = (r3)r3_caster;
    value = {arr[0], arr[1], arr[2]};
    return true;
  }

  static handle cast(real3 src, return_value_policy policy, handle parent) {
    r3 arr{src.x, src.y, src.z};
    return type_caster<r3>::cast(arr, policy, parent);
  }

 private:
  type_caster<r3> r3_caster;
};
}  // namespace detail
}  // namespace pybind11
