#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tuple>

#include "datatypes.hpp"

namespace pybind11 {
namespace detail {

/** Cast int3 to tuple and vice versa. */
template <>
struct type_caster<int3> {
  typedef std::tuple<int, int, int> i3;

 public:
  /**
   * This macro establishes the name 'int3' in
   * function signatures and declares a local variable
   * 'value' of type int3
   */
  PYBIND11_TYPE_CASTER(int3, _("int3"));

  /**
   * Conversion part 1 (Python->C++): convert a PyObject into an int3
   * instance or return false upon failure. The second argument
   * indicates whether implicit conversions should be applied.
   */
  bool load(handle src, bool convert) {
    if (!i3_caster.load(src, convert)) {
      return false;
    }

    auto tuple = static_cast<i3>(i3_caster);
    auto [x, y, z] = tuple;
    value = {x, y, z};

    return true;
  }

  /**
   * Conversion part 2 (C++ -> Python): convert an int3 instance into
   * a Python object. The second and third arguments are used to
   * indicate the return value policy and parent object (for
   * ``return_value_policy::reference_internal``) and are generally
   * ignored by implicit casters.
   */
  static handle cast(int3 src, return_value_policy policy, handle parent) {
    auto tuple = std::make_tuple(src.x, src.y, src.z);

    return type_caster<i3>::cast(tuple, policy, parent);
  }

 private:
  type_caster<i3> i3_caster;
};

/** Cast real3 to tuple and vice versa. */
template <>
struct type_caster<real3> {
  typedef std::tuple<real, real, real> r3;

 public:
  /**
   * This macro establishes the name 'real3' in
   * function signatures and declares a local variable
   * 'value' of type real3
   */
  PYBIND11_TYPE_CASTER(real3, _("real3"));

  /**
   * Conversion part 1 (Python->C++): convert a PyObject into a real3
   * instance or return false upon failure. The second argument
   * indicates whether implicit conversions should be applied.
   */
  bool load(handle src, bool convert) {
    if (!r3_caster.load(src, convert)) {
      return false;
    }

    auto tuple = static_cast<r3>(r3_caster);
    auto [x, y, z] = tuple;
    value = {x, y, z};

    return true;
  }

  /**
   * Conversion part 2 (C++ -> Python): convert a real3 instance into
   * a Python object. The second and third arguments are used to
   * indicate the return value policy and parent object (for
   * ``return_value_policy::reference_internal``) and are generally
   * ignored by implicit casters.
   */
  static handle cast(real3 src, return_value_policy policy, handle parent) {
    auto tuple = std::make_tuple(src.x, src.y, src.z);

    return type_caster<r3>::cast(tuple, policy, parent);
  }

 private:
  type_caster<r3> r3_caster;
};
}  // namespace detail
}  // namespace pybind11
