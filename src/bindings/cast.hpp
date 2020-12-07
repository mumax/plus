#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "datatypes.hpp"

namespace pybind11 {
namespace detail {

/** Cast int3 to tuple and vice versa. */
template <>
struct type_caster<int3> {
  typedef std::array<int, 3> i3;

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

    auto arr = (i3)i3_caster;
    value = {arr[0], arr[1], arr[2]};

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
    size_t size = 3;
    tuple out(size);

    auto out_x = reinterpret_steal<object>(PyLong_FromLong(src.x));
    auto out_y = reinterpret_steal<object>(PyLong_FromLong(src.y));
    auto out_z = reinterpret_steal<object>(PyLong_FromLong(src.z));

    if (!out_x || !out_y || !out_z) {
      return handle();
    }

    PyTuple_SET_ITEM(out.ptr(), 0,
                     out_x.release().ptr());  // steals a reference
    PyTuple_SET_ITEM(out.ptr(), 1,
                     out_y.release().ptr());  // steals a reference
    PyTuple_SET_ITEM(out.ptr(), 2,
                     out_z.release().ptr());  // steals a reference

    return out.release();
  }

 private:
  type_caster<i3> i3_caster;
};

/** Cast real3 to tuple and vice versa. */
template <>
struct type_caster<real3> {
  typedef std::array<real, 3> r3;

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

    auto arr = (r3)r3_caster;
    value = {arr[0], arr[1], arr[2]};

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
    size_t size = 3;
    tuple out(size);

    auto out_x = reinterpret_steal<object>(PyFloat_FromDouble(src.x));
    auto out_y = reinterpret_steal<object>(PyFloat_FromDouble(src.y));
    auto out_z = reinterpret_steal<object>(PyFloat_FromDouble(src.z));

    if (!out_x || !out_y || !out_z) {
      return handle();
    }

    PyTuple_SET_ITEM(out.ptr(), 0,
                     out_x.release().ptr());  // steals a reference
    PyTuple_SET_ITEM(out.ptr(), 1,
                     out_y.release().ptr());  // steals a reference
    PyTuple_SET_ITEM(out.ptr(), 2,
                     out_z.release().ptr());  // steals a reference

    return out.release();
  }

 private:
  type_caster<r3> r3_caster;
};
}  // namespace detail
}  // namespace pybind11
