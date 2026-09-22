#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <type_traits>

#include "field.hpp"

namespace py = pybind11;

template <typename T = real>
py::array_t<T> fieldToArray(const Field& f) {
  real* fieldData = new real[f.grid().ncells() * f.ncomp()];
  f.getData(fieldData);

  // Cast FieldData to type T if not real
  T* data;
  if (std::is_same<T, real>::value) {
    data = reinterpret_cast<T*>(fieldData);
  } else {
    data = new T[f.grid().ncells() * f.ncomp()];
    for (size_t i = 0; i < f.grid().ncells() * f.ncomp(); ++i) {
      data[i] = static_cast<T>(fieldData[i]);
    }
    delete[] fieldData;
  }

  // Create a Python object that will free the allocated
  // memory when destroyed
  // TODO: figure out how this works
  // https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
  py::capsule free_when_done(data, [](void* p) {
    T* data = reinterpret_cast<T*>(p);
    delete[] data;
  });

  int shape[4] = {static_cast<int>(f.ncomp()), static_cast<int>(f.grid().size().z),
                  static_cast<int>(f.grid().size().y),
                  static_cast<int>(f.grid().size().x)};

  int strides[4];
  strides[0] = sizeof(T) * shape[3] * shape[2] * shape[1];
  strides[1] = sizeof(T) * shape[3] * shape[2];
  strides[2] = sizeof(T) * shape[3];
  strides[3] = sizeof(T);

  return py::array_t<T>(shape, strides, data, free_when_done);
}
