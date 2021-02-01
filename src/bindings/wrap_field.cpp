#include <pybind11/numpy.h>

#include <sstream>
#include <stdexcept>
#include <vector>

#include "field.hpp"
#include "wrappers.hpp"

void wrap_field(py::module& m) {
  py::class_<Field>(m, "Field")
      .def_property_readonly("grid", &Field::grid)
      .def_property_readonly("ncomp", &Field::ncomp)
      .def("get", [](const Field* f) { return fieldToArray(*f); })
      .def("set",
           [](Field* f, py::array_t<real> data) { setArrayInField(*f, data); });
}

py::array_t<real> fieldToArray(const Field& f) {
  real* data = new real[f.grid().ncells() * f.ncomp()];
  f.getData(data);

  // Create a Python object that will free the allocated
  // memory when destroyed
  // TODO: figure out how this works
  // https://stackoverflow.com/questions/44659924/returning-numpy-arrays-via-pybind11
  py::capsule free_when_done(data, [](void* p) {
    real* data = reinterpret_cast<real*>(p);
    delete[] data;
  });

  int shape[4];
  shape[0] = f.ncomp();
  shape[1] = f.grid().size().z;
  shape[2] = f.grid().size().y;
  shape[3] = f.grid().size().x;

  int strides[4];
  strides[0] = sizeof(real) * shape[3] * shape[2] * shape[1];
  strides[1] = sizeof(real) * shape[3] * shape[2];
  strides[2] = sizeof(real) * shape[3];
  strides[3] = sizeof(real);

  return py::array_t<real>(shape, strides, data, free_when_done);
}

void setArrayInField(Field& f, py::array_t<real> data) {
  ssize_t ndim = data.ndim();

  if (ndim == 1) {
    if (data.shape(0) != f.ncomp()) {
      std::stringstream ss;
      ss << "The number of components do not match, "
         << "expected " << data.shape(0) << ", got " << f.ncomp() << ".";
      throw std::invalid_argument(ss.str());
    }

    py::buffer_info buf = data.request();
    real* cValues = reinterpret_cast<real*>(buf.ptr);

    int N = f.grid().ncells() * f.ncomp();
    int nCells = N / f.ncomp();

    std::vector<real> buffer(N);

    for (int i = 0; i < N; i++) {
      int c = i / nCells;
      buffer[i] = cValues[c];
    }

    f.setData(buffer.data());
  } else if (ndim == 4) {
    std::vector<int> shape(ndim);
    shape[0] = f.ncomp();
    shape[1] = f.grid().size().z;
    shape[2] = f.grid().size().y;
    shape[3] = f.grid().size().x;

    for (ssize_t i = 0; i < ndim; i++) {
      if (shape[i] != data.shape(i)) {
        std::stringstream ss;
        ss << "The shape of the data does not match the shape of the field, "
           << "expected (" << shape[0] << ", " << shape[1] << ", " << shape[2]
           << ", " << shape[3] << "), got (" << data.shape(0) << ", "
           << data.shape(1) << ", " << data.shape(2) << ", " << data.shape(3)
           << ").";
        throw std::invalid_argument(ss.str());
      }
    }
    py::buffer_info buf = data.request();
    f.setData(reinterpret_cast<real*>(buf.ptr));

  } else {
    throw std::invalid_argument(
        "The shape of the data does not match the shape of the field");
  }
}
