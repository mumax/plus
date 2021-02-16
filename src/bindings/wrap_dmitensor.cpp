#include "dmitensor.hpp"
#include "wrappers.hpp"

void wrap_dmitensor(py::module& m) {
  py::class_<DmiTensor>(m, "DmiTensor")
      .def_readonly("xxy", &DmiTensor::xxy)
      .def_readonly("xyz", &DmiTensor::xyz)
      .def_readonly("xxz", &DmiTensor::xxz)
      .def_readonly("yxy", &DmiTensor::yxy)
      .def_readonly("yyz", &DmiTensor::yyz)
      .def_readonly("yxz", &DmiTensor::yxz)
      .def_readonly("zxy", &DmiTensor::zxy)
      .def_readonly("zyz", &DmiTensor::zyz)
      .def_readonly("zxz", &DmiTensor::zxz);
}
