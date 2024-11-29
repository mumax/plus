#include <memory>

#include "quantityevaluator.hpp"
#include "field.hpp"
#include "fieldquantity.hpp"
#include "wrappers.hpp"

void wrap_fieldquantity(py::module& m) {
  py::class_<FieldQuantity>(m, "FieldQuantity")
      .def_property_readonly("name", &FieldQuantity::name)
      .def_property_readonly("unit", &FieldQuantity::unit)
      .def_property_readonly("ncomp", &FieldQuantity::ncomp)
      .def_property_readonly("grid", &FieldQuantity::grid)
      .def_property_readonly("system", &FieldQuantity::system)
      .def("eval",
           [](const FieldQuantity* q) { return fieldToArray(q->eval()); })
      // exec does the same as eval but without returning the result (useful for
      // benchmarking)
      .def("exec", [](const FieldQuantity* q) { q->eval(); })
      .def("average", &FieldQuantity::average)
      .def("get_rgb", &FieldQuantity::getRGB);
}

void wrap_magnetfieldquantity(py::module& m) {
  py::class_<M_FieldQuantity, FieldQuantity>(m, "MagnetFieldQuantity");
}

void wrap_ferromagnetfieldquantity(py::module& m) {
  py::class_<FM_FieldQuantity, FieldQuantity>(m, "FerromagnetFieldQuantity");
}

void wrap_antiferromagnetfieldquantity(py::module& m) {
  py::class_<AFM_FieldQuantity, FieldQuantity>(m, "AntiferromagnetFieldQuantity");
}