#include <memory>

#include "field.hpp"
#include "fieldquantity.hpp"
#include "parameter.hpp"
#include "variable.hpp"
#include "wrappers.hpp"

void wrap_parameter(py::module& m) {
  py::class_<Parameter, FieldQuantity>(m, "Parameter")
      .def("is_uniform", &Parameter::isUniform)
      .def("set", [](Parameter* p, real value) { p->set(value); })
      .def("set", [](Parameter* p, py::array_t<real> data) {
        Field tmp(p->system(), 1);
        setArrayInField(tmp, data);
        p->set(std::move(tmp));
      });

  py::class_<VectorParameter, FieldQuantity>(m, "VectorParameter")
      .def("is_uniform", &VectorParameter::isUniform)
      .def("set", [](VectorParameter* p, real3 value) { p->set(value); })
      .def("set", [](VectorParameter* p, py::array_t<real> data) {
        Field tmp(p->system(), 3);
        setArrayInField(tmp, data);
        p->set(std::move(tmp));
      });
}
