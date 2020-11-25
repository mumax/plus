#include <memory>

#include "field.hpp"
#include "fieldquantity.hpp"
#include "variable.hpp"
#include "wrappers.hpp"

void wrap_variable(py::module& m) {
  py::class_<Variable, FieldQuantity>(m, "Variable")
      .def("get", [](const Variable* v) { return fieldToArray(v->field()); })
      .def("set", [](const Variable* v, py::array_t<real> data) {
        Field tmp(v->system(), v->ncomp());
        setArrayInField(tmp, data);
        v->set(std::move(tmp));
      });
}