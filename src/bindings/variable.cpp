#include "variable.hpp"

#include <memory>

#include "field.hpp"
#include "wrappers.hpp"
#include "quantity.hpp"

void wrap_variable(py::module& m) {
  py::class_<Variable, Quantity>(m, "Variable")
      .def("get", [](const Variable* v) { return fieldToArray(v->field()); })
      .def("set", [](const Variable* v, py::array_t<real> data) {
        std::unique_ptr<Field> tmp(new Field(v->grid(), v->ncomp()));
        setArrayInField(tmp.get(), data);
        v->set(tmp.get());
      });
}