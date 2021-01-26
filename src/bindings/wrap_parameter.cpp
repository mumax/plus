#include <memory>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "field.hpp"
#include "fieldquantity.hpp"
#include "parameter.hpp"
#include "variable.hpp"
#include "wrappers.hpp"

void wrap_parameter(py::module& m) {
  py::class_<Parameter, FieldQuantity>(m, "Parameter")
      .def("add_time_terms", py::overload_cast<const std::function<real(real)>&>(&Parameter::addTimeDependentTerm))
      .def("add_time_terms", [](Parameter* p,
                                std::function<real(real)>& term,
                                py::array_t<real> mask) {
          Field field_mask(p->system(), 1);
          setArrayInField(field_mask, mask);
          p->addTimeDependentTerm(term, field_mask);
        })
      .def("is_uniform", &Parameter::isUniform)
      .def("is_dynamic", &Parameter::isDynamic)
      .def("remove_time_terms", &Parameter::removeAllTimeDependentTerms)
      .def("set", [](Parameter* p, real value) { p->set(value); })
      .def("set", [](Parameter* p, py::array_t<real> data) {
        Field tmp(p->system(), 1);
        setArrayInField(tmp, data);
        p->set(std::move(tmp));
      });

  py::class_<VectorParameter, FieldQuantity>(m, "VectorParameter")
      .def("add_time_terms", py::overload_cast<const std::function<real3(real)>&>(&VectorParameter::addTimeDependentTerm))
      .def("add_time_terms", [](VectorParameter* p,
                                std::function<real3(real)>& term,
                                py::array_t<real> mask) {
        Field field_mask(p->system(), 1);
        setArrayInField(field_mask, mask);
        p->addTimeDependentTerm(term, field_mask);
      })
      .def("is_uniform", &VectorParameter::isUniform)
      .def("is_dynamic", &VectorParameter::isDynamic)
      .def("remove_time_terms", &VectorParameter::removeAllTimeDependentTerms)
      .def("set", [](VectorParameter* p, real3 value) { p->set(value); })
      .def("set", [](VectorParameter* p, py::array_t<real> data) {
        Field tmp(p->system(), 3);
        setArrayInField(tmp, data);
        p->set(std::move(tmp));
      });
}
