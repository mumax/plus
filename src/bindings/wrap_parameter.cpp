#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <memory>

#include "field.hpp"
#include "fieldquantity.hpp"
#include "parameter.hpp"
#include "variable.hpp"
#include "wrappers.hpp"

void wrap_parameter(py::module& m) {
  py::class_<Parameter, FieldQuantity>(m, "Parameter")
      .def("add_time_term",
           py::overload_cast<const std::function<real(real)>&>(
               &Parameter::addTimeDependentTerm))
      .def("add_time_term",
           [](Parameter* p, std::function<real(real)>& term,
              py::array_t<real> mask) {
             Field field_mask(p->system(), 1);
             setArrayInField(field_mask, mask);
             p->addTimeDependentTerm(term, field_mask);
           })
      .def("eval",
           [](Parameter* p) {
             auto f = p->eval();
             return fieldToArray(f);
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
      .def(
          "add_time_term",
          [](VectorParameter* p, std::function<py::array_t<real>(real)>& term) {
            auto cpp_term = [term](real t) -> real3 {
              auto np_ndarray = term(t);
              auto buffer = np_ndarray.request();

              if (buffer.ndim != 1)
                throw std::invalid_argument("Number of dimensions must be one.");

              if (buffer.size != 3)
                throw std::invalid_argument(
                    "VectorPameter value should be of size 3, got " +
                    buffer.size);

              real* ptr = static_cast<real*>(buffer.ptr);

              return real3{ptr[0], ptr[1], ptr[2]};
            };

            p->addTimeDependentTerm(cpp_term);
          })
      .def("add_time_term",
           [](VectorParameter* p, std::function<py::array_t<real>(real)>& term,
              py::array_t<real> mask) {
             int ncomp = 3;
             Field field_mask(p->system(), ncomp);
             setArrayInField(field_mask, mask);

             auto cpp_term = [term](real t) -> real3 {
               auto np_ndarray = term(t);
               auto buffer = np_ndarray.request();

               if (buffer.ndim != 1)
                 throw std::invalid_argument("Number of dimensions must be one.");

               if (buffer.size != 3)
                 throw std::invalid_argument(
                     "VectorPameter value should be of size 3, got " +
                     buffer.size);

               real* ptr = static_cast<real*>(buffer.ptr);

               return real3{ptr[0], ptr[1], ptr[2]};
             };

             p->addTimeDependentTerm(cpp_term, field_mask);
           })
      .def("eval",
           [](VectorParameter* p) {
             auto f = p->eval();
             return fieldToArray(f);
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