#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <memory>

#include "field.hpp"
#include "fieldquantity.hpp"
#include "inter_parameter.hpp"
#include "parameter.hpp"
#include "variable.hpp"
#include "wrappers.hpp"

void wrap_parameter(py::module& m) {
  // ===== Parameter =====
  py::class_<Parameter, FieldQuantity>(m, "Parameter")
      .def("add_time_term", py::overload_cast<const std::function<real(real)>&>(
                                &Parameter::addTimeDependentTerm))
      .def("add_time_term",
           [](Parameter* p, std::function<real(real)>& term,
              py::array_t<real> mask) {
             Field field_mask(p->system(), 1);
             setArrayInField(field_mask, mask);
             p->addTimeDependentTerm(term, field_mask);
           })
      .def_property_readonly("is_uniform", &Parameter::isUniform)
      .def_property_readonly("is_dynamic",
                             [](Parameter* p) { return p->isDynamic(); })
      .def_property("uniform_value", &Parameter::getUniformValue,
                    static_cast<void (Parameter::*)(real)>(&Parameter::set))
      .def("remove_time_terms", &Parameter::removeAllTimeDependentTerms)
      .def("set", [](Parameter* p, real value) { p->set(value); })
      .def("set", [](Parameter* p, py::array_t<real> data) {
        Field tmp(p->system(), 1);
        setArrayInField(tmp, data);
        p->set(std::move(tmp));
      })
      .def("set_in_region", [](Parameter* p, unsigned int regionIdx, real value)
                                { p->setInRegion(regionIdx, value);
      });

  // ===== VectorParameter =====
  py::class_<VectorParameter, FieldQuantity>(m, "VectorParameter")
      .def(
          "add_time_term",
          [](VectorParameter* p, std::function<py::array_t<real>(real)>& term) {
            auto cpp_term = [term](real t) -> real3 {
              auto np_ndarray = term(t);
              auto buffer = np_ndarray.request();

              if (buffer.ndim != 1)
                throw std::invalid_argument(
                    "Number of dimensions must be one.");

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
                 throw std::invalid_argument(
                     "Number of dimensions must be one.");

               if (buffer.size != 3)
                 throw std::invalid_argument(
                     "VectorPameter value should be of size 3, got " +
                     buffer.size);

               real* ptr = static_cast<real*>(buffer.ptr);

               return real3{ptr[0], ptr[1], ptr[2]};
             };

             p->addTimeDependentTerm(cpp_term, field_mask);
           })
      .def_property_readonly("is_uniform", &VectorParameter::isUniform)
      .def_property_readonly("is_dynamic",
                             [](VectorParameter* p) { return p->isDynamic(); })
      .def_property("uniform_value", &VectorParameter::getUniformValue,
           static_cast<void (VectorParameter::*)(real3)>(&VectorParameter::set))
      .def("remove_time_terms", &VectorParameter::removeAllTimeDependentTerms)
      .def("set", [](VectorParameter* p, real3 value) { p->set(value); })
      .def("set", [](VectorParameter* p, py::array_t<real> data) {
        Field tmp(p->system(), 3);
        setArrayInField(tmp, data);
        p->set(std::move(tmp));
      })
      .def("set_in_region", [](VectorParameter* p, unsigned int regionIdx, real3 value)
                                { p->setInRegion(regionIdx, value);
      });

  // ===== InterParameter =====
  py::class_<InterParameter>(m, "InterParameter")
      .def_property_readonly("name", &InterParameter::name)
      .def_property_readonly("unit", &InterParameter::unit)
      .def_property_readonly("ncomp", &InterParameter::ncomp)
      .def_property_readonly("number_of_regions", &InterParameter::numberOfRegions)
      .def_property_readonly("unique_regions", &InterParameter::uniqueRegions)
      .def_property_readonly("is_uniform", &InterParameter::isUniform)
      .def_property("uniform_value", &InterParameter::getUniformValue,
                                     &InterParameter::set)

      .def("set", &InterParameter::set, py::arg("value"))
      .def("set_between", &InterParameter::setBetween,
           py::arg("i"), py::arg("j"), py::arg("value"))
      .def("get_between", &InterParameter::getBetween, py::arg("i"), py::arg("j"))
      .def("eval", &InterParameter::eval)
      ;
}
