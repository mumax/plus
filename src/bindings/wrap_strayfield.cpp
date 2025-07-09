#include <memory>
#include <stdexcept>

#include "magnet.hpp"
#include "field.hpp"
#include "fieldquantity.hpp"
#include "strayfield.hpp"
#include "wrappers.hpp"

void wrap_strayfield(py::module& m) {
  py::class_<StrayField, FieldQuantity>(m, "StrayField")
      .def(py::init([](Magnet* magnet, Grid grid) {
             return std::unique_ptr<StrayField>(new StrayField(magnet, grid));
           }),
           py::arg("magnet"), py::arg("grid"))
      .def("set_method", [](StrayField* strayField, std::string method) {
        if (method == "fft") {
          strayField->setMethod(StrayFieldExecutor::METHOD_FFT);
        } else if (method == "brute") {
          strayField->setMethod(StrayFieldExecutor::METHOD_BRUTE);
        } else {
          throw std::invalid_argument("Method should be \"fft\" or \"brute\"");
        }
      })
      .def_property("order", &StrayField::order, &StrayField::setOrder)
      .def_property("epsilon", &StrayField::eps, &StrayField::setEps)
      .def_property("switching_radius", &StrayField::switchingradius, &StrayField::setSwitchingradius);
}
