#include <memory>

#include "dynamicequation.hpp"
#include "field.hpp"
#include "fieldquantity.hpp"
#include "timesolver.hpp"
#include "variable.hpp"
#include "wrappers.hpp"

void wrap_timesolver(py::module& m) {
  py::class_<TimeSolver>(m, "TimeSolver")
      .def(py::init([](Variable* x, FieldQuantity* rhs) {
             return std::unique_ptr<TimeSolver>(
                 new TimeSolver(DynamicEquation(x, rhs)));
           }),
           py::arg("variable"), py::arg("rhs"))
      .def_property_readonly("time", &TimeSolver::time)
      .def("step", &TimeSolver::step)
      .def("steps", &TimeSolver::steps)
      .def_property("timestep", &TimeSolver::timestep, &TimeSolver::setTimeStep)
      .def_property("adaptive_timestep", &TimeSolver::adaptiveTimeStep,
                    [](TimeSolver& solver, bool adaptive) {
                      if (adaptive) {
                        solver.enableAdaptiveTimeStep();
                      } else {
                        solver.disableAdaptiveTimeStep();
                      }
                    })
      .def("run", &TimeSolver::run);
}