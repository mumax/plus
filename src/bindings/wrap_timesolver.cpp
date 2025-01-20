#include <memory>
#include <string>
#include <vector>

#include "butchertableau.hpp"
#include "dynamicequation.hpp"
#include "quantityevaluator.hpp"
#include "field.hpp"
#include "fieldquantity.hpp"
#include "timesolver.hpp"
#include "variable.hpp"
#include "wrappers.hpp"
#include <pybind11/functional.h>  // for run_while

void wrap_timesolver(py::module& m) {
  py::class_<TimeSolver>(m, "TimeSolver")
      .def_property("time", &TimeSolver::time, &TimeSolver::setTime)
      .def_property_readonly("sensible_timestep", &TimeSolver::sensibleTimeStep)
      .def("set_method",
           [](TimeSolver& solver, std::string methodName) {
             RKmethod method = getRungeKuttaMethodFromName(methodName);
             solver.setRungeKuttaMethod(method);
           })
      .def_property("headroom", &TimeSolver::headroom, &TimeSolver::setHeadroom)
      .def_property("lower_bound", &TimeSolver::lowerBound, &TimeSolver::setLowerBound)
      .def_property("magnetization_max_error", &TimeSolver::magnetizationMaxError,
                                               &TimeSolver::setMagnetizationMaxError)
      .def_property("displacement_max_error", &TimeSolver::displacementMaxError,
                                              &TimeSolver::setDisplacementMaxError)
      .def_property("velocity_max_error", &TimeSolver::velocityMaxError,
                                          &TimeSolver::setVelocityMaxError)
      .def_property("sensible_factor", &TimeSolver::sensibleFactor, &TimeSolver::setSensibleFactor)
      .def_property("upper_bound", &TimeSolver::upperBound, &TimeSolver::setUpperBound)
      .def("step", &TimeSolver::step)
      .def("steps", &TimeSolver::steps)
      .def_property("timestep", &TimeSolver::timestep, &TimeSolver::setTimeStep)
      .def_property("adaptive_timestep", &TimeSolver::hasAdaptiveTimeStep,
                    [](TimeSolver& solver, bool adaptive) {
                      if (adaptive) {
                        solver.enableAdaptiveTimeStep();
                      } else {
                        solver.disableAdaptiveTimeStep();
                      }
                    })
      .def("run", &TimeSolver::run)
      .def("run_while", &TimeSolver::runwhile);
}
