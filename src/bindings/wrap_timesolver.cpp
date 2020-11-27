#include <memory>
#include <vector>

#include "dynamicequation.hpp"
#include "ferromagnetquantity.hpp"
#include "field.hpp"
#include "fieldquantity.hpp"
#include "timesolver.hpp"
#include "variable.hpp"
#include "wrappers.hpp"

void wrap_timesolver(py::module& m) {
  py::class_<TimeSolver>(m, "TimeSolver")
      .def(py::init([](Variable* x, FM_FieldQuantity rhs) {
             return std::unique_ptr<TimeSolver>(new TimeSolver(DynamicEquation(
                 x, std::shared_ptr<FieldQuantity>(rhs.clone()))));
           }),
           py::arg("variable"), py::arg("rhs"))
      .def(py::init([](Variable* x, FM_FieldQuantity rhs,
                       FM_FieldQuantity noise) {
             return std::unique_ptr<TimeSolver>(new TimeSolver(DynamicEquation(
                 x, std::shared_ptr<FieldQuantity>(rhs.clone()),
                 std::shared_ptr<FieldQuantity>(noise.clone()))));
           }),
           py::arg("variable"), py::arg("rhs"), py::arg("noise"))
      // .def(py::init(
      //         [](std::vector<std::pair<Variable*, FieldQuantity*>> eqPairs) {
      //           std::vector<DynamicEquation> eqs;
      //           for (const auto& eqPair : eqPairs)
      //             eqs.emplace_back(eqPair.first, eqPair.second);
      //           return std::unique_ptr<TimeSolver>(new TimeSolver(eqs));
      //         }))
      .def_property_readonly("time", &TimeSolver::time)
      .def_property_readonly("sensible_timestep", &TimeSolver::sensibleTimeStep)
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
      .def("run", &TimeSolver::run);
}
