#include "stepper.hpp"

void Stepper::setParentTimeSolver(TimeSolver* solver) {
  solver_ = solver;
}
