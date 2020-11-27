#include "timesolver.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

#include "field.hpp"
#include "fieldquantity.hpp"
#include "reduce.hpp"
#include "rungekutta.hpp"
#include "stepper.hpp"

TimeSolver::TimeSolver()
    : TimeSolver::TimeSolver(std::vector<DynamicEquation>{}) {}

TimeSolver::TimeSolver(DynamicEquation eq)
    : TimeSolver::TimeSolver(std::vector<DynamicEquation>{eq}) {}

TimeSolver::TimeSolver(std::vector<DynamicEquation> eqs) {
  setEquations(eqs);  // This call sets the initial timestep
                      // by calling initializeTimeStep
  stepper_ = std::make_unique<RungeKuttaStepper>(this, FEHLBERG);
}

TimeSolver::~TimeSolver() {}

void TimeSolver::initializeTimeStep() {
  if (fixedTimeStep_ || eqs_.empty())
    return;

  real globalMaxNorm = 0;
  for (auto eq : eqs_)
    if (real maxNorm = maxVecNorm(eq.rhs->eval()); maxNorm > globalMaxNorm)
      globalMaxNorm = maxNorm;

  // initial guess for the timestep
  timestep_ = 0.01 / globalMaxNorm;
}

void TimeSolver::setEquations(std::vector<DynamicEquation> eqs) {
  eqs_ = eqs;
  initializeTimeStep();
}

void TimeSolver::adaptTimeStep(real correctionFactor) {
  if (fixedTimeStep_)
    return;

  real headroom = 0.8;

  if (std::isnan(correctionFactor))
    correctionFactor = 1.;

  correctionFactor *= headroom;
  correctionFactor = correctionFactor > 2.0 ? 2.0 : correctionFactor;
  correctionFactor = correctionFactor < 0.5 ? 0.5 : correctionFactor;

  timestep_ *= correctionFactor;
}

void TimeSolver::step() {
  if (timestep_ <= 0)
    std::runtime_error(
        "Timesolver can not make a step because the timestep is smaller than "
        "or equal to zero.");

  stepper_->step();
}

void TimeSolver::steps(unsigned int nSteps) {
  for (int i = 0; i < nSteps; i++)
    step();
}

void TimeSolver::runwhile(std::function<bool(void)> runcondition) {
  while (runcondition()) {
    step();
  }
}

void TimeSolver::run(real duration) {
  if (duration <= 0)
    return;

  real stoptime = time_ + duration;
  auto runcondition = [this, stoptime]() {
    return this->time() < stoptime - this->timestep();
  };
  runwhile(runcondition);

  // make final time step to end exactly at stoptime
  setTimeStep(stoptime - time_);
  step();
}
