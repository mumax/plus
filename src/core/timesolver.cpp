#include "timesolver.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

#include "field.hpp"
#include "fieldquantity.hpp"
#include "reduce.hpp"
#include "rungekutta.hpp"
#include "stepper.hpp"

std::unique_ptr<TimeSolver> TimeSolver::Factory::create() {
  return std::unique_ptr<TimeSolver>(new TimeSolver());
}

TimeSolver::TimeSolver() {
  setRungeKuttaMethod(RKmethod::FEHLBERG);
}

TimeSolver::~TimeSolver() {}

void TimeSolver::setRungeKuttaMethod(RKmethod method) {
  stepper_ = std::make_unique<RungeKuttaStepper>(this, method);
  timestep_ = sensibleTimeStep();
}

real TimeSolver::sensibleTimeStep() const {
  if (eqs_.empty())
    return 0.0;  // Timestep is irrelevant if there are no equations to solve

  real globalMaxNorm = 0;
  for (auto eq : eqs_)
    if (real maxNorm = maxVecNorm(eq.rhs->eval()); maxNorm > globalMaxNorm)
      globalMaxNorm = maxNorm;

  return 0.01 / globalMaxNorm;
}

void TimeSolver::setEquations(std::vector<DynamicEquation> eqs) {
  eqs_ = eqs;
  timestep_ = sensibleTimeStep();
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
