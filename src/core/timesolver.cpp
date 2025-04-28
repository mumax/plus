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
  if (!fixedTimeStep_) timestep_ = sensibleTimeStep();
  method_ = method;
}

void TimeSolver::setRungeKuttaMethod(const std::string& method) {
  setRungeKuttaMethod(getRungeKuttaMethodFromName(method));
}

RKmethod TimeSolver::getRungeKuttaMethod() {
  return method_;
}

real TimeSolver::sensibleTimeStep() const {
  if (eqs_.empty())
    return 0.0;  // Timestep is irrelevant if there are no equations to solve
  real globalMaxNorm = 0;
  for (auto eq : eqs_)
    if (real maxNorm = maxVecNorm(eq.rhs->eval()); maxNorm > globalMaxNorm)
      globalMaxNorm = maxNorm;
  if (globalMaxNorm == 0) {
    throw std::runtime_error("Timesolver cannot be executed since the right hand "
                             "sides of all dynamic equations are zero.");
  }
  return sensibleFactor_ / globalMaxNorm;
}

void TimeSolver::setEquations(std::vector<DynamicEquation> eqs) {
  eqs_ = eqs;
  if (!fixedTimeStep_) timestep_ = sensibleTimeStep();
}

void TimeSolver::adaptTimeStep(real correctionFactor) {
  if (fixedTimeStep_)
    return;

  if (std::isnan(correctionFactor))
    correctionFactor = 1.;

  correctionFactor *= headroom_;
  if (lowerBound_ >= upperBound_) {
    throw std::runtime_error("The lower bound should be lower than the upper bound.");
  }
  correctionFactor = correctionFactor > upperBound_ ? upperBound_ : correctionFactor;
  correctionFactor = correctionFactor < lowerBound_ ? lowerBound_ : correctionFactor;

  timestep_ *= correctionFactor;
}

void TimeSolver::step() {
  if (timestep_ <= 0)
    throw std::runtime_error(
        "Timesolver can not make a step because the timestep is smaller than "
        "or equal to zero.");
  stepper_->step();
}

void TimeSolver::steps(unsigned int nSteps) {
  for (int i = 0; i < nSteps; i++) {
    step();
  }
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
  real oldTimestep = timestep();
  setTimeStep(stoptime - time_);
  step();
  if (fixedTimeStep_) setTimeStep(oldTimestep);
}
