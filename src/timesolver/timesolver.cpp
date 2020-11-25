#include "timesolver.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>

#include "field.hpp"
#include "fieldquantity.hpp"
#include "reduce.hpp"
#include "rungekutta.hpp"
#include "stepper.hpp"

TimeSolver::TimeSolver(DynamicEquation eq)
    : time_(0), maxerror_(1e-5), fixedTimeStep_(false) {
  eqs_.push_back(eq);
  stepper_ = new RungeKuttaStepper(this, FEHLBERG);

  // initial guess for the timestep
  timestep_ = 0.01 / maxVecNorm(eqs_[0].rhs->eval());
}

TimeSolver::TimeSolver(std::vector<DynamicEquation> eqs)
    : time_(0), maxerror_(1e-5), eqs_(eqs), fixedTimeStep_(false) {
  stepper_ = new RungeKuttaStepper(this, FEHLBERG);

  real globalMaxNorm = 0;
  for (auto eq : eqs_)
    if (real maxNorm = maxVecNorm(eq.rhs->eval()); maxNorm > globalMaxNorm)
      globalMaxNorm = maxNorm;

  // initial guess for the timestep
  timestep_ = 0.01 / globalMaxNorm;
}

TimeSolver::~TimeSolver() {
  if (stepper_)
    delete stepper_;
}

const real& TimeSolver::time() const {
  return time_;
}

DynamicEquation TimeSolver::equation(int idx) const {
  return eqs_.at(idx);
}

int TimeSolver::nEquations() const {
  return eqs_.size();
}

const real& TimeSolver::timestep() const {
  return timestep_;
}

real TimeSolver::maxerror() const {
  return maxerror_;
}

void TimeSolver::setTime(real time) {
  time_ = time;
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

void TimeSolver::setTimeStep(real dt) {
  if (dt <= 0.0)
    std::invalid_argument("Time step has to be larger than zero");
  timestep_ = dt;
}

void TimeSolver::enableAdaptiveTimeStep() {
  fixedTimeStep_ = false;
}

void TimeSolver::disableAdaptiveTimeStep() {
  fixedTimeStep_ = true;
}

bool TimeSolver::adaptiveTimeStep() const {
  return !fixedTimeStep_;
}

void TimeSolver::step() {
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