#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "datatypes.hpp"
#include "dynamicequation.hpp"

class Stepper;

class TimeSolver {
 public:
  //------------- CONSTRUCTORS -------------------------------------------------

  TimeSolver();
  explicit TimeSolver(DynamicEquation eq);
  explicit TimeSolver(std::vector<DynamicEquation> eqs);
  ~TimeSolver();

  //------------- GET SOLVER SETTINGS ------------------------------------------

  const std::vector<DynamicEquation>& equations() const { return eqs_; }
  const real& time() const { return time_; }
  const real& timestep() const { return timestep_; }
  bool hasAdaptiveTimeStep() const { return !fixedTimeStep_; }
  real maxerror() const { return maxerror_; }

  //------------- SET SOLVER SETTINGS ------------------------------------------

  void setEquations(std::vector<DynamicEquation> eq);
  void setTime(real time) { time_ = time; }
  void setTimeStep(real dt) { timestep_ = dt; }
  void enableAdaptiveTimeStep() { fixedTimeStep_ = false; }
  void disableAdaptiveTimeStep() { fixedTimeStep_ = true; }
  void setMaxError(real maxerror) { maxerror_ = maxerror; }

  //------------- EXECUTING THE SOLVER -----------------------------------------

  void step();
  void steps(unsigned int nsteps);
  void runwhile(std::function<bool(void)>);
  void run(real duration);

  //------------- HELPER FUNCTIONS FOR ADAPTIVE TIMESTEPPING -------------------

  void initializeTimeStep(); /** Make smart guess for initial timestep */
  void adaptTimeStep(real corr);

 private:
  //------------- SOLVER SETTINGS ----------------------------------------------

  real maxerror_ = 1e-5;
  real time_ = 0.0;
  real timestep_ = 0.0;
  bool fixedTimeStep_ = false;
  std::vector<DynamicEquation> eqs_;

  //------------- THE INTERNAL STEPPER -----------------------------------------

  std::unique_ptr<Stepper> stepper_;
};
