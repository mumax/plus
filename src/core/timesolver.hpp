#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "datatypes.hpp"
#include "dynamicequation.hpp"
#include "world.hpp"

class Stepper;
enum class RKmethod;

class TimeSolver {
  //------------- CONSTRUCTORS -------------------------------------------------

 public:
  class Factory {  // Can only be used by the constructor of the world
    static std::unique_ptr<TimeSolver> create();
    friend World::World(real3, Grid);
  };

 private:
  TimeSolver();

 public:
  ~TimeSolver();

  //------------- GET SOLVER SETTINGS ------------------------------------------

  const std::vector<DynamicEquation>& equations() const { return eqs_; }
  RKmethod getRungeKuttaMethod();
  real time() const { return time_; }
  real timestep() const { return timestep_; }
  bool hasAdaptiveTimeStep() const { return !fixedTimeStep_; }
  bool hasPrecession() const { return precession_; }
  real maxerror() const { return maxerror_; }
  

  //------------- SET SOLVER SETTINGS ------------------------------------------

  void setRungeKuttaMethod(RKmethod);
  void setRungeKuttaMethod(std::string);
  void setEquations(std::vector<DynamicEquation> eq);
  void setTime(real time) { time_ = time; }
  void setTimeStep(real dt) { timestep_ = dt; }
  void enableAdaptiveTimeStep() { fixedTimeStep_ = false; }
  void disableAdaptiveTimeStep() { fixedTimeStep_ = true; }
  void enablePrecession() { precession_ = true; }
  void disablePrecession() { precession_ = false; }
  void setMaxError(real maxerror) { maxerror_ = maxerror; }

  //------------- EXECUTING THE SOLVER -----------------------------------------

  void step();
  void steps(unsigned int nsteps);
  void runwhile(std::function<bool(void)>);
  void run(real duration);

  //------------- HELPER FUNCTIONS FOR ADAPTIVE TIMESTEPPING -------------------

  real sensibleTimeStep() const; /** Computes a sensible timestep */
  void adaptTimeStep(real corr);

 private:
  //------------- SOLVER SETTINGS ----------------------------------------------

  real maxerror_ = 1e-5;
  real time_ = 0.0;
  real timestep_ = 0.0;
  bool fixedTimeStep_ = false;
  bool precession_ = true; // LL precession
  std::vector<DynamicEquation> eqs_;

  //------------- THE INTERNAL STEPPER -----------------------------------------

  std::unique_ptr<Stepper> stepper_;
  RKmethod method_;
};
