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
    friend class World;
  };

 private:
  TimeSolver();

 public:
  ~TimeSolver();

  //------------- GET SOLVER SETTINGS ------------------------------------------

  const std::vector<DynamicEquation>& equations() const { return eqs_; }
  RKmethod getRungeKuttaMethod();
  real headroom() const { return headroom_; }
  real lowerBound() const { return lowerBound_; }
  real maxError() const { return maxError_; }
  real sensibleFactor() const { return sensibleFactor_; }
  real sensibleTimestepDefault() const { return sensibleTimestepDefault_; }
  real time() const { return time_; }
  real timestep() const { return timestep_; }
  real upperBound() const { return upperBound_; }
  bool hasAdaptiveTimeStep() const { return !fixedTimeStep_; }

  //------------- SET SOLVER SETTINGS ------------------------------------------

  void setRungeKuttaMethod(RKmethod);
  void setRungeKuttaMethod(const std::string& method);
  void setEquations(std::vector<DynamicEquation> eq);
  void setHeadroom(real headroom) { headroom_ = headroom; }
  void setLowerBound(real lowerBound) { lowerBound_ = lowerBound; }
  void setMaxError(real maxError) { maxError_ = maxError; }
  void setSensibleFactor(real factor) { sensibleFactor_ = factor; }
  void setSensibleTimestepDefault(real dt);
  void setTime(real time) { time_ = time; }
  void setTimeStep(real dt) { timestep_ = dt; }
  void setUpperBound(real upperBound) { upperBound_ = upperBound; }
  void enableAdaptiveTimeStep() { fixedTimeStep_ = false; }
  void disableAdaptiveTimeStep() { fixedTimeStep_ = true; }

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

  real headroom_ = 0.8;
  real lowerBound_ = 0.5;
  real maxError_ = 1e-5;
  real sensibleFactor_ = 0.01;
  real sensibleTimestepDefault_ = 1e-14;
  real time_ = 0.0;
  real timestep_ = 0.0;
  real upperBound_ = 2.0;
  bool fixedTimeStep_ = false;
  std::vector<DynamicEquation> eqs_;

  //------------- THE INTERNAL STEPPER -----------------------------------------

  std::unique_ptr<Stepper> stepper_;
  RKmethod method_;
};
