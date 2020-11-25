#pragma once

#include "butchertableau.hpp"
#include "stepper.hpp"

class TimeSolver;
class Field;
class RungeKuttaStageExecutor; // declared and defined in rungakutta.cpp
class Variable;

class RungeKuttaStepper : public Stepper {
 public:
  RungeKuttaStepper(TimeSolver*, RKmethod);
  int nStages() const;
  void step();

 private:
  ButcherTableau butcher_;

  friend class RungeKuttaStageExecutor;
};
