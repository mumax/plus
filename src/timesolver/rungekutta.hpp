#pragma once

#include "butchertableau.hpp"
#include "stepper.hpp"

class TimeSolver;
class Field;
// class RungeKuttaStageExecutor;  // declared and defined in rungakutta.cpp
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


class RungeKuttaStageExecutor {
 public:
  RungeKuttaStageExecutor(DynamicEquation eq,  const RungeKuttaStepper& stepper);

  void setStageK(int stage);
  void setStageX(int stage);
  void setFinalX();
  void resetX();
  real getError() const;

 private:
  Field x0;
  const ButcherTableau& butcher;
  const RungeKuttaStepper& stepper;
  const Variable& x;  // TODO: make this non constant
  std::optional<Field> noise;
  std::vector<Field> k;
  DynamicEquation eq_;
};
