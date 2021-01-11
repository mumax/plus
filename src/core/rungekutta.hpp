#pragma once

#include <optional>
#include <vector>

#include "butchertableau.hpp"
#include "dynamicequation.hpp"
#include "field.hpp"
#include "stepper.hpp"

class TimeSolver;
class Variable;

class RungeKuttaStepper : public Stepper {
 public:
  RungeKuttaStepper(TimeSolver*, RKmethod);
  int nStages() const;
  void step();

 private:
  ButcherTableau butcher_;
  class RungeKuttaStageExecutor;
};

class RungeKuttaStepper::RungeKuttaStageExecutor {
 public:
  RungeKuttaStageExecutor(DynamicEquation, const RungeKuttaStepper&);

  void setStageK(int);
  void setStageX(int);
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
