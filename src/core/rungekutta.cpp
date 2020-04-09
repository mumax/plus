#include "rungekutta.hpp"

#include <vector>

#include "butchertableau.hpp"
#include "dynamicequation.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "quantity.hpp"
#include "timesolver.hpp"
#include "variable.hpp"

RungeKuttaStepper::RungeKuttaStepper(TimeSolver* solver, RKmethod method)
    : butcher_(constructTableau(method)) {
  setParentTimeSolver(solver);

  k_.reserve(butcher_.nStages);
  for (int stage = 0; stage < butcher_.nStages; stage++) {
    std::unique_ptr<Field> k(
        new Field(solver_->eq().grid(), solver_->eq().ncomp()));
    k_.push_back(std::move(k));
  }
}

void RungeKuttaStepper::step() {
  DynamicEquation eq = solver_->eq();

  real dt = solver_->timestep();
  real t0 = solver_->time();

  std::unique_ptr<Field> x0(new Field(eq.grid(), eq.ncomp()));
  std::unique_ptr<Field> xstage(new Field(eq.grid(), eq.ncomp()));

  x0->copyFrom(eq.x->field());

  // Apply the stages
  for (int stage = 0; stage < butcher_.nStages; stage++) {
    if (stage > 0) {
      std::vector<real> weights(1 + stage);
      std::vector<const Field*> fields(1 + stage);
      weights[0] = 1;
      fields[0] = x0.get();
      for (int i = 0; i < stage; i++) {
        weights[i + 1] = dt * butcher_.rkMatrix[stage][i];
        fields[i + 1] = k_[i].get();
      }
      add(xstage.get(), fields, weights);
      eq.x->set(xstage.get());
      solver_->setTime(t0 + dt * butcher_.nodes[stage]);
    }

    eq.rhs->evalIn(k_[stage].get());
  }

  // Make the actual step
  std::vector<real> weights(1 + butcher_.nStages);
  std::vector<const Field*> fields(1 + butcher_.nStages);
  weights[0] = 1;
  fields[0] = x0.get();
  for (int i = 0; i < butcher_.nStages; i++) {
    weights[i + 1] = dt * butcher_.weights1[i];
    fields[i + 1] = k_[i].get();
  }
  add(xstage.get(), fields, weights);
  eq.x->set(xstage.get());
  solver_->setTime(t0 + dt);
}