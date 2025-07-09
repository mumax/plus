#include "antiferromagnet.hpp"
#include "butchertableau.hpp"
#include "dynamicequation.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "fieldops.hpp"
#include "mumaxworld.hpp"
#include "reduce.hpp"
#include "relaxer.hpp"
#include "thermalnoise.hpp"
#include "timesolver.hpp"
#include "torque.hpp"

#include <algorithm>

Relaxer::Relaxer(const Magnet* magnet, std::vector<real> RelaxTorqueThreshold, real tol)
    : magnets_({magnet}),
      timesolver_(magnet->world()->timesolver()),
      world_(magnet->mumaxWorld()),
      tol_(tol),
      threshold_(RelaxTorqueThreshold) {}

Relaxer::Relaxer(const MumaxWorld* world, real RelaxTorqueThreshold, real tol)
    : timesolver_(world->timesolver()),
      world_(world),
      tol_(tol) {
        for (const auto& pair : world->magnets()) {
          magnets_.push_back(pair.second);
          threshold_.push_back(RelaxTorqueThreshold);
        }
}

std::vector<DynamicEquation> Relaxer::getEquation(const Magnet* magnet) {
  /////////
  // TODO: this function looks too much like mumaxworld.resetTimeSolverEquations.
  /////////
    std::vector<DynamicEquation> eqs;
    if (const Ferromagnet* mag = magnet->asFM()) {
      DynamicEquation eq(
          mag->magnetization(),
          std::shared_ptr<FieldQuantity>(relaxTorqueQuantity(mag).clone()),
          std::shared_ptr<FieldQuantity>(thermalNoiseQuantity(mag).clone()));
      eqs.push_back(eq);
    }
    else if (auto host = magnet->asHost()) {
      for (const Ferromagnet* sub : host->sublattices()) {
          eqs.emplace_back(
              sub->magnetization(),
              std::shared_ptr<FieldQuantity>(relaxTorqueQuantity(sub).clone()),
              std::shared_ptr<FieldQuantity>(thermalNoiseQuantity(sub).clone()));
      }
    }
    else
      throw std::invalid_argument("Cannot relax quantity which is "
                                  "no Ferromagnet or (non-collinear) Antiferromagnet");
    return eqs;
}

std::vector<FM_FieldQuantity> Relaxer::getTorque() {
  std::vector<FM_FieldQuantity> torque;
  for (auto magnet : magnets_) {
    if (const Ferromagnet* mag = magnet->asFM())
     torque.push_back(relaxTorqueQuantity(mag));

    else if (auto host = magnet->asHost()) {
      for (const Ferromagnet* sub : host->sublattices())
        torque.push_back(relaxTorqueQuantity(sub));
    }
    else
      throw std::invalid_argument("Cannot relax quantity which is "
                                  "no Ferromagnet or (non-collinear) Antiferromagnet");
  }
  return torque;
}

real Relaxer::calcTorque(std::vector<FM_FieldQuantity> torque) {
  real t = 0;
  for (size_t i = 0; i < torque.size(); i++)
      t += dotSum(torque[i].eval(), torque[i].eval());
  return t;
}

real Relaxer::calcEnergy() {
  real E = 0;
  for (auto magnet : magnets_)
    E += evalTotalEnergy(magnet);
  return E;
}

void Relaxer::exec() {
  
  // Store current solver settings
  real time = timesolver_.time();
  real timestep = timesolver_.timestep();
  bool adaptive = timesolver_.hasAdaptiveTimeStep();
  real maxerr = timesolver_.maxError();
  std::string method = getRungeKuttaNameFromMethod(timesolver_.getRungeKuttaMethod());
  auto eqs = timesolver_.equations();

  // Set solver settings for relax
  timesolver_.enableAdaptiveTimeStep();
  timesolver_.setRungeKuttaMethod("Fehlberg");
  if (magnets_.size() == 1) { timesolver_.setEquations(getEquation(magnets_[0])); }
  else {world_->resetTimeSolverEquations(relaxTorqueQuantity);}

  // Run while monitoring energy
  const int N = 3; // evaluates energy every N steps (expenisve)  

  real E0 = calcEnergy();
  timesolver_.steps(N);
  real E1 = calcEnergy();

  while (E1 < E0) {
    timesolver_.steps(N);
    E0 = E1;
    E1 = calcEnergy();
  }
  // Run while monitoring torque
  // If threshold < 0 (default = -1): relax until torque is steady or increasing.
  if (std::all_of(threshold_.begin(), threshold_.end(), [](real t) { return t < 0; })) {

    std::vector<FM_FieldQuantity> torque = getTorque();
    real t0 = 0;
    real t1 = calcTorque(torque);
  
    real err = timesolver_.maxError();

    while (err > tol_) {
      err /= std::sqrt(2);
      timesolver_.setMaxError(err);

      timesolver_.steps(N);
      t0 = t1;
      t1 = calcTorque(torque);
      
      while (t1 < t0) {
        timesolver_.steps(N);
        t0 = t1;
        t1 = calcTorque(torque);
      }    
    }
  }

  else if (std::find(threshold_.begin(), threshold_.end(), 0) != threshold_.end())
    throw std::invalid_argument("The relax threshold should not be zero.");

  // If threshold is set by user: relax until torque is smaller than or equal to threshold.
  else {

    real err = timesolver_.maxError();
    std::vector<FM_FieldQuantity> torque = getTorque();

    while (err > tol_) {
      bool torqueConverged = true;
      for (size_t i = 0; i < torque.size(); i++) {
        if (maxVecNorm(torque[i].eval()) > threshold_[i]) {
          torqueConverged = false;
          break;
        }
      }

      if (torqueConverged) {    
        err /= std::sqrt(2);
        timesolver_.setMaxError(err);
      }
      timesolver_.steps(N);
    }
  }

  // Restore solver settings after relaxing
  timesolver_.setRungeKuttaMethod(method);
  timesolver_.setMaxError(maxerr);
  if (!adaptive) { timesolver_.disableAdaptiveTimeStep(); }
  timesolver_.setTime(time);
  timesolver_.setTimeStep(timestep); 
  timesolver_.setEquations(eqs);
}
