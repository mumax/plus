#pragma once

#include "quantityevaluator.hpp"
#include "timesolver.hpp"

class DynamicEquation;
class Magnet;
class MumaxWorld;
class TimeSolver;
// Relax the system to a minimum energy state by first minimizing the total energy
// and then minimizing the total torque.

class Relaxer {
 public:
  Relaxer(const Magnet*, std::vector<real> RelaxTorqueThreshold, real tol);
  Relaxer(const MumaxWorld*, real RelaxTorqueThreshold, real tol);
  
  // Execute relaxing procedure
  void exec();

  // Helper functions
 private:
  std::vector<DynamicEquation> getEquation(const Magnet*);
  std::vector<FM_FieldQuantity> getTorque();
  real calcTorque(std::vector<FM_FieldQuantity>);
  real calcEnergy();

 private:
  std::vector<const Magnet*> magnets_;
  std::vector<real> threshold_;
  TimeSolver &timesolver_;
  const MumaxWorld* world_;
  real tol_;
  real maxerr_;
};