#pragma once

#include "ferromagnetquantity.hpp"
#include "timesolver.hpp"

class Ferromagnet;
//class TimeSolver;

// Relax the system to a minimum energy state by first minimizing the total energy
// and then minimizing the total torque.

class Relaxer {
 public:
  Relaxer(const Ferromagnet*, real RelaxTorqueThreshold);
  
  //void getWorldTimesolver(TimeSolver&);

  void exec();



 private:
  const Ferromagnet* magnet_;
  real threshold_;
  FM_FieldQuantity torque_;
};