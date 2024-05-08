#include "butchertableau.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "reduce.hpp"
#include "relaxer.hpp"
#include "timesolver.hpp"
#include "torque.hpp"

Relaxer::Relaxer(const Ferromagnet* magnet, real RelaxTorqueThreshold)
    : magnet_(magnet),
      threshold_(RelaxTorqueThreshold),
      torque_(relaxTorqueQuantity(magnet)) {}
      
void Relaxer::exec() {

  TimeSolver &timesolver = magnet_->world()->timesolver();
  
  // Store current solver settings
  real time = timesolver.time();
  real timestep = timesolver.timestep();
  bool adaptive = timesolver.hasAdaptiveTimeStep();
  real maxerr = timesolver.maxerror();
  bool prec = timesolver.hasPrecession();
  std::string method = getRungeKuttaNameFromMethod(timesolver.getRungeKuttaMethod());

  // Set solver settings for relax
  timesolver.disablePrecession();
  timesolver.enableAdaptiveTimeStep();
  timesolver.setRungeKuttaMethod("BogackiShampine");

  // Run while monitoring energy
  const int N = 3; // evaluates energy every N steps (expenisve)  

  real E0 = evalTotalEnergy(magnet_, true) + evalTotalEnergy(magnet_, false);
  timesolver.steps(N);
  real E1 = evalTotalEnergy(magnet_, true) + evalTotalEnergy(magnet_, false);
  while (E1 < E0) {
    timesolver.steps(N);
    E0 = E1;
    E1 = evalTotalEnergy(magnet_, true) + evalTotalEnergy(magnet_, false);
  }
  
  // Run while monitoring torque
  // If threshold = -1 (default): relax until torque is steady or increasing.
  if (threshold_ < 0) {
    real t0 = 0;
    real t1 = dotSum(torque_.eval(), torque_.eval());
    real err = timesolver.maxerror();
    while (err > 1e-9) {
      err /= std::sqrt(2);
      timesolver.setMaxError(err);

      timesolver.steps(N);
      t0 = t1;
      t1 = dotSum(torque_.eval(), torque_.eval());

      while (t1 < t0) {
        timesolver.steps(N);
        t0 = t1;
        t1 = dotSum(torque_.eval(), torque_.eval());
      }
    }
  }

  // If threshold is set by user: relax until torque is smaller than or equal to threshold.
  else { 
    real err = timesolver.maxerror();
    while (err > 1e-9) {
      while (maxVecNorm(torque_.eval())  > threshold_) {timesolver.steps(N);}
      err /= std::sqrt(2);
      timesolver.setMaxError(err);
    }
  }

  // Restore solver settings after relaxing
  timesolver.setRungeKuttaMethod(method);
  timesolver.setMaxError(maxerr);
  if (!adaptive) { timesolver.disableAdaptiveTimeStep(); }
  if (prec) { timesolver.enablePrecession(); }
  timesolver.setTime(time);
  timesolver.setTimeStep(timestep); 
}
