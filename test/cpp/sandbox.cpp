#include <iostream>

#include "datatypes.hpp"
#include "dynamicequation.hpp"
#include "ferromagnet.hpp"
#include "grid.hpp"
#include "timer.hpp"
#include "timesolver.hpp"
#include "world.hpp"

int main() {
  World world({1.0, 1.0, 1.0});
  Ferromagnet* magnet = world.addFerromagnet(Grid({128, 64, 1}),"my_magnet");
  DynamicEquation llg(magnet->magnetization(), magnet->torque());
  TimeSolver solver(llg);

  solver.steps(1000);

  timer.printTimings();

  return 0;
}