#include <fstream>
#include <iostream>
#include <vector>

#include "dynamicequation.hpp"
#include "ferromagnet.hpp"
#include "grid.hpp"
#include "mumaxworld.hpp"
#include "timesolver.hpp"
#include "torque.hpp"

int main() {
  std::cout << "Standard Problem 4" << std::endl;

  real length = 500E-9, width = 125E-9, thickness = 3E-9;
  int3 n = make_int3(128, 32, 1);
  real3 cellsize{length / n.x, width / n.y, thickness / n.z};
  std::string ferromagnet_name = "";

  MumaxWorld mWorld(cellsize);
  Grid mGrid(n);
  auto magnet = mWorld.addFerromagnet(mGrid, ferromagnet_name);

  magnet->msat.set(800E3);
  magnet->aex.set(13E-12);
  magnet->alpha.set(0.02);
  magnet->magnetization()->set(real3{1, 0.1, 0});
  magnet->minimize();

  real3 B1{-24.6e-3, 4.3e-3, 0}, B2{-35.5e-3, -6.3e-3, 0};
  // choose B1 or B2 here
  mWorld.biasMagneticField = B1;

  // --- SCHEDULE THE OUTPUT ---
  int n_timepoints = 1000;
  real start = 0, stop = 1E-9, delta = (stop - start) / (n_timepoints - 1);

  std::string out_file_path = "./magnetization.csv";
  std::ofstream magn_csv(out_file_path);
  magn_csv << "t,mx,my,mz," << std::endl;

  for (int i = 0; i < n_timepoints; i++) {
    mWorld.timesolver()->run(delta);
    auto m = magnet->magnetization()->average();
    magn_csv << mWorld.timesolver()->time() << "," << m[0] << "," << m[1] << ","
             << m[2] << "," << std::endl;
  }

  std::cout << "Simulation results were saved into " << out_file_path
            << std::endl;
}
