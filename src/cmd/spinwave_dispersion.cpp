#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

#include "dynamicequation.hpp"
#include "examples.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "mumaxworld.hpp"
#include "timesolver.hpp"
#include "torque.hpp"

void spinwave_dispersion() {
  std::cout << "*********** Spinwave Dispersion ***********" << std::endl;

  // NUMERICAL PARAMETERS
  real fmax = 20E9;  // maximum frequency(in Hz) of the sinc pulse
  real T = 1E-8;     // simulation time(longer->better frequency resolution)
  real dt = 1 / (2.0 * fmax);  // the sample time
  real dx = 4E-9;              // cellsize
  int nx = 1024;               // number of cells

  // MATERIAL / SYSTEM PARAMETERS
  real Bz = 0.2;         // Bias field along the z direction
  real Aex = 13E-12;     // exchange constant
  real Ms = 800E3;       // saturation magnetization
  real alpha = 0.05;     // damping parameter
  real gamma = 1.76E11;  // gyromagnetic ratio

  int3 grid_size{nx, 1, 1};
  real3 cellsize{dx, dx, dx};
  std::string ferromagnet_name = "";

  MumaxWorld mWorld(cellsize);
  Grid mGrid(grid_size);
  GpuBuffer<bool> geometry;
  GpuBuffer<unsigned int> regions;
  auto magnet = mWorld.addFerromagnet(mGrid, geometry, regions, ferromagnet_name);

  magnet->enableDemag = false;
  magnet->msat.set(Ms);
  magnet->aex.set(Aex);
  magnet->alpha.set(alpha);
  magnet->magnetization()->set(real3{0, 0, 1});

  real3 B_ext{0, 0, Bz};
  mWorld.biasMagneticField = B_ext;

  // rectangular region, 2x1, defregion(1, rect(2 * {dx}, inf))
  int ncomp = 3;
  Field mask(magnet->system(), ncomp);
  // set the field to zero outside the given boundary
  std::vector<real> maskValues(nx * ncomp, 0);
  int nCellsDyn = 2;
  // the sinc signal will be located at the center of the simulation box
  for (int i = 0, start_pos = (nx / 2) - 1; i < ncomp; i++) {
    maskValues[i * nx + start_pos] = 1;
    maskValues[i * nx + start_pos + 1] = 1;
  }
  mask.setData(maskValues);

  auto sinc = [](real x) -> real {
    if (std::abs(x) <= std::numeric_limits<real>::epsilon())
      return 1.0;
    else
      return sin(x) / x;
  };

  auto Bt = [&sinc, fmax, T](real t) {
    return real3{0.01F * sinc(2 * M_PI * fmax * (t - T / 2)), 0, 0};
  };

  magnet->biasMagneticField.addTimeDependentTerm(Bt);

  magnet->minimize();

  // --- SCHEDULE THE OUTPUT ---
  real start = 0, stop = T;
  int n_timepoints = static_cast<int>(1 + (stop - start) / dt);

  std::string out_file_path = "spinwave_dispersion_m.csv";
  std::ofstream magn_csv(out_file_path);
  magn_csv.precision(10);
  magn_csv << "t,mx,my,mz,bx" << std::endl;

  std::cout << "Running simulation..." << std::endl;

  for (int i = 0; i < n_timepoints; i++) {
    mWorld.timesolver().run(dt);
    auto m = magnet->magnetization()->average();
    magn_csv << mWorld.time() << "," << m[0] << "," << m[1] << "," << m[2]
             << "," << Bt(i * dt).x << "," << std::endl;
  }

  std::cout << "Simulation results were saved into\n"
            << std::filesystem::current_path().append(out_file_path).string()
            << std::endl;
}
