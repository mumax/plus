#include "examples.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <cmath>

#include "dynamicequation.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "grid.hpp"
#include "mumaxworld.hpp"
#include "timesolver.hpp"
#include "torque.hpp"

void skyrmion_excitation() {
    std::cout << "Skyrmion Excitation (Vairable Anisotropy)" << std::endl;

    real fmax = 50E9;  // maximum frequency(in Hz) of the sinc pulse
    real T = 2E-9;  // simulation time(longer->better frequency resolution) 
    real dt =
        1 / (2 * fmax);  // sample time(Nyquist theorem taken into account)
    real t0 = 1 / fmax;


    real length = 100E-9, width = 100E-9, thickness = 1E-9;
    int3 n = make_int3(32, 32, 1);
    real3 cellsize{length / n.x, width / n.y, thickness / n.z};
    std::string ferromagnet_name = "";

    MumaxWorld mWorld(cellsize);
    Grid mGrid(n);
    auto magnet = mWorld.addFerromagnet(mGrid, ferromagnet_name);

    magnet->msat.set(1E6);
    magnet->aex.set(15E-12);
    magnet->alpha.set(0.001);
    magnet->anisU.set(real3{0, 0, 1});
    magnet->idmi.set(3E-3);

    auto sinc = [](real x) -> real {
      if (std::abs(x) <= std::numeric_limits<real>::epsilon())
        return 1.0;
      else
        return sin(x) / (x);
    };

    auto Ku1t = [&sinc, fmax, T, t0](real t) -> real {
      return 1E6 * (1.0 + 0.01 * sinc(2 * M_PI * fmax * (t - t0)));
    };

    magnet->ku1.addTimeDependentTerm(Ku1t); // do we need a mask??

    magnet->minimize();

    // how to create a Neel skyrmion here??
    // m = neelskyrmion(-1, 1)

    real start = 0, stop = T;
    int n_timepoints = static_cast<int>(1 + (stop - start) / dt);

    std::string out_file_path = "./spin_wave_m.csv";
    std::ofstream magn_csv(out_file_path);
    magn_csv << "t,mx,my,mz,bx" << std::endl;

    for (int i = 0; i < n_timepoints; i++) {
      mWorld.timesolver().run(dt);
      auto m = magnet->magnetization()->average();
      magn_csv << mWorld.time() << "," << m[0] << "," << m[1] << "," << m[2]
               << "," << Ku1t(i * dt) << "," << std::endl;
    }
}
