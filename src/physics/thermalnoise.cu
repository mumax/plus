#include <curand.h>

#include "constants.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "parameter.hpp"
#include "thermalnoise.hpp"
#include "world.hpp"

#if FP_PRECISION == SINGLE
const auto& generateRandNormal = curandGenerateNormal;
#elif FP_PRECISION == DOUBLE
const auto& generateRandNormal = curandGenerateNormalDouble;
#endif

bool thermalNoiseAssuredZero(const Ferromagnet* magnet) {
  return magnet->temperature.assuredZero();
}

__global__ void k_thermalNoise(CuField noiseField,
                               const CuParameter msat,
                               const CuParameter alpha,
                               const CuParameter temperature,
                               real preFactor) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!noiseField.cellInGrid(idx))
    return;

  real Ms = msat.valueAt(idx);
  real T = temperature.valueAt(idx);
  real a = alpha.valueAt(idx);
  real3 noise = noiseField.vectorAt(idx);

  noise *= sqrt(preFactor * a * T / ((1 + a * a) * Ms));

  noiseField.setVectorInCell(idx, noise);
}

Field evalThermalNoise(const Ferromagnet* magnet) {
  Field noise(magnet, 3);
  if (thermalNoiseAssuredZero(magnet)) {
    noise.makeZero();
    return noise;
  }

  int N = noise.grid().ncells();
  real mean = 0.0;
  real stddev = 1.0;
  for (int c = 0; c < 3; c++) {
    generateRandNormal(magnet->randomGenerator, noise.devptr(c), N, mean,
                       stddev);
  }

  auto msat = magnet->msat.cu();
  auto alpha = magnet->alpha.cu();
  auto temperature = magnet->temperature.cu();
  real cellVolume = magnet->world()->cellVolume();
  real preFactor = 2 * KB * GAMMALL / cellVolume;
  cudaLaunch(N, k_thermalNoise, noise.cu(), msat, alpha, temperature,
             preFactor);
  return noise;
}

FM_FieldQuantity thermalNoiseQuantity(const Ferromagnet* magnet) {
  return FM_FieldQuantity(magnet, evalThermalNoise, 3, "thermalNoise", "");
}
