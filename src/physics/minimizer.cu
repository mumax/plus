#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "minimizer.hpp"
#include "reduce.hpp"
#include "torque.hpp"

Minimizer::Minimizer(const Ferromagnet* magnet,
                     real stopMaxMagDiff,
                     int nMagDiffSamples)
    : magnet_(magnet),
      torque_(relaxTorqueQuantity(magnet)),
      nMagDiffSamples_(nMagDiffSamples),
      stopMaxMagDiff_(stopMaxMagDiff) {
  stepsize_ = 1e-14;  // TODO: figure out how to make descent guess

  // TODO: check if input arguments are sane
}

void Minimizer::exec() {
  nsteps_ = 0;
  lastMagDiffs_.clear();

  while (!converged())
    step();
}

__global__ void k_step(CuField mField,
                       const CuField m0Field,
                       const CuField torqueField,
                       real dt) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!mField.cellInGrid(idx))
    return;

  real3 m0 = m0Field.vectorAt(idx);
  real3 t = torqueField.vectorAt(idx);

  real t2 = dt * dt * dot(t, t);
  real3 m = ((4 - t2) * m0 + 4 * dt * t) / (4 + t2);

  mField.setVectorInCell(idx, m);
}

static inline real BarzilianBorweinStepSize(Field& dm, Field& dtorque, int n) {
  real nom, div;
  if (n % 2 == 0) {
    nom = dotSum(dm, dm);
    div = dotSum(dm, dtorque);
  } else {
    nom = dotSum(dm, dtorque);
    div = dotSum(dtorque, dtorque);
  }
  if (div == 0.0)
    return 1e-14;  // TODO: figure out safe stepsize

  return nom / div;
}

void Minimizer::step() {
  m0 = magnet_->magnetization()->eval();

  if (nsteps_ == 0)
    t0 = torque_.eval();
  else
    t0 = t1;

  m1 = Field(magnet_, 3);
  int N = m1.grid().ncells();
  cudaLaunch(N, k_step, m1.cu(), m0.cu(), t0.cu(), stepsize_);

  magnet_->magnetization()->set(m1);  // normalizes

  t1 = torque_.eval();

  Field dm = add(real(+1), m1, real(-1), m0);
  Field dt = add(real(-1), t1, real(+1), t0);  // TODO: check sign difference

  stepsize_ = BarzilianBorweinStepSize(dm, dt, nsteps_);

  addMagDiff(maxVecNorm(dm));

  nsteps_ += 1;
}

bool Minimizer::converged() const {
  if (lastMagDiffs_.size() < nMagDiffSamples_)
    return false;

  for (auto dm : lastMagDiffs_)
    if (dm > stopMaxMagDiff_)
      return false;

  return true;
}

void Minimizer::addMagDiff(real dm) {
  lastMagDiffs_.push_back(dm);
  if (lastMagDiffs_.size() > nMagDiffSamples_)
    lastMagDiffs_.pop_front();
}
