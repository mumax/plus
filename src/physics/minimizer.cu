#include "antiferromagnet.hpp"
#include "cudalaunch.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "minimizer.hpp"
#include "mumaxworld.hpp"
#include "ncafm.hpp"
#include "reduce.hpp"
#include "torque.hpp"

Minimizer::Minimizer(const Ferromagnet* magnet,
                     real stopMaxMagDiff,
                     int nMagDiffSamples)
    : magnets_({magnet}),
      torques_({relaxTorqueQuantity(magnet)}),
      nMagDiffSamples_(nMagDiffSamples),
      stopMaxMagDiff_(stopMaxMagDiff),
      t0(1),
      t1(1),
      m0(1),
      m1(1) {
  stepsizes_ = {1e-14};  // TODO: figure out how to make descent guess
  // TODO: check if input arguments are sane
}

Minimizer::Minimizer(const HostMagnet* magnet,
                     real stopMaxMagDiff,
                     int nMagDiffSamples)
    : magnets_(magnet->sublattices()),
      nMagDiffSamples_(nMagDiffSamples),
      stopMaxMagDiff_(stopMaxMagDiff),
      t0(magnets_.size()),
      t1(magnets_.size()),
      m0(magnets_.size()),
      m1(magnets_.size()) {
  stepsizes_.assign(magnets_.size(), 1e-14);
  for (auto sub : magnets_)
    torques_.push_back(relaxTorqueQuantity(sub));
}

Minimizer::Minimizer(const MumaxWorld* world,
                     real stopMaxMagDiff,
                     int nMagDiffSamples)
    : stopMaxMagDiff_(stopMaxMagDiff) {
  // Total number of ferromagnets (FM instances or sublattices)
  size_t N = world->ferromagnets().size() + 2 * world->antiferromagnets().size();
  
  nMagDiffSamples_ = nMagDiffSamples * N;

  t0.resize(N);
  t1.resize(N);
  m0.resize(N);
  m1.resize(N);

  for (const auto pair : world->magnets()) {
    if (auto host = pair.second->asHost()) {
      for (auto sub : host->sublattices())
        magnets_.push_back(sub);
    }
    else if (const Ferromagnet* mag = pair.second->asFM())
      magnets_.push_back(mag);
  }

  for (auto magnet : magnets_)
    torques_.push_back(relaxTorqueQuantity(magnet));
    
  stepsizes_.assign(N, 1e-14);
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
  for (size_t i = 0; i < magnets_.size(); i++) {

    m0[i] = magnets_[i]->magnetization()->eval();

    if (nsteps_ == 0)
      t0[i] = torques_[i].eval();
    else
      t0[i] = t1[i];

    m1[i] = Field(magnets_[i]->system(), 3);
    int ncells = m1[i].grid().ncells();

    cudaLaunch(ncells, k_step, m1[i].cu(), m0[i].cu(), t0[i].cu(), stepsizes_[i]);
  }
  
  for (size_t i = 0; i < magnets_.size(); i++)
    magnets_[i]->magnetization()->set(m1[i]);  // normalizes
    
  for (size_t i = 0; i < magnets_.size(); i++)
    t1[i] = torques_[i].eval();

  for (size_t i = 0; i < magnets_.size(); i++) {
    Field dm = add(real(+1), m1[i], real(-1), m0[i]);
    Field dt = add(real(-1), t1[i], real(+1), t0[i]);  // TODO: check sign difference

    stepsizes_[i] = BarzilianBorweinStepSize(dm, dt, nsteps_);

    addMagDiff(maxVecNorm(dm));
  }
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
