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
  stepsize_ = 1e-14;  // TODO: figure out how to make descent guess
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
  stepsize_ = 1e-14;
  for (auto sub : magnets_)
    torques_.push_back(relaxTorqueQuantity(sub));
}

Minimizer::Minimizer(const MumaxWorld* world,
                     real stopMaxMagDiff,
                     int nMagDiffSamples)
    : stopMaxMagDiff_(stopMaxMagDiff) {
  // Find all ferromagnets (FM instances or sublattices)
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

  nMagDiffSamples_ = nMagDiffSamples;
  stepsize_ = 1e-14;

  size_t N = magnets_.size();
  t0.resize(N);
  t1.resize(N);
  m0.resize(N);
  m1.resize(N);
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

  // The explicit form of the implicit iteration scheme of eq. (8) in Exl et al.
  real t2 = dt * dt * dot(t, t);
  real3 m = ((4 - t2) * m0 + 4 * dt * t) / (4 + t2);

  mField.setVectorInCell(idx, m);
}

static inline real BarzilaiBorweinStepSize(std::vector<Field>& dm,
                                           std::vector<Field>& dtorque, int n) {
  real nom = 0, div = 0;
  for (size_t i = 0; i < dm.size(); i++) {
    if (n % 2 == 0) {
      nom += dotSum(dm[i], dm[i]);
      div += dotSum(dm[i], dtorque[i]);
    } else {
      nom += dotSum(dm[i], dtorque[i]);
      div += dotSum(dtorque[i], dtorque[i]);
    }
  }

  if (div == 0.0)
    return 1e-14;  // TODO: figure out safe stepsize

  return nom / div;
}

void Minimizer::step() {
  for (size_t i = 0; i < magnets_.size(); i++) {

    m0[i] = magnets_[i]->magnetization()->eval();

    if (nsteps_ == 0) {
      t0[i] = torques_[i].eval();
      m1[i] = Field(magnets_[i]->system(), 3);
    } else {
      t0[i] = t1[i];
    }

    int ncells = m1[i].grid().ncells();
    cudaLaunch(ncells, k_step, m1[i].cu(), m0[i].cu(), t0[i].cu(), stepsize_);
  }
  
  for (size_t i = 0; i < magnets_.size(); i++)
    magnets_[i]->magnetization()->set(m1[i]);  // normalizes
    
  for (size_t i = 0; i < magnets_.size(); i++)
    t1[i] = torques_[i].eval();

  // Reuse m0 and t0 for efficiency, but declare alias for clarity
  std::vector<Field> &dm = m0, &dt = t0;
  real magDiff = 0;
  for (size_t i = 0; i < magnets_.size(); i++) {
    add(dm[i], real(+1), m1[i], real(-1), m0[i]);
    add(dt[i], real(-1), t1[i], real(+1), t0[i]);  // opposite sign
    // The Barzilai-Borwein step uses the difference in steepest ascend,
    // while relax torque is the steepest *descend* direction.

    magDiff = std::max(magDiff, maxVecNorm(dm[i]));
  }

  stepsize_ = BarzilaiBorweinStepSize(dm, dt, nsteps_);
  addMagDiff(magDiff);

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
