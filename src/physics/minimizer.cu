#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "antiferromagnet.hpp"
#include "cudaerror.hpp"
#include "cudalaunch.hpp"
#include "cudastream.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "gpubuffer.hpp"
#include "linesearch.hpp"
#include "minimizer.hpp"
#include "mumaxworld.hpp"
#include "ncafm.hpp"
#include "reduce.hpp"
#include "rigidbodymodes_common.cuh"
#include "torque.hpp"

// Max per-cell |dm| tried in order (as d/maxDirNorm along the torque) when a line
// search fails. The first one that passes the GLL test is accepted.
static const std::array<real, 3> fallbackDisplacements = {0.1, 0.05, 0.01};
// Largest |B_perp| (T) for which skipping the BB clamp check is assumed safe.
static const double minimizerBMax = 1e6;

// With verbose >= 3, line search summaries are only printed for the first
// lsLogMaxStep minimizer steps, to keep the output manageable.
static const int lsLogMaxStep = 200;

Minimizer::Minimizer(const Ferromagnet* magnet,
                     real stopMaxMagDiff,
                     int nMagDiffSamples,
                     int lineSearchMode,
                     double maxStepAngle,
                     int maxStall,
                     int verbose)
    : magnets_({magnet}),
      torques_({relaxTorqueQuantity(magnet)}),
      nMagDiffSamples_(nMagDiffSamples),
      stopMaxMagDiff_(stopMaxMagDiff),
      mode_(lineSearchMode),
      maxStepAngle_(maxStepAngle),
      maxStall_(maxStall),
      verbose_(verbose),
      t0(1),
      t1(1),
      m0(1),
      m1(1) {
  stepsize_ = 1e-14;  // TODO: figure out how to make descent guess
  // TODO: check if input arguments are sane
  if (mode_ < 0 || mode_ > 3)
    throw std::invalid_argument("Minimizer: lineSearchMode must be 0..3.");
}

Minimizer::Minimizer(const HostMagnet* magnet,
                     real stopMaxMagDiff,
                     int nMagDiffSamples,
                     int lineSearchMode,
                     double maxStepAngle,
                     int maxStall,
                     int verbose)
    : magnets_(magnet->sublattices()),
      nMagDiffSamples_(nMagDiffSamples),
      stopMaxMagDiff_(stopMaxMagDiff),
      mode_(lineSearchMode),
      maxStepAngle_(maxStepAngle),
      maxStall_(maxStall),
      verbose_(verbose),
      t0(magnets_.size()),
      t1(magnets_.size()),
      m0(magnets_.size()),
      m1(magnets_.size()) {
  stepsize_ = 1e-14;
  for (auto sub : magnets_)
    torques_.push_back(relaxTorqueQuantity(sub));
  if (mode_ < 0 || mode_ > 3)
    throw std::invalid_argument("Minimizer: lineSearchMode must be 0..3.");
}

Minimizer::Minimizer(const MumaxWorld* world,
                     real stopMaxMagDiff,
                     int nMagDiffSamples,
                     int lineSearchMode,
                     double maxStepAngle,
                     int maxStall,
                     int verbose)
    : nMagDiffSamples_(nMagDiffSamples),
      stopMaxMagDiff_(stopMaxMagDiff),
      mode_(lineSearchMode),
      maxStepAngle_(maxStepAngle),
      maxStall_(maxStall),
      verbose_(verbose) {
  if (mode_ < 0 || mode_ > 3)
    throw std::invalid_argument("Minimizer: lineSearchMode must be 0..3.");

  // Find all ferromagnets (FM instances or sublattices)
  for (const auto pair : world->magnets()) {
    if (auto host = pair.second->asHost()) {
      for (auto sub : host->sublattices())
        magnets_.push_back(sub);
    } else if (const Ferromagnet* mag = pair.second->asFM()) {
      magnets_.push_back(mag);
    }
  }

  for (auto magnet : magnets_)
    torques_.push_back(relaxTorqueQuantity(magnet));

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

  // line-search state
  tTrial_.assign(magnets_.size(), Field());
  lastEnergies_.clear();
  firstStep_ = true;
  stuck_ = false;
  hBadNext_ = false;
  gllOff_ = false;
  stallCount_ = 0;
  fRef_ = 0;
  lastF_ = 0;
  lastScale_ = 0;
  haveF_ = false;
  rescueStall_ = 0;
  fRescue_ = 0;
  haveRescue_ = false;
  energyScale_ = 0;
  nLineSearches_ = 0;
  nGllRescues_ = 0;

  // Gate for the BB step clamp: |h|*max|t| can only exceed 1/eps when
  // |h| > 1/(eps * gammaMax * bMax), with t = gamma * B_perp.
  double gammaMax = 0;
  for (const Ferromagnet* magnet : magnets_) {
    const Parameter& g = magnet->gamma;
    double gm = g.isUniform() ? std::abs(double(g.getUniformValue()))
                              : double(maxAbsValue(g.eval()));
    gammaMax = std::max(gammaMax, gm);
  }
  hCheck_ = gammaMax > 0 ? 1 / (realEps * gammaMax * minimizerBMax)
                         : std::numeric_limits<double>::infinity();

  while (!converged() && !stuck_)
    step();

  if (verbose_ > 0)
    std::cout << "Minimize: " << nsteps_ << " steps, " << nLineSearches_
              << " line searches, " << nGllRescues_ << " GLL rescues" << std::endl;
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
                                           std::vector<Field>& dtorque,
                                           int n,
                                           bool& degenerate) {
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

  degenerate = (div == 0.0);
  if (degenerate)
    return 1e-14;  // TODO: figure out safe stepsize

  return nom / div;
}

static real maxVecNormAll(const std::vector<Field>& v) {
  real m = 0;
  for (const Field& f : v)
    m = std::max(m, maxVecNorm(f));
  return m;
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
  }

  real h = stepsize_;

  // Modes 2 and 3: sanitize the BB step size.
  //   - NaN, Inf, 0, or a degenerate BB formula on the previous step: bad step,
  //     goes to recoverStep.
  //   - hMax keeps dt*dt finite.
  //   - Upper clamp: the Cayley update is an exact rotation by 2*atan(d/2), d =
  //   |h|*max|t|.
  //     Once d > 1/eps the tangent part of m_new (~4/d) is below the resolution of
  //     |m| = 1, so d is clamped there (sign of h is kept). max|t| is only computed
  //     (one reduction) when |h| > hCheck.
  const double hMax = 1e18;

  bool badH = false;
  if (mode_ >= 2) {
    double hf = h;
    if (hBadNext_ || !std::isfinite(hf) || hf == 0) {
      badH = true;
    } else {
      if (std::abs(hf) > hMax)
        hf = std::copysign(hMax, hf);
      // Only compute max|t| when the clamp could possibly fire (see hCheck_ in exec()).
      if (std::abs(hf) > hCheck_) {
        double tMax = maxVecNormAll(t0);
        if (tMax > 0 && std::abs(hf) * tMax > 1 / realEps)
          hf = std::copysign(1 / (realEps * tMax), hf);
      }
      h = hf;
    }
  }
  hBadNext_ = false;
  real trialF = 0;
  bool moved = true;

  if (mode_ == 0) {
    // Original behavior: no line search at all, unconditional BB step every time.
    stepAlong(m0, t0, h);
    for (size_t i = 0; i < magnets_.size(); i++)
      t1[i] = torques_[i].eval();
    firstStep_ = false;
  } else {
    auto energyAtM0 = [&]() {  // cached, or computed lazily
      if (!haveF_) {
        lastF_ = energyAtAccepted();
        lastScale_ = energyScale_;
        haveF_ = true;
      }
      return lastF_;
    };

    if (firstStep_ || badH) {
      // Cold start and bad BB step: line search from the cold seed,
      // then fixed-displacement fallback.
      real f0 = energyAtM0();
      moved = recoverStep(m0, t0, f0, lastScale_, trialF);
      firstStep_ = false;
      if (moved) {
        lastF_ = trialF;
        lastScale_ = energyScale_;
        haveF_ = true;
      }

    } else if (mode_ == 3) {
      real f0 = energyAtM0();
      real f0Scale = lastScale_;
      stepAlong(m0, t0, h);
      trialF = trialEnergyAndTorque();
      if (!std::isfinite(trialF)) {
        // BB step produced a non-finite energy: reject it and recover from m0.
        setMagnetization(m0);
        moved = recoverStep(m0, t0, f0, f0Scale, trialF);
      } else if (!gllOff_ && !lastEnergies_.empty() &&
                 trialF > maxEnergy() + 4 * realEps * energyScale_) {
        // GLL rejection (Exl et al. 2014).
        moved = gllRescue(f0, f0Scale, h, trialF);
      } else {
        std::swap(t1, tTrial_);  // accept BB step; t1 = torque
      }
      if (moved) {
        lastF_ = trialF;
        lastScale_ = energyScale_;
        haveF_ = true;
      }

    } else {  // modes 1 and 2 after the first step: plain BB, no energy evaluation
      stepAlong(m0, t0, h);
      for (size_t i = 0; i < magnets_.size(); i++)
        t1[i] = torques_[i].eval();
      haveF_ = false;
    }

    if (mode_ == 3 && moved) {
      // end a no-progress streak (and re-enable GLL rescues) once the energy has
      // dropped below the streak's reference by more than noise
      if (stallCount_ > 0 && trialF < fRef_ - 4 * realEps * lastScale_) {
        stallCount_ = 0;
        gllOff_ = false;
      }
      addEnergy(trialF);
    }
  }

  if (!moved) {
    // No descent step exists at working precision: M == m0, t0 is the torque there.
    // Tell exec() to stop;  the dm ring (deque in mmplus) is untouched.
    // mmplus doesn't report torque to the user, so no setMaxTorque
    std::cerr << "WARNING: Minimize: no descent step found, max|torque|="
              << maxVecNormAll(t0) << "; declaring converged" << std::endl;
    stuck_ = true;
    nsteps_ += 1;
    return;
  }

  // again,mmplus doesn't report torque to the user, so no setMaxTorque

  // The line search moves M along a different curve than the BB step, so take
  // the magnetization after the move rather than the BB result in m1.
  if (mode_ >= 1) {
    for (size_t i = 0; i < magnets_.size(); i++)
      m1[i] = magnets_[i]->magnetization()->eval();
  }

  // Reuse m0 and t0 for efficiency, but declare alias for clarity
  std::vector<Field>&dm = m0, &dt = t0;
  real magDiff = 0;
  for (size_t i = 0; i < magnets_.size(); i++) {
    add(dm[i], real(+1), m1[i], real(-1), m0[i]);
    add(dt[i], real(-1), t1[i], real(+1), t0[i]);  // opposite sign
    // The Barzilai-Borwein step uses the difference in steepest ascent,
    // while relax torque is the steepest *descent* direction.

    magDiff = std::max(magDiff, maxVecNorm(dm[i]));
  }

  bool degenerate;
  stepsize_ = BarzilaiBorweinStepSize(dm, dt, nsteps_, degenerate);
  if (degenerate && mode_ >= 2)
    hBadNext_ = true;  // let recoverStep handle it
  addMagDiff(magDiff);

  // Mode 2 and 3: the magnetization differences say "converged"; try one line search
  // before exiting.
  if (mode_ >= 2 && converged())
    convergenceRescue();

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

//--- new linesearch stuff ---

// Per-cell contribution to phi'(step) for phi(a) = E(normalize(base + a*dir)):
//   -Ms (torque . dir) / |base + step*dir|,
// where torque (= -gradient, tangent to m) is evaluated at the normalized trial
// point. The 1/|x| factor is the derivative of the normalization.
// level 1: per-thread Kahan sum of phi'(step) contributions, then one partial per block
__global__ void k_slopePartial(real* partials,
                               const CuField baseField,
                               const CuField dirField,
                               const CuField torqueField,
                               const CuParameter msat,
                               const CuParameter gamma,
                               real step) {
  int ncells = baseField.system.grid.ncells();
  int stride = blockDim.x * gridDim.x;

  real acc[1] = {0};
  real comp[1] = {0};
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < ncells; i += stride) {
    if (!baseField.cellInGeometry(i))
      continue;
    real3 s = dirField.vectorAt(i);
    real3 t = torqueField.vectorAt(i);  // torque at the trial point
    real gam = gamma.valueAt(i);        // torque = gam * B_perp
    if (gam == 0)
      continue;
#if LS_CAYLEY
    // phi'(a) = Ms * g . dm/da with g = -t/gam and
    // dm/da = 4[(4 - a^2 S) s - 4 a S m0] / (4 + a^2 S)^2,  S = |s|^2
    real3 m0 = baseField.vectorAt(i);
    real S = dot(s, s);
    real a2S = step * step * S;
    real D = 4 + a2S;
    kahanAdd(acc[0], comp[0],
             -4 * msat.valueAt(i) *
                 ((4 - a2S) * dot(t, s) - 4 * step * S * dot(t, m0)) / (gam * D * D));
#else
    real3 x = baseField.vectorAt(i) + step * s;
    real n2 = dot(x, x);
    if (n2 > 0)
      kahanAdd(acc[0], comp[0], -msat.valueAt(i) * dot(t, s) / (gam * sqrt(n2)));
#endif
  }

  blockReduceSum<real, 1>(acc);
  if (threadIdx.x == 0)
    partials[blockIdx.x] = acc[0];
}

// level 2: single block, double accumulation
__global__ void k_slopeCombine(double* out, const real* partials, int np) {
  double acc[1] = {0};
  for (int i = threadIdx.x; i < np; i += blockDim.x)
    acc[0] += partials[i];

  blockReduceSum<double, 1>(acc);
  if (threadIdx.x == 0)
    *out = acc[0];
}

// phi'(step) = V * sum over cells of -Ms (torque . dir) / |base + step*dir|, summed
// over magnets
double Minimizer::slopeAlongLine(const std::vector<Field>& base,
                                 const std::vector<Field>& dir,
                                 const std::vector<Field>& torque,
                                 real step) {
  double slope = 0;
  for (size_t i = 0; i < magnets_.size(); i++) {
    const Ferromagnet* magnet = magnets_[i];
    if (magnet->msat.assuredZero())
      continue;

    int ncells = base[i].grid().ncells();
    int numBlocks = numReductionBlocks(ncells, k_slopePartial);
    GpuBuffer<real> d_partials(numBlocks);
    GpuBuffer<double> d_sum(1);

    k_slopePartial<<<numBlocks, BLOCKDIM, 0, getCudaStream()>>>(
        d_partials.get(), base[i].cu(), dir[i].cu(), torque[i].cu(), magnet->msat.cu(),
        magnet->gamma.cu(), step);
    checkCudaError(cudaPeekAtLastError());
    k_slopeCombine<<<1, BLOCKDIM, 0, getCudaStream()>>>(d_sum.get(), d_partials.get(),
                                                        numBlocks);
    checkCudaError(cudaPeekAtLastError());

    double sum;
    checkCudaError(cudaMemcpyAsync(&sum, d_sum.get(), sizeof(double),
                                   cudaMemcpyDeviceToHost, getCudaStream()));
    checkCudaError(cudaStreamSynchronize(getCudaStream()));
    slope +=
        sum * magnet->world()->cellVolume();  // volume applied on the host in double
  }
  return slope;
}

// ---------------------------------------------------------------------------
// line searches

// Inexact (Moré-Thuente) line search from base (energy f0) along dir, which must
// be the torque at base. Used for the first step, for bad-BB-step recovery, for
// BB-step rejections and for convergence rescues.
//
// Contract:
//   - returns true:  M is at the accepted point, t1 holds the torque there,
//                    newF is the energy there and step is the accepted step.
//   - returns false: M == base, t1 untouched, newF == f0 and step == 0.
// dir may alias t1 (convergence rescue); it is only read until the search ends.
bool Minimizer::sdLineSearch(const std::vector<Field>& base,
                             const std::vector<Field>& dir,
                             real f0,
                             real f0Scale,
                             double seed,
                             real& newF,
                             double& step) {
  nLineSearches_++;
  const double coldDisplacement =
      0.1;  // initial max |dm| per cell on a cold start; tunable

  double maxDirNorm = maxVecNormAll(dir);
  double step0 = seed;
  if (!(step0 > 0) || std::isinf(step0)) {  // also catches NaN
    if (maxDirNorm > 0)
      step0 = coldDisplacement / maxDirNorm;
  }

  // dir is the torque at base, so it is also the torque at step 0
  double slope0 = slopeAlongLine(base, dir, dir, 0);

  auto eval = [&](double alpha) -> std::pair<double, double> {
    lsEvals_++;
#if LS_CAYLEY
    stepAlong(base, dir, real(alpha));  // same curve as the BB step
#else
    stepStraight(base, dir, real(alpha));
#endif
    double f = trialEnergyAndTorque();
    double slope = slopeAlongLine(base, dir, tTrial_, real(alpha));
    return {f, slope};
  };
  const int evals0 = lsEvals_;
  LineSearchResult res = MTLineSearch(f0, slope0, maxDirNorm, step0, maxStepAngle_,
                                      f0Scale, eval, verbose_ > 2 ? 1 : verbose_);

  bool ok = res.step > 0 && std::isfinite(res.f) &&
            (res.info == LS_CONVERGED || res.info == LS_AT_MAX_STEP || res.f < f0);

  if (verbose_ > 2 && nsteps_ < lsLogMaxStep)
    std::printf("LS @%d: seed_d=%.3e acc_d=%.3e f-f0=%.3e info=%d evals=%d ok=%d\n",
                nsteps_, step0 * maxDirNorm, res.step * maxDirNorm, res.f - f0,
                res.info, lsEvals_ - evals0, int(ok));
  if (!ok) {
    // M may be at a trial point (step > 0 with lsAtMinStep / lsNoProgress /
    // lsMaxEvals and f >= f0, or a non-finite trial). Put it back; t1 was never
    // touched.
    setMagnetization(base);
    newF = f0;
    step = 0;
    return false;
  }

  std::swap(t1, tTrial_);  // t1 = torque at the accepted point
  newF = res.f;
  step = res.step;
  return true;
}

// Produces a step from base when there is no usable BB step (cold start, a
// non-finite BB step/energy, or a GLL rescue whose seeded search failed): a line
// search from the cold seed, then fixed displacements from fallbackDisplacements
// (the first that passes the GLL test is accepted). Same contract as sdLineSearch.
bool Minimizer::recoverStep(const std::vector<Field>& base,
                            const std::vector<Field>& dir,
                            real f0,
                            real f0Scale,
                            real& newF) {
  setMagnetization(base);
  double step;
  if (sdLineSearch(base, dir, f0, f0Scale, 0, newF, step))
    return true;

  double maxDirNorm = maxVecNormAll(dir);
  if (!(maxDirNorm > 0)) {  // zero (or NaN) torque: nothing to step along
    newF = f0;
    return false;
  }

  real ref = f0;
  if (!lastEnergies_.empty())
    ref = maxEnergy();

  for (real d : fallbackDisplacements) {
    stepAlong(base, dir, d / maxDirNorm);
    real f = trialEnergyAndTorque();
    if (f <= ref + 4 * realEps * energyScale_) {  // false for NaN
      std::swap(t1, tTrial_);
      newF = f;
      return true;
    }
  }

  // No descent step found at any scale.M back at m0, t1 still the torque at m0.
  setMagnetization(base);
  newF = f0;
  return false;
}

// Handles a GLL-rejected BB step: runs a line search from m0 seeded at |h|.
// If the search fails, it falls back to recoverStep rather than taking the
// rejected BB step. Tracks no-progress streaks: after maxStall_ rescues in a row
// without the energy dropping below the streak's reference, GLL rescues are
// disabled (gllOff_) from the next step on.
bool Minimizer::gllRescue(real f0, real f0Scale, real h, real& newF) {
  nGllRescues_++;
  if (stallCount_ == 0)
    fRef_ = f0;
  stallCount_++;
  if (stallCount_ >= maxStall_)
    gllOff_ = true;  // takes effect next step; this rescue still runs

  setMagnetization(m0);  // early returns of the search leave M untouched
  double step;
  if (verbose_ > 2 && nsteps_ < lsLogMaxStep)
    std::printf("  (BB h %s)\n", h < 0 ? "negative" : "positive");
  if (sdLineSearch(m0, t0, f0, f0Scale, std::abs(double(h)), newF, step))
    return true;

  return recoverStep(m0, t0, f0, f0Scale, newF);
}

// Called (modes 2 and 3) when the magnetization-difference criterion says "converged".
// Tries one line search along the current torque. If that finds a real energy
// decrease with a displacement larger than stopMaxMagDiff_, the BB iteration had
// stalled: take the step, clear the history and carry on. Otherwise leave the
// history full so the run stops. On return M and t1 are consistent in every case.
void Minimizer::convergenceRescue() {
  // Energy at the current (accepted) M. Mode 3 cached it when it accepted this
  // step; mode 2's plain BB step did not, so compute it.
  real f0, f0Scale;
  if (haveF_) {
    f0 = lastF_;
    f0Scale = lastScale_;
  } else {
    f0 = energyAtAccepted();
    f0Scale = energyScale_;  // set by the evaluation on the line above
  }
  real noise = 4 * realEps * f0Scale;

  // Is this candidate a new record?
  if (!haveRescue_ || f0 < fRescue_ - noise) {
    // new record energy at a convergence candidate
    fRescue_ = f0;
    haveRescue_ = true;
    rescueStall_ = 0;
  } else {
    rescueStall_++;
    if (rescueStall_ >= maxStall_)
      return;  // bottomed out: let the run converge
  }

  std::vector<Field> mBase(magnets_.size());
  for (size_t i = 0; i < magnets_.size(); i++)
    mBase[i] = magnets_[i]->magnetization()->eval();

  double dirNorm = maxVecNormAll(t1);  // before the search replaces t1
  real newF;
  double step;
  if (!sdLineSearch(mBase, t1, f0, f0Scale, 0, newF, step))
    return;  // M, t1 untouched: converged

  // M moved: keep the cache and the GLL state consistent with the new point.
  // (M is at the last evaluated trial, so energyScale_ matches newF.)
  lastF_ = newF;
  lastScale_ = energyScale_;
  haveF_ = true;
  if (mode_ == 3) {
    addEnergy(newF);
    // same GLL streak-ending rule as in step()
    if (stallCount_ > 0 && newF < fRef_ - 4 * realEps * lastScale_) {
      stallCount_ = 0;
      gllOff_ = false;
    }
  }

  if (newF < f0 - noise && step * dirNorm > stopMaxMagDiff_) {
    // real progress available: BB had stalled
    if (newF < fRescue_ - noise)
      rescueStall_ = 0;                   // the rescue itself set a new record
    fRescue_ = std::min(fRescue_, newF);  // running minimum: never rises
    lastMagDiffs_.clear();                // re-arm the convergence check
    stepsize_ = step;   // restart BB from the step the search accepted
    hBadNext_ = false;  // stepsize_ is now a valid BB seed
  }
  // else: the step was accepted (it did not raise E) but there is nothing
  // significant left, so the history stays full and the run exits.
}

// ---------------------------------------------------------------------------
// energy

// Total energy of all minimized magnets, without throwing on NaN so that callers
// can reject a bad trial point. Also records energyScale_ = max(|E|, sum|E_i|),
// the scale for noise in E: cancellation between magnets makes |E| alone too small.
real Minimizer::minimizerEnergy() {
  real E = 0, sum = 0;
  for (const Ferromagnet* magnet : magnets_) {
    real e = evalTotalEnergy(magnet);
    E += e;
    sum += std::abs(e);
  }
  energyScale_ = std::max(std::abs(E), sum);
  return E;
}

// minimizerEnergy for the current (accepted) magnetization. A non-finite energy
// there cannot be rolled back, so this fails loudly.
real Minimizer::energyAtAccepted() {
  real E = minimizerEnergy();
  if (!std::isfinite(E))
    throw std::runtime_error(
        "Minimize: non-finite energy at the accepted magnetization");
  return E;
}

// Evaluates torque (into tTrial_) and energy at the current magnetization.
real Minimizer::trialEnergyAndTorque() {
  for (size_t i = 0; i < magnets_.size(); i++)
    tTrial_[i] = torques_[i].eval();
  return minimizerEnergy();
}

void Minimizer::addEnergy(real f) {
  lastEnergies_.push_back(f);
  if (lastEnergies_.size() > nMagDiffSamples_)
    lastEnergies_.pop_front();
}

real Minimizer::maxEnergy() const {
  return *std::max_element(lastEnergies_.begin(), lastEnergies_.end());
}

// ---------------------------------------------------------------------------
// moving the magnetization

void Minimizer::setMagnetization(const std::vector<Field>& m) {
  for (size_t i = 0; i < magnets_.size(); i++)
    magnets_[i]->magnetization()->set(m[i]);  // normalizes
}

// M = Cayley step from base along dir with step size h (the BB update).
void Minimizer::stepAlong(const std::vector<Field>& base,
                          const std::vector<Field>& dir,
                          real h) {
  for (size_t i = 0; i < magnets_.size(); i++) {
    int ncells = m1[i].grid().ncells();
    cudaLaunch(ncells, k_step, m1[i].cu(), base[i].cu(), dir[i].cu(), h);
  }
  setMagnetization(m1);
}

// M = normalize(base + alpha*dir), the curve the line search optimizes along.
void Minimizer::stepStraight(const std::vector<Field>& base,
                             const std::vector<Field>& dir,
                             real alpha) {
  for (size_t i = 0; i < magnets_.size(); i++) {
    add(m1[i], real(1), base[i], alpha, dir[i]);
    // normalize(m1[i]);
  }
  setMagnetization(m1);
}
