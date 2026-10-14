#pragma once

#include <deque>

#include "quantityevaluator.hpp"

class MumaxWorld;

// Minimize follows the steepest descent method as per Exl et al., JAP 115,
// 17D118 (2014).
//
// lineSearchMode:
//   0: BB steps, no line search.
//   1: line search for the first step only.
//   2: as 1, and also recover from bad BB steps (NaN/Inf/0), and try a line
//      search when the magnetization difference criterion would declare convergence.
//   3: as 1, and also recover from bad BB steps, and apply the non-monotone
//      (GLL) energy test over the last nMagDiffSamples energies to every BB step.
// maxStepAngle: max rotation (degrees) per line-search trial step, <=0 disables.
// maxStall: consecutive no-progress rescues before rescues are switched off.

class Minimizer {
 public:
  Minimizer(const Ferromagnet*,
            real stopMaxMagDiff,
            int nMagDiffSamples,
            int lineSearchMode = 0,
            double maxStepAngle = 0,
            int maxStall = 3,
            int verbose = 0);
  Minimizer(const HostMagnet*,
            real stopMaxMagDiff,
            int nMagDiffSamples,
            int lineSearchMode = 0,
            double maxStepAngle = 0,
            int maxStall = 3,
            int verbose = 0);
  Minimizer(const MumaxWorld*,
            real stopMaxMagDiff,
            int nMagDiffSamples,
            int lineSearchMode = 0,
            double maxStepAngle = 0,
            int maxStall = 3,
            int verbose = 0);

  void exec();

 private:
  void step();
  std::vector<const Ferromagnet*> magnets_;
  real stepsize_;
  int nsteps_;

  std::vector<FM_FieldQuantity> torques_;
  std::vector<Field> t0, t1, m0, m1;

  real stopMaxMagDiff_;
  int nMagDiffSamples_;
  bool converged() const;
  void addMagDiff(real);
  std::deque<real> lastMagDiffs_;

  // --- line search ---
  int mode_;
  double maxStepAngle_;
  int maxStall_;
  int verbose_;
  int nLineSearches_;  // number of sdLineSearch calls (diagnostic)
  int nGllRescues_;    // number of GLL-rejected BB steps (diagnostic)
  double hCheck_;  // |h| above which the BB step clamp must check max|t|(set in exec())

  std::vector<Field> tTrial_;      // torque at the most recent trial point
  std::deque<real> lastEnergies_;  // energy history for the non-monotone (GLL) test
  void addEnergy(real);
  real maxEnergy() const;

  bool firstStep_;  // true only before the first step has been taken
  bool stuck_;      // no descent step exists at working precision
  bool hBadNext_;  // BB formula degenerate; force recovery on the next step(modes 2, 3)
  bool gllOff_;    // GLL rescues disabled until progress is made (mode 3)
  int stallCount_;  // consecutive GLL rescues without progress
  real fRef_;       // energy at the start of the current no-progress streak

  real lastF_;      // energy at the current magnetization; valid only if haveF_
  real lastScale_;  // energy noise scale at the current magnetization; valid only if
                    // haveF_
  bool haveF_;

  int rescueStall_;  // consecutive convergence candidates without progress (mode 2)
  real fRescue_;     // energy at the last accepted convergence-rescue point (mode 2)
  bool haveRescue_;
  int lsEvals_ = 0;

  real energyScale_;  // max(|E|, sum|E_i|) from the most recent energy evaluation
  real minimizerEnergy();
  real energyAtAccepted();
  real trialEnergyAndTorque();

  void setMagnetization(const std::vector<Field>& m);
  void stepAlong(const std::vector<Field>& base, const std::vector<Field>& dir, real h);
  void stepStraight(const std::vector<Field>& base,
                    const std::vector<Field>& dir,
                    real alpha);
  double slopeAlongLine(const std::vector<Field>& base,
                        const std::vector<Field>& dir,
                        const std::vector<Field>& torque,
                        real step);

  bool sdLineSearch(const std::vector<Field>& base,
                    const std::vector<Field>& dir,
                    real f0,
                    real f0Scale,
                    double seed,
                    real& newF,
                    double& step);
  bool recoverStep(const std::vector<Field>& base,
                   const std::vector<Field>& dir,
                   real f0,
                   real f0Scale,
                   real& newF);
  bool gllRescue(real f0, real f0Scale, real h, real& newF);
  void convergenceRescue();
};
