#pragma once

#include <deque>

#include "quantityevaluator.hpp"

class MumaxWorld;

// Minimize follows the steepest descent method as per Exl et al., JAP 115,
// 17D118 (2014).

class Minimizer {
 public:
  Minimizer(const Ferromagnet*, real stopMaxMagDiff, int nMagDiffSamples);
  Minimizer(const HostMagnet*, real stopMaxMagDiff, int nMagDiffSamples);
  Minimizer(const MumaxWorld*, real stopMaxMagDiff, int nMagDiffSamples);

  void exec();

 private:
  void step();
  std::vector<const Ferromagnet*> magnets_;
  std::vector<real> stepsizes_;
  int nsteps_;

  std::vector<FM_FieldQuantity> torques_;
  std::vector<Field> t0, t1, m0, m1;

  real stopMaxMagDiff_;
  int nMagDiffSamples_;
  bool converged() const;
  void addMagDiff(real);
  std::deque<real> lastMagDiffs_;
};
