#pragma once

#include <functional>
#include <limits>
#include <utility>

#include "datatypes.hpp"

// Machine epsilon of real (1.19e-7 for float, 2.2e-16 for double). Replaces
// mumax3's f32eps, which was the float epsilon.
constexpr double realEps = std::numeric_limits<real>::epsilon();

enum LineSearchInfo {
  LS_NO_DESCENT = -1,  // slope0 >= 0: nothing evaluated
  LS_RUNNING = 0,      // internal: keep iterating
  LS_CONVERGED = 1,    // strong Wolfe conditions satisfied
  LS_BRACKET_TOL = 2,  // bracket narrower than tolerance
  LS_MAX_EVALS = 3,    // evaluation budget exhausted (returns best point)
  LS_AT_MIN_STEP = 4,  // stuck at minStep
  LS_AT_MAX_STEP = 5,  // stuck at maxStep (includes the max-rotation-angle cap)
  LS_NO_PROGRESS = 6   // cstep failed, or trial left the bracket
};

// A/B switch for the curve the line search moves along.
//   0: m(alpha) = normalize(base + alpha*dir)         (original)
//   1: m(alpha) = Cayley rotation, as in the BB step  (k_step in minimizer.cu)
// Override at build time with -DLS_CAYLEY=0 or 1. This header is included by
// both linesearch.cpp and minimizer.cu, so both pick it up.
#ifndef LS_CAYLEY
#define LS_CAYLEY 1
#endif

struct LineSearchResult {
  double f;     // energy at the returned step
  double step;  // 0 means no improvement was found, callers must treat as failure
  int info;
};

// Given a step α, sets m = normalize(m0 + α*s), evaluates torque and
// energy there, and returns (energy, slope along s). Must leave the magnetization
// and torque at the trial point, so that on return they correspond to the
// returned step.
using LineSearchEval = std::function<std::pair<double, double>(double step)>;

// Moré-Thuente line search (Moré & Thuente, ACM TOMS 20(3), 1994; MINPACK-2) for
// φ(α) = E(normalize(m0 + α s)), looking for α satisfying the strong Wolfe conditions.
//   f0, slope0: φ(0) and φ'(0) (must be < 0)
//   maxDirNorm: max |s| over cells; maxStepAngle in degrees, <=0 disables
//   eScale: max(|E|, sum|E_i|) at m0, the scale of float32 noise in E
LineSearchResult MTLineSearch(double f0,
                              double slope0,
                              double maxDirNorm,
                              double step0,
                              double maxStepAngle,
                              double eScale,
                              const LineSearchEval& eval);
