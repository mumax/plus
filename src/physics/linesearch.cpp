#include "linesearch.hpp"

#include <algorithm>
#include <cmath>

namespace {

struct lsPoint {
  double step, f, slope;
};

void cubicCoeffs(double fa,
                 double da,
                 double sa,
                 double fb,
                 double db,
                 double sb,
                 double& theta,
                 double& gamma) {
  theta = 3.0 * (fa - fb) / (sb - sa) + da + db;
  double s = std::max(std::abs(theta), std::max(std::abs(da), std::abs(db)));
  gamma = s * std::sqrt(std::max(0.0, (theta / s) * (theta / s) - (da / s) * (db / s)));
}

// Given the two current bracket endpoints and a newly evaluated trial point,
// update the bracket and propose a next trial step (newStep). Returns the case code
// (0: no usable step, newStep is the trial step itself).
// stpmin/stpmax are NOT a global clamp, they only bound the step while
// extrapolating (unbracketed cases 3 and 4).
int cstep(lsPoint& best,
          lsPoint& other,
          const lsPoint& trial,
          bool& bracketed,
          double stpmin,
          double stpmax,
          double fref,
          double fnoise,
          double& newStep) {
  // check: 1) bracketed, 2) min<step<max, 3) slope points toward trial, 4)
  // stepmax>stepmin
  if ((bracketed && (trial.step <= std::min(best.step, other.step) ||
                     trial.step >= std::max(best.step, other.step))) ||
      best.slope * (trial.step - best.step) >= 0.0 || stpmax < stpmin) {
    newStep = trial.step;
    return 0;
  }

  bool higher = trial.f > best.f && trial.f > fref + fnoise;
  double sgnd = trial.slope * (best.slope / std::abs(best.slope));
  double chosen, cubic, quad, theta, gamma, p, q, r;
  int casecode;

  if (higher) {
    // Case 1: trial overshot, minimum bracketed between best and trial.
    // Cubic if it is closer to best than the quadratic, else their average.
    casecode = 1;
    cubicCoeffs(best.f, best.slope, best.step, trial.f, trial.slope, trial.step, theta,
                gamma);
    if (trial.step < best.step) {
      gamma = -gamma;
    }
    p = (gamma - best.slope) + theta;
    q = ((gamma - best.slope) + gamma) + trial.slope;
    r = p / q;
    cubic = best.step + r * (trial.step - best.step);
    quad =
        best.step +
        ((best.slope / ((best.f - trial.f) / (trial.step - best.step) + best.slope)) /
         2.0) *
            (trial.step - best.step);
    if (std::abs(cubic - best.step) < std::abs(quad - best.step)) {
      chosen = cubic;
    } else {
      chosen = (cubic + quad) / 2.0;
    }
    bracketed = true;

  } else if (sgnd < 0.0) {
    // Case 2: lower f, slopes of opposite sign. Bracketed.
    // Cubic if it is farther from trial than the secant, else the secant (minimizer of
    // the quadratic).
    casecode = 2;
    cubicCoeffs(best.f, best.slope, best.step, trial.f, trial.slope, trial.step, theta,
                gamma);
    if (trial.step > best.step) {
      gamma = -gamma;
    }
    p = (gamma - trial.slope) + theta;
    q = ((gamma - trial.slope) + gamma) + best.slope;
    r = p / q;
    cubic = trial.step + r * (best.step - trial.step);
    quad = trial.step +
           (trial.slope / (trial.slope - best.slope)) * (best.step - trial.step);
    if (std::abs(cubic - trial.step) > std::abs(quad - trial.step)) {
      chosen = cubic;
    } else {
      chosen = quad;
    }
    bracketed = true;

  } else if (std::abs(trial.slope) < std::abs(best.slope)) {
    // Case 3: lower f, same-sign slopes, |slope| decreasing.
    casecode = 3;
    cubicCoeffs(best.f, best.slope, best.step, trial.f, trial.slope, trial.step, theta,
                gamma);
    if (trial.step > best.step) {
      gamma = -gamma;
    }
    p = (gamma - trial.slope) + theta;
    q = (gamma + (best.slope - trial.slope)) + gamma;
    r = p / q;
    // Cubic only if it tends to infinity in the step direction or its
    // minimum lies beyond trial; otherwise use the extrapolation limit.
    if (r < 0.0 && gamma != 0.0) {
      cubic = trial.step + r * (best.step - trial.step);
    } else if (trial.step > best.step) {
      cubic = stpmax;
    } else {
      cubic = stpmin;
    }
    quad = trial.step +
           (trial.slope / (trial.slope - best.slope)) * (best.step - trial.step);
    if (bracketed) {
      // Cubic if closer to trial than the secant, else the secant; then
      // stay within 66% of the way to the other endpoint.
      if (std::abs(cubic - trial.step) < std::abs(quad - trial.step)) {
        chosen = cubic;
      } else {
        chosen = quad;
      }
      if (trial.step > best.step) {
        chosen = std::min(trial.step + 0.66 * (other.step - trial.step), chosen);
      } else {
        chosen = std::max(trial.step + 0.66 * (other.step - trial.step), chosen);
      }
    } else {
      // Extrapolating: cubic if farther from trial than the secant.
      if (std::abs(cubic - trial.step) > std::abs(quad - trial.step)) {
        chosen = cubic;
      } else {
        chosen = quad;
      }
      chosen = std::min(stpmax, chosen);
      chosen = std::max(stpmin, chosen);
    }

  } else {
    // Case 4: lower f, same-sign slopes, |slope| not decreasing.
    casecode = 4;
    if (bracketed) {
      cubicCoeffs(trial.f, trial.slope, trial.step, other.f, other.slope, other.step,
                  theta, gamma);
      if (trial.step > other.step) {
        gamma = -gamma;
      }
      p = (gamma - trial.slope) + theta;
      q = ((gamma - trial.slope) + gamma) + other.slope;
      r = p / q;
      chosen = trial.step + r * (other.step - trial.step);
    } else if (trial.step > best.step) {
      chosen = stpmax;
    } else {
      chosen = stpmin;
    }
  }

  // Update the bracket endpoints with the newly evaluated trial point.
  if (higher) {
    other = trial;
  } else {
    if (sgnd < 0.0) {
      other = best;
    }
    best = trial;
  }

  newStep = chosen;
  return casecode;
}

}  // namespace

LineSearchResult MTLineSearch(double f0,
                              double slope0,
                              double maxDirNorm,
                              double step0,
                              double maxStepAngle,
                              double eScale,
                              const LineSearchEval& eval) {
  // these are all tunable
  const double ftol = 1e-4;  // μ (Wolfe c1)
  const double gtol = 0.9;   // η (Wolfe c2); loose, standard for quasi-Newton
  const double extrapFactor =
      4.0;  // unbracketed: next trial <= step + extrapFactor*(step-best)
  const double extrapLower =
      1.1;  // unbracketed: next trial >= step + extrapLower*(step-best)
  const int maxEvals = 20;
  const double bracketTol =
      1e-7;  // stop when bracket width <= bracketTol * upper end, floored at 4*eps
  const double maxAngleLimit = 89.0;

  if (!(step0 > 0) || std::isinf(step0)) {  // also catches NaN
    return {f0, 0, LS_NO_PROGRESS};
  }

  // slope0= Σ V·M_s·g·s at step 0 (|m0| = 1, so the 1/|x| factor is 1)
  // slope0= g.s is φ'(0); must be negative (s is a descent direction)
  if (!(slope0 < 0)) {  // also catches NaN
    return {f0, 0, LS_NO_DESCENT};
  }

  // Step limits, as limits on the displacement alpha*|s|, so they mean the same
  // thing whatever the torque scale.
#if LS_CAYLEY
  // Cayley curve: the rotation is 2*atan(alpha*|s|/2), which saturates at pi.
  // Beyond alpha*|s| ~ 1e2 (rotation > 177 deg) a larger step adds nothing.
  const double maxDisp = 1e3;  // tunable
#else
  const double maxDisp = 1e3;
#endif
  double minStep = 1e-15;
  double maxStep = 1e15;
  if (maxDirNorm > 0) {
    minStep = 1e-15 / maxDirNorm;
    maxStep = maxDisp / maxDirNorm;
  }

  // Angle cap on the rotation of any cell.
  if (maxStepAngle > 0 && maxDirNorm > 0) {
    double effAngle = std::min(maxStepAngle, maxAngleLimit);
#if LS_CAYLEY
    // exact rotation 2*atan(alpha*|s|/2)  =>  alpha_max = 2*tan(theta/2)/|s|
    double capStep = 2.0 * std::tan(0.5 * effAngle * M_PI / 180.0) / maxDirNorm;
#else
    // s is ~tangent, so a step alpha rotates a cell by atan(alpha*|s|)
    double capStep = std::tan(effAngle * M_PI / 180.0) / maxDirNorm;
#endif
    if (capStep < maxStep)
      maxStep = capStep;
  }
  maxStep = std::max(maxStep, minStep);  // so cstep doesn't see stpmax < stpmin

  // Stage 1 works on ψ(α) = φ(α) - φ(0) - μ φ'(0) α, for which ψ(α) <= 0
  // is exactly sufficient decrease. φ(0) is dropped here.
  const double armijoSlope = ftol * slope0;  // μφ'(0): slope of the sufficient-decrease
                                             // line φ(0) + armijoSlope·α (negative)
  auto toPsi = [&](const lsPoint& p) {
    return lsPoint{p.step, p.f - p.step * armijoSlope, p.slope - armijoSlope};
  };
  auto fromPsi = [&](const lsPoint& p) {
    return lsPoint{p.step, p.f + p.step * armijoSlope, p.slope + armijoSlope};
  };

  lsPoint bestPoint{0, f0, slope0};
  lsPoint otherEndPoint = bestPoint;
  double fLow = f0;    // lowest phi seen in this search
  double psiLow = f0;  // lowest psi seen (toPsi does not subtract f0, so psi(0) = f0)

  double step = step0;
  bool bracketed = false, stage1 = true;
  int casecode = 1;  // 0 means no usable cstep step, fall back to bestPoint
  int numEvals = 0;
  double width = maxStep - minStep,
         width1 = 2 * (maxStep - minStep);  // bisection safeguard
  double stmin, stmax;
  // float32 resolution of the energy
  double fnoise = realEps * std::max(std::abs(f0), eScale);

  while (true) {
    if (std::isnan(step) || std::isinf(step)) {
      casecode = 0;
    }
    // Limits handed to cstep.
    if (bracketed) {
      stmin = std::min(bestPoint.step, otherEndPoint.step);
      stmax = std::max(bestPoint.step, otherEndPoint.step);
    } else {
      if (numEvals == 0) {
        stmin = 0;  // no forced growth before the first evaluation
      } else {
        stmin = step + extrapLower * (step - bestPoint.step);
      }
      stmax = step + extrapFactor * (step - bestPoint.step);
    }
    // Bracket too narrow to subdivide: relative to the step, and absolute
    // displacement in m (float32 can't resolve below ~4 ulps of |m| = 1).
    bool bracketTight =
        bracketed && (stmax - stmin <= bracketTol * stmax ||
                      (maxDirNorm > 0 && (stmax - stmin) * maxDirNorm <= 4 * realEps));

    // step stays double and must not be rounded to float here: the
    // step == maxStep / step == minStep tests below rely on the clamp assigning
    // the limits exactly. The cast to real happens only at the use sites.
    step = std::min(std::max(step, minStep), maxStep);

    // Out of evals, stalled, or bracket collapsed: fall back to the best point so
    // M and the torque end up there. If bestPoint.step == 0 this evaluates at m0
    // and returns step == 0: callers MUST treat that as a failed line search.
    if ((bracketed && (step <= stmin || step >= stmax)) || numEvals >= maxEvals - 1 ||
        casecode == 0 || bracketTight) {
      step = bestPoint.step;
    }

    auto [f, slope] = eval(step);
    numEvals++;

    // A non-finite trial must not reach cstep: NaN compares false everywhere, so
    // cstep would take it as an improvement.
    if (std::isnan(f) || std::isinf(f) || std::isnan(slope) || std::isinf(slope)) {
      if (step == bestPoint.step || numEvals >= maxEvals) {
        // M and torque are NOT restored here: callers must restore M.
        return {f0, 0, LS_NO_PROGRESS};
      }
      step = bestPoint.step + 0.25 * (step - bestPoint.step);
      casecode = 1;  // 0 would trigger the best-point fallback at the top of the loop
      continue;
    }

    double ftest1 = f0 + step * armijoSlope +
                    fnoise;  // Armijo bound: φ(0) + μ α φ'(0), plus fnoise slack
    bool curvatureOK = std::abs(slope) <= gtol * (-slope0);

    int info = LS_RUNNING;

    // bracketed but hit edge of bracket or outside
    if ((bracketed && (step <= stmin || step >= stmax)) || casecode == 0) {
      info = LS_NO_PROGRESS;
    }
    // hit maxStep but still decreasing sufficiently (and so is slope)
    if (step == maxStep && f <= ftest1 && slope <= armijoSlope) {
      info = LS_AT_MAX_STEP;
    }
    // hit minStep, or step too small for float32 to resolve
    bool unresolvable = maxDirNorm > 0 && step * maxDirNorm <= 4 * realEps;
    if ((step == minStep || unresolvable) && (f > ftest1 || slope >= armijoSlope)) {
      info = LS_AT_MIN_STEP;
    }
    if (numEvals >= maxEvals) {
      info = LS_MAX_EVALS;
    }
    // bracket shrank too small
    if (bracketTight) {
      info = LS_BRACKET_TOL;
    }
    // passes both strong Wolfe conditions
    if (f <= ftest1 && curvatureOK) {
      info = LS_CONVERGED;
    }
    if (info != LS_RUNNING) {
      return {f, step, info};
    }

    if (stage1 && f <= ftest1 && slope >= 0) {
      stage1 = false;
    }

    lsPoint trial{step, f, slope};
    fLow = std::min(fLow, f);
    psiLow = std::min(psiLow, f - step * armijoSlope);
    // if stage1, f is better than bestp, and psi>0, bracket on psi
    if (stage1 && f <= fLow + fnoise && f > ftest1) {
      // Stage 1: apply the updating algorithm to ψ instead of φ.
      lsPoint b = toPsi(bestPoint), o = toPsi(otherEndPoint);
      casecode =
          cstep(b, o, toPsi(trial), bracketed, stmin, stmax, psiLow, fnoise, step);
      bestPoint = fromPsi(b);
      otherEndPoint = fromPsi(o);
    } else {
      // Work on φ itself: stage 2, or stage 1 when the trial is worse than best/trial
      // satisfies Armijo
      casecode = cstep(bestPoint, otherEndPoint, trial, bracketed, stmin, stmax, fLow,
                       fnoise, step);
    }
    // Bisection safeguard: if the bracket hasn't shrunk to <66% of its width two
    // iterations ago, discard cstep's proposal and bisect. new trial point is midway
    // between bracket endpoints
    if (bracketed) {
      if (std::abs(otherEndPoint.step - bestPoint.step) >= 0.66 * width1) {
        step = bestPoint.step + 0.5 * (otherEndPoint.step - bestPoint.step);
      }
      width1 = width;
      width = std::abs(otherEndPoint.step - bestPoint.step);
    }
  }
}
