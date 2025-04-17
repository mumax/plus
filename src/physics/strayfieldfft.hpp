#pragma once

#include <cufft.h>

#include <memory>
#include <vector>

#include "datatypes.hpp"
#include "strayfield.hpp"
#include "strayfieldkernel.hpp"

class Field;
class System;
class Magnet;

/** A StraFieldFFTExecutor uses the FFT method to compute stray fields. */
class StrayFieldFFTExecutor : public StrayFieldExecutor {
 public:
  /**
   * Construct a StrayFieldFFTExecutor.
   *
   * @param magnet the source of the stray field
   * @param system the system in which to compute the stray field
   */
  StrayFieldFFTExecutor(const Magnet* magnet,
                        std::shared_ptr<const System> system, int order);

  /** Destruct the executor. */
  ~StrayFieldFFTExecutor();

  /** Compute and return the stray field. */
  Field exec() const;

  /** Return the computation method which is METHOD_FFT. */
  Method method() const { return StrayFieldExecutor::METHOD_FFT; }

  /** Return the asymptotic computation order. */
  int order() const { return kernel_.order(); }

 private:
  StrayFieldKernel kernel_;
  int3 fftSize;
  std::vector<complex*> kfft, mfft, hfft;
  cufftHandle forwardPlan;
  cufftHandle backwardPlan;
};
