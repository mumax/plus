#pragma once

#include <cufft.h>

#include <memory>
#include <vector>

#include "datatypes.hpp"
#include "strayfield.hpp"
#include "strayfieldkernel.hpp"

class Parameter;
class Field;
class System;
class Ferromagnet;

class StrayFieldFFTExecutor : public StrayFieldExecutor {
 public:
  StrayFieldFFTExecutor(const Ferromagnet* magnet,
                        std::shared_ptr<const System> system);
  ~StrayFieldFFTExecutor();
  Field exec() const;
  Method method() const { return StrayFieldExecutor::METHOD_FFT; }

 private:
  StrayFieldKernel kernel_;
  int3 fftSize;
  std::vector<complex*> kfft, mfft, hfft;
  cufftHandle forwardPlan;
  cufftHandle backwardPlan;
};
