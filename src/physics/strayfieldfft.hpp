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

class StrayFieldFFTExecutor : public StrayFieldExecutor {
 public:
  StrayFieldFFTExecutor(std::shared_ptr<const System> systemIn,
                        std::shared_ptr<const System> systemOut);
  ~StrayFieldFFTExecutor();
  void exec(Field* h, const Field* m, const Parameter* msat) const;
  Method method() const { return StrayFieldExecutor::METHOD_FFT; }

 private:
  StrayFieldKernel kernel_;
  int3 fftSize;
  std::vector<complex*> kfft, mfft, hfft;
  cufftHandle forwardPlan;
  cufftHandle backwardPlan;
};
