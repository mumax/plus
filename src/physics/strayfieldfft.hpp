#pragma once

#include <cufft.h>

#include <vector>

#include "datatypes.hpp"
#include "grid.hpp"
#include "strayfield.hpp"
#include "strayfieldkernel.hpp"

class Parameter;
class Field;

class StrayFieldFFTExecutor : public StrayFieldExecutor {
 public:
  StrayFieldFFTExecutor(Grid gridOut, Grid gridIn, real3 cellsize);
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
