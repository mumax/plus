#pragma once

#include "strayfield.hpp"
#include "strayfieldkernel.hpp"

class StrayFieldBruteExecutor : public StrayFieldExecutor {
 public:
  StrayFieldBruteExecutor(Grid gridOut, Grid gridIn, real3 cellsize);
  void exec(Field* h, const Field* m, const Parameter* msat) const;
  Method method() const { return StrayFieldExecutor::METHOD_BRUTE; }

 private:
  StrayFieldKernel kernel_;
};
