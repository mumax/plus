#pragma once

#include <memory>

#include "strayfield.hpp"
#include "strayfieldkernel.hpp"

class System;
class Field;
class Parameter;

class StrayFieldBruteExecutor : public StrayFieldExecutor {
 public:
  StrayFieldBruteExecutor(std::shared_ptr<const System> inSystem,
                          std::shared_ptr<const System> outSystem);
  void exec(Field* h, const Field* m, const Parameter* msat) const;
  Method method() const { return StrayFieldExecutor::METHOD_BRUTE; }

 private:
  StrayFieldKernel kernel_;
};
