#pragma once

#include <memory>

#include "strayfield.hpp"
#include "strayfieldkernel.hpp"

class System;
class Field;
class Parameter;
class Ferromagnet;

class StrayFieldBruteExecutor : public StrayFieldExecutor {
 public:
  StrayFieldBruteExecutor(const Ferromagnet* magnet,
                          std::shared_ptr<const System> system);
  Field exec() const;
  Method method() const { return StrayFieldExecutor::METHOD_BRUTE; }

 private:
  StrayFieldKernel kernel_;
};
