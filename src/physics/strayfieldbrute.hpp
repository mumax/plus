#pragma once

#include <memory>

#include "strayfield.hpp"
#include "strayfieldkernel.hpp"

class System;
class Field;
class Parameter;
class Magnet;

/**
 * A StraFieldBruteExecutor uses a brute force method to compute stray fields.
 */
class StrayFieldBruteExecutor : public StrayFieldExecutor {
 public:
  /**
   * Construct a StrayFieldBruteExecutor.
   *
   * @param magnet the source of the stray field
   * @param system the system in which to compute the stray field
   */
  StrayFieldBruteExecutor(const Magnet* magnet,
                          std::shared_ptr<const System> system, int order);

  /** Compute and return the stray field. */
  Field exec() const;

  /** Return the computation method which is METHOD_BRUTE. */
  Method method() const { return StrayFieldExecutor::METHOD_BRUTE; }

  /** Return the computation order. */
  int order() const { return kernel_.order(); }

 private:
  StrayFieldKernel kernel_;
};
