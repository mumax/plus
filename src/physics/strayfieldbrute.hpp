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
                          std::shared_ptr<const System> system, int order, double eps, double switchingradius);

  /** Compute and return the stray field. */
  Field exec() const;

  /** Return the computation method which is METHOD_BRUTE. */
  Method method() const { return StrayFieldExecutor::METHOD_BRUTE; }

  /** Return the asymptotic computation order. */
  int order() const { return kernel_.order(); }

  /** Return epsilon. The parameter used to determine the analytical error
   * using epsilon * R³/V
   */
  double eps() const { return kernel_.eps(); }

  /** Return the switching radius. */
  double switchingradius() const { return kernel_.switchingradius();}

 private:
  StrayFieldKernel kernel_;
};
