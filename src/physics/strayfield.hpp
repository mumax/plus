#pragma once

#include <memory>

#include "fieldquantity.hpp"

class Parameter;
class Ferromagnet;
class Field;
class System;
class Grid;

/**
 * A StrayFieldExecutor computes the stray field of a magnet on a system.
 *
 * The StrayFieldExecutor is an abstract class. Actual stray field executors
 * should inherit from this class. At the moment, there are two
 * StrayFieldExecutors implemented:
 *
 * @see StrayFieldFFTExecutor
 * @see StrayFieldBrutexecutor
 */
class StrayFieldExecutor {
 public:
  /** Enum of stray field computation methods. */
  enum Method { METHOD_AUTO, METHOD_BRUTE, METHOD_FFT };

  /**
   * Factory method to construct a StrayFieldExecutor.
   *
   * @param magnet the source of the stray field
   * @param system the system in which we compute the stray field
   * @param method the method used for the computation, METHOD_AUTO defaults to
   *               METHOD_FFT at the moment
   */
  static std::unique_ptr<StrayFieldExecutor> create(
      const Ferromagnet* magnet,
      std::shared_ptr<const System> system,
      Method method);

 protected:
  /** Constructor only to be used in constructor of derived classes. */
  StrayFieldExecutor(const Ferromagnet* magnet,
                     std::shared_ptr<const System> system);

 public:
  /** Empty virtual destructor. */
  virtual ~StrayFieldExecutor() {}

  /** Compute and return the stray field. */
  virtual Field exec() const = 0;

  /** Return the method of the executor. */
  virtual Method method() const = 0;

 protected:
  /** Source of the stray field*/
  const Ferromagnet* magnet_;

  /** System in which the stray field will be computed. */
  const std::shared_ptr<const System> system_;
};

/**
 * StrayField is a field quantity which evaluates the stray field of a
 * ferromagnet in a specified system or grid.
 *
 * StrayField uses a StrayFieldExecutor internally to do the actual computation.
 *
 * Side note: if the specified grid matches the grid of the ferromagnet, then
 * this stray field is by definition the demagnetization field of the magnet.
 */
class StrayField : public FieldQuantity {
 public:
  /** Constructor of a StrayField on a specified system.
   *
   * @param magnet the source of the stray field
   * @param system the system in which we compute the stray field
   * @param method the used method for the computation
   */
  StrayField(const Ferromagnet* magnet,
             std::shared_ptr<const System> system,
             StrayFieldExecutor::Method = StrayFieldExecutor::METHOD_AUTO);

  /**
   * Constructor of a StrayField on a specified grid.
   *
   * Internally, a new system will be created (from the specified grid) in which
   * the stray field will be computed.
   * @param magnet the source of the stray field
   * @param grid   used to create a system in which the stray field is computed
   * @param method the used method for the computation
   */
  StrayField(const Ferromagnet* magnet,
             Grid grid,
             StrayFieldExecutor::Method = StrayFieldExecutor::METHOD_AUTO);

  /** Destructor. */
  ~StrayField();

  /** Set the method for the computation of the stray field. */
  void setMethod(StrayFieldExecutor::Method);

  /** Return the magnet which is the source of the stray field. */
  const Ferromagnet* source() const;

  /** Return the number of components of the stray field which is 3. */
  int ncomp() const;

  /** Return the system in which the stray field is computed. */
  std::shared_ptr<const System> system() const;

  /** Return the unit of the stray field, which is "T". */
  std::string unit() const;

  /** Compute and return the stray field. */
  Field eval() const;

  /** Return true if one can be sure that the stray field is exactly zero. */
  bool assuredZero() const;

 private:
  std::shared_ptr<const System> system_;
  const Ferromagnet* magnet_;
  std::unique_ptr<StrayFieldExecutor> executor_;
};
