#pragma once

#include <memory>
#include <string>

#include "fieldquantity.hpp"

class Parameter;
class Magnet;
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
      const Magnet* magnet,
      std::shared_ptr<const System> system,
      Method method, int order, double eps, double switchingradius);

 protected:
  /** Constructor only to be used in constructor of derived classes. */
  StrayFieldExecutor(const Magnet* magnet,
                     std::shared_ptr<const System> system);

 public:
  /** Empty virtual destructor. */
  virtual ~StrayFieldExecutor() {}

  /** Compute and return the stray field. */
  virtual Field exec() const = 0;

  /** Return the method of the executor. */
  virtual Method method() const = 0;

  /** Return the order of the executor. */
  virtual int order() const = 0;

  /** Return epsilon. The parameter used to determine the analytical error
   * using epsilon * R³/V
   */
  virtual double eps() const = 0;

  /** Return the switching radius of the executor. */
  virtual double switchingradius() const = 0;

 protected:
  /** Source of the stray field*/
  const Magnet* magnet_;

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
  StrayField(const Magnet* magnet,
             std::shared_ptr<const System> system,
             StrayFieldExecutor::Method = StrayFieldExecutor::METHOD_AUTO,
             int order = 11,
             double eps = 5e-10,
             double switchingradius = -1);

  /**
   * Constructor of a StrayField on a specified grid.
   *
   * Internally, a new system will be created (from the specified grid) in which
   * the stray field will be computed.
   * @param magnet the source of the stray field
   * @param grid   used to create a system in which the stray field is computed
   * @param method the used method for the computation
   */
  StrayField(const Magnet* magnet,
             Grid grid,
             StrayFieldExecutor::Method = StrayFieldExecutor::METHOD_AUTO,
             int order = 11,
             double eps = 5e-10,
             double switchingradius = -1);

  /** Destructor. */
  ~StrayField();

  /** Set the method for the computation of the stray field. */
  void setMethod(StrayFieldExecutor::Method);

  /** Set the order for the asymptotic computation of the stray field. */
  int order() const {return executor_->order();}
  void setOrder(int);

  /** Set epsilon to determine the error of the analytical method. */
  double eps() const {return executor_->eps();}
  void setEps(double);

  /** Set the radius from which the asymptotic expansion should be used. */
  double switchingradius() const {return executor_->switchingradius();}
  void setSwitchingradius(double);

  /** Recreate the StrayFieldExecutor. */
  void recreateStrayFieldExecutor();

  /** Return the magnet which is the source of the stray field. */
  const Magnet* source() const;

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
  const Magnet* magnet_;
  std::unique_ptr<StrayFieldExecutor> executor_;
};
