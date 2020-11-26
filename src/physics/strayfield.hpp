#pragma once

#include <memory>

#include "fieldquantity.hpp"

class Parameter;
class Ferromagnet;
class Field;
class System;
class Grid;

class StrayFieldExecutor {
 public:
  enum Method { METHOD_BRUTE, METHOD_FFT, METHOD_AUTO };

  /** Factory method for StrayFieldExecutor */
  static std::unique_ptr<StrayFieldExecutor> create(
      std::shared_ptr<const System> inSystem,
      std::shared_ptr<const System> outSystem,
      Method method);

 public:
   virtual ~StrayFieldExecutor() {}
  virtual void exec(Field* h, const Field* m, const Parameter* msat) const = 0;
  virtual Method method() const = 0;
};

/// StrayField is a field quantity which evaluates the stray field of a
/// ferromagnet on a specified system.
/// Note: if the specified grid matches the grid of the ferromagnet, then this
/// stray field is the demagnetization field of the magnet
class StrayField : public FieldQuantity {
 public:
  StrayField(const Ferromagnet* magnet,
             std::shared_ptr<const System> system,
             StrayFieldExecutor::Method = StrayFieldExecutor::METHOD_AUTO);

  StrayField(const Ferromagnet* magnet,
             Grid grid,
             StrayFieldExecutor::Method = StrayFieldExecutor::METHOD_AUTO);

  ~StrayField();

  void setMethod(StrayFieldExecutor::Method);

  const Ferromagnet* source() const;

  int ncomp() const;
  std::shared_ptr<const System> system() const;
  std::string unit() const;
  Field eval() const;

  bool assuredZero() const;

 private:
  std::shared_ptr<const System> system_;
  const Ferromagnet* magnet_;
  std::unique_ptr<StrayFieldExecutor> executor_;
};
