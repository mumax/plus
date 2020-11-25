#pragma once

#include <memory>

#include "fieldquantity.hpp"
#include "grid.hpp"

enum MagnetFieldComputationMethod {
  MAGNETFIELDMETHOD_BRUTE,
  MAGNETFIELDMETHOD_FFT,
  MAGNETFIELDMETHOD_AUTO
};

class Parameter;
class Ferromagnet;
class Field;
class System;

class MagnetFieldExecutor {
 public:
  virtual ~MagnetFieldExecutor() {}
  virtual void exec(Field* h, const Field* m, const Parameter* msat) const = 0;
};

/// MagnetField is a field quantity which evaluates the magnetic field of a
/// ferromagnet on a specified system.
/// Note: if the specified grid matches the grid of the ferromagnet, then this
/// magnetic field is the demagnetization field of the magnet
class MagnetField : public FieldQuantity {
 public:
  MagnetField(const Ferromagnet* magnet,
              std::shared_ptr<const System> system,
              MagnetFieldComputationMethod method = MAGNETFIELDMETHOD_AUTO);

  MagnetField(const Ferromagnet* magnet,
              Grid grid,
              MagnetFieldComputationMethod method = MAGNETFIELDMETHOD_AUTO);

  ~MagnetField();

  void setMethod(MagnetFieldComputationMethod);

  const Ferromagnet* source() const;

  int ncomp() const;
  std::shared_ptr<const System> system() const;
  std::string unit() const;
  Field eval() const;

  bool assuredZero() const;

 private:
  std::shared_ptr<const System> system_;
  const Ferromagnet* magnet_;
  MagnetFieldExecutor* executor_;
};
