#pragma once

#include <memory>
#include <string>

#include "fieldquantity.hpp"
#include "grid.hpp"

class CuField;
class System;

enum class MaxError {
  MAGNETIZATION,
  DISPLACEMENT,
  VELOCITY
};

class Variable : public FieldQuantity {
 public:
  Variable(std::shared_ptr<const System> system, int ncomp,
           std::string name = "", std::string unit = "",
           MaxError maxError = MaxError::MAGNETIZATION);
  ~Variable();

  int ncomp() const;
  std::shared_ptr<const System> system() const;
  std::string name() const;
  std::string unit() const;
  MaxError maxError() const;

  Field eval() const;

  const Field& field() const;

  virtual void set(const Field&) const;
  virtual void set(real) const;
  virtual void set(real3) const;

  // Assignment operators which call the respective set function
  void operator=(const Field& f) const { set(f); }
  void operator=(real val) const { set(val); }
  void operator=(real3 val) const { set(val); }

 protected:
  Field* field_;

 private:
  std::string name_;
  std::string unit_;
  MaxError maxError_;  // enum to select appropriate max error for adaptive time stepping
};

// Exactly the same as variable, but when values are set, the values are
// normalized
class NormalizedVariable : public Variable {
 public:
  NormalizedVariable(std::shared_ptr<const System> system, int ncomp,
                     std::string name = "", std::string unit = "",
                     MaxError maxError = MaxError::MAGNETIZATION);
  void set(const Field&) const;
  void set(real) const;
  void set(real3) const;
};
