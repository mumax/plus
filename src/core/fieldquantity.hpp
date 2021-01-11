#pragma once

#include <memory>
#include <string>
#include <vector>

#include "datatypes.hpp"
#include "grid.hpp"

class Field;
class System;
class World;

/// FieldQuantity interface
class FieldQuantity {
 public:
  /// Virtual destructor which does nothing
  virtual ~FieldQuantity() {}

  /***** PURE VIRTUAL FUNCTIONS *****/

  /// Returns the number of components of the quantity. In most cases this
  /// would be either 1 (scalar field) or 3 (vector fields)
  virtual int ncomp() const = 0;

  /// Returns the system of the field quantity
  virtual std::shared_ptr<const System> system() const = 0;

  /// Evaluates the quantity, the returned Field is moved instead of copied
  virtual Field eval() const = 0;

  /***** NON-PURE VIRTUAL FUNCTIONS *****/

  /// Returns the unit of the quantity (empty string by default)
  virtual std::string unit() const { return ""; }

  /// Returns the name of the quantity (empty string by default)
  virtual std::string name() const { return ""; }

  /// Evaluates the quantity and add it to the given field
  virtual void addToField(Field&) const;

  /// Eval the quantity and return the average of each component
  virtual std::vector<real> average() const;

  /// If assuredZero() returns true, then addTo(field) doesn't add anything to
  /// the field. This function returns false, but can be overriden in derived
  /// classes for optimization. In this case, it is also recommended to check in
  /// eval() if the quantity is zero, for an early exit.
  virtual bool assuredZero() const { return false; }

  /***** NON VIRTUAL FUNCTIONS *****/

  /// Returns the grid of the underlying system
  Grid grid() const;

  /// Return the world in which the quantity lives.
  /// Return nullptr if the system is not attached to a World.
  const World* world() const;
};

inline bool sameFieldDimensions(const FieldQuantity& q1,
                                const FieldQuantity& q2) {
  return q1.grid() == q2.grid() && q1.ncomp() == q2.ncomp();
}
