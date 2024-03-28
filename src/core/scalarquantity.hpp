#pragma once

#include <string>

#include "datatypes.hpp"

class ScalarQuantity {
 public:
  virtual ~ScalarQuantity();
  virtual real eval() const = 0;
  virtual bool sub2() const;
  virtual std::string name() const;
  virtual std::string unit() const;
};
