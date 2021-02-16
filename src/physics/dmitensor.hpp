#pragma once

#include <memory>
#include <vector>

#include "parameter.hpp"
#include "system.hpp"

class CuDmiTensor;

/** DmiTensor holds the 9 DMI parameters */
struct DmiTensor {
  Parameter xxy;
  Parameter xyz;
  Parameter xxz;
  Parameter yxy;
  Parameter yyz;
  Parameter yxz;
  Parameter zxy;
  Parameter zyz;
  Parameter zxz;

  /** Construct the DMI tensor for a given system */
  explicit DmiTensor(std::shared_ptr<const System> system);

  /** Return CuDmiTensor */
  CuDmiTensor cu() const;

  /** Returns true if all 9 DMI parameters are equal to zero. */
  bool assuredZero() const;
};

struct CuDmiTensor {
  CuParameter xxy;
  CuParameter xyz;
  CuParameter xxz;
  CuParameter yxy;
  CuParameter yyz;
  CuParameter yxz;
  CuParameter zxy;
  CuParameter zyz;
  CuParameter zxz;
};
