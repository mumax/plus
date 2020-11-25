#pragma once

#include <vector>

#include "datatypes.hpp"
#include "grid.hpp"

class Field;
class System;

class MagnetFieldKernel {
 public:
  MagnetFieldKernel(Grid grid, real3 cellsize);
  MagnetFieldKernel(Grid dst, Grid src, real3 cellsize);
  ~MagnetFieldKernel();

  Grid grid() const;
  System* kernelSystem();
  real3 cellsize() const;
  const Field& field() const;

  void compute();

 private:
  std::unique_ptr<System> kernelSystem_;
  real3 cellsize_;
  Field* kernel_;

 public:
  // Helper function which determines the kernel grid
  static Grid kernelGrid(Grid dst, Grid src);
};