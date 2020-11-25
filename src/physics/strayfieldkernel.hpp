#pragma once

#include <vector>

#include "datatypes.hpp"
#include "grid.hpp"

class Field;
class System;

class StrayFieldKernel {
 public:
  StrayFieldKernel(Grid grid, real3 cellsize);
  StrayFieldKernel(Grid dst, Grid src, real3 cellsize);
  ~StrayFieldKernel();

  Grid grid() const;
  std::shared_ptr<const System> kernelSystem() const;
  real3 cellsize() const;
  const Field& field() const;

  void compute();

 private:
  std::shared_ptr<System> kernelSystem_;
  real3 cellsize_;
  Field* kernel_;

 public:
  // Helper function which determines the kernel grid
  static Grid kernelGrid(Grid dst, Grid src);
};