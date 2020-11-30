#pragma once

#include <memory>
#include <vector>

#include "datatypes.hpp"
#include "grid.hpp"

class Field;
class System;
class World;

class StrayFieldKernel {
 public:
  StrayFieldKernel(Grid grid, const World* world);
  StrayFieldKernel(Grid dst, Grid src, const World* world);
  ~StrayFieldKernel();

  Grid grid() const;
  std::shared_ptr<const System> kernelSystem() const;
  real3 cellsize() const;
  const Field& field() const;

  void compute();

 private:
  std::unique_ptr<Field> kernel_;

 public:
  // Helper function which determines the kernel grid
  static Grid kernelGrid(Grid dst, Grid src);
};
