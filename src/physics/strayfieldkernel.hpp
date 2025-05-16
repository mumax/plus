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
  StrayFieldKernel(Grid grid, const World* world, int order, double eps, double switchingradius);
  StrayFieldKernel(Grid dst, Grid src, const World* world, int order, double eps, double switchingradius);
  ~StrayFieldKernel();

  Grid grid() const;
  Grid mastergrid() const;
  const int3 pbcRepetitions() const;
  std::shared_ptr<const System> kernelSystem() const;
  real3 cellsize() const;
  int order() const;
  double eps() const;
  double switchingradius() const;
  const Field& field() const;

  void compute();

 private:
  std::unique_ptr<Field> kernel_;
  int order_;
  double eps_;
  double switchingradius_;

 public:
  // Helper function which determines the kernel grid
  static Grid kernelGrid(Grid dst, Grid src);
};
