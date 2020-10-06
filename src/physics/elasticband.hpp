#pragma once

#include <memory>
#include <vector>

#include "datatypes.hpp"
#include "field.hpp"

class Ferromagnet;

class ElasticBand {
 public:
  ElasticBand(Ferromagnet*, const std::vector<Field>& images);
  int nImages() const { return images_.size(); }
  void relaxEndPoints();
  // void solve();
  void step(real);
  void selectImage(int);
  real geodesicDistanceImages(int, int);
  std::vector<real> energies();
  std::vector<Field> perpendicularForces();

 private:
  std::vector<Field> images_;
  std::vector<Field> velocities_;
  Ferromagnet* magnet_;
};