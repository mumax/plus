#pragma once

#include <memory>
#include <vector>

#include"datatypes.hpp"

class Ferromagnet;
class Field;

class ElasticBand {
 public:
  ElasticBand(Ferromagnet*, std::vector<Field*> images);
  int nImages() const;
  void relaxEndPoints();
  void solve();
  void step(real);
  void selectImage(int);

 private:
  std::vector<std::unique_ptr<Field>> images_;
  std::vector<std::unique_ptr<Field>> velocities_;
  std::vector<std::unique_ptr<Field>> forces_;
  Ferromagnet* magnet_;
};