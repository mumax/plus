#include "scalarquantity.hpp"

ScalarQuantity::~ScalarQuantity() {}

int ScalarQuantity::lattice() const {
  return 0;
}

std::string ScalarQuantity::unit() const {
  return "";
}

std::string ScalarQuantity::name() const {
  return "";
}