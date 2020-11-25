#include "fieldquantity.hpp"

#include <stdexcept>
#include <vector>

#include "datatypes.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "reduce.hpp"
#include "system.hpp"

Grid FieldQuantity::grid() const {
  return system()->grid();
}

void FieldQuantity::addToField(Field& f) const {
  if (!sameFieldDimensions(*this, f))
    throw std::invalid_argument(
        "Can not add the quantity to given field because the fields are "
        "incompatible.");
  if (assuredZero())
    return;
  f += eval();
}

std::vector<real> FieldQuantity::average() const {
  return fieldAverage(eval());
}