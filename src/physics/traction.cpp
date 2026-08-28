#include "traction.hpp"

#include "parameter.hpp"
#include "system.hpp"

BoundaryTraction::BoundaryTraction(std::shared_ptr<const System> system, std::string name)
    : posXside(system, {0.,0.,0.}, name + ":pos_x_side", "Pa"),
      negXside(system, {0.,0.,0.}, name + ":neg_x_side", "Pa"),
      posYside(system, {0.,0.,0.}, name + ":pos_y_side", "Pa"),
      negYside(system, {0.,0.,0.}, name + ":neg_y_side", "Pa"),
      posZside(system, {0.,0.,0.}, name + ":pos_z_side", "Pa"),
      negZside(system, {0.,0.,0.}, name + ":neg_z_side", "Pa") {}

CuBoundaryTraction BoundaryTraction::cu() const {
  return CuBoundaryTraction{
      posXside.cu(), negXside.cu(), 
      posYside.cu(), negYside.cu(), 
      posZside.cu(), negZside.cu()
  };
}

bool BoundaryTraction::assuredZero() const {
  return posXside.assuredZero() && negXside.assuredZero() &&
         posYside.assuredZero() && negYside.assuredZero() &&
         posZside.assuredZero() && negZside.assuredZero();
}
