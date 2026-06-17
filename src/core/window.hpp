#pragma once

#include "datatypes.hpp"
#include "field.hpp"
#include "parameter.hpp"

#include <vector>

enum class Boundary {
  Left,
  Right
};

class Window {
  public:
   Window();
   ~Window() = default;

   // Set values to be inserted at the boundaries
   void setMagValue(Boundary side, real3 value) { magValues_[idx(side)] = value; }
   // Get values to be inserted at the boundaries
   std::array<real3, 2> getMagValues() { return magValues_; }

   Field centerOnExcitation(const Field& field, int dir, int axis, int comp);

   // Get total amount shifted
   real position() const { return ext_pos_; }
   real velocity() const { return ext_vel_; }
   real GetTotalShift() const { return total_dist_; }

  private:
   // Values to insert at the boundaries
   std::array<real3, 2> magValues_;


   // Current excitation position and velocity
   real ext_pos_;
   real ext_vel_;

   // Total distance travelled by excitation
   real total_dist_;

   static constexpr size_t idx(Boundary b) { return static_cast<size_t>(b); }
};