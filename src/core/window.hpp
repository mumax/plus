#pragma once

#include "datatypes.hpp"
#include "field.hpp"

#include <vector>

enum class Boundary {
  Left,
  Right
};

class Window {
  public:
   Window();
   ~Window() = default;

   // Set origin of the simulation window
   void setOrigin(real3 origin) { ext_pos_ = origin; }
   // Set values to be inserted at the boundaries
   void setMagValue(Boundary side, real3 value) { magValues_[idx(side)] = value; }
   // Get values to be inserted at the boundaries
   std::array<real3, 2> getMagValues() { return magValues_; }

   Field centerOnExcitation(const Field& field, int dir, int axis, int comp);

   // Get total amount shifted
   real3 position() const { return ext_pos_; }
   real3 velocity() const { return ext_vel_; }
   real3 GetTotalShift() const { return total_dist_; }

  private:
   // Values to insert at the boundaries
   std::array<real3, 2> magValues_;


   // Current excitation position and velocity
   real3 ext_pos_;
   real3 ext_vel_;

   // Total distance travelled by excitation
   real3 total_dist_;

   static constexpr size_t idx(Boundary b) { return static_cast<size_t>(b); }
};