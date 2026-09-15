#pragma once

#include "datatypes.hpp"
#include "field.hpp"

#include <array>
#include <vector>

enum class Boundary {
  Left,
  Right
};

class MumaxWorld;
class Window {
  public:
   explicit Window(MumaxWorld& world);
   ~Window() = default;

   // Set origin of the simulation window
   void setOrigin(real3 origin) { origin_ = origin; }
   // Set values to be inserted at the boundaries
   void setMagValue(Boundary side, real3 value) { magValues_[idx(side)] = value; }
   // Get values to be inserted at the boundaries
   std::array<real3, 2> getMagValues() { return magValues_; }

   void move(int dir, int axis, int comp);
   Field centerOnExcitation(const Field& field, int dir, int axis, int comp);
   void disableMotion();

   // Get total amount shifted
   real3 position() const { return origin_ + position_; }
   real3 velocity() const { return velocity_; }
   real3 totalShift() const { return total_dist_; }

  private:
   // Keep reference of the world to which this window belongs
   MumaxWorld& world_;
   real3 origin_;
   // Values to insert at the boundaries
   std::array<real3, 2> magValues_;
   // Current window position and velocity
   real3 position_;
   real3 velocity_;
   // Total distance travelled by window
   real3 total_dist_;

   static constexpr size_t idx(Boundary b) { return static_cast<size_t>(b); }
};