#pragma once

#include "datatypes.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "parameter.hpp"

#include <vector>

// TODO: generalize to 3D?
enum class Boundary {
  Left,
  Right,
  Top,
  Bottom
};

class Window {
  public:
   Window();
   ~Window() = default;

   // Set values to be inserted at the boundaries
   void setMagValue(Boundary side, real3 value) { magValues_[idx(side)] = value; }
   void setGeoValue(Boundary side, bool value) { geoValues_[idx(side)] = value; }
   void setRegValue(Boundary side, unsigned int value) { regValues_[idx(side)] = value; }
 
   template <typename T>
   GpuBuffer<T> centerOnExcitation(const GpuBuffer<T>& data, int dir, int ncells, int nx, int ny) const;
   Field centerOnExcitation(const Field& field, int dir, int comp=0);

   // Get DW position
   real getDWPositionX(const Field& field) const;


   // Get total amount shifted
   real GetTotalShift() const { return -ext_pos_; }
   real velocity() const { return ext_vel_; }

  private:
   // Values to insert at the boundaries
   std::array<real3, 4> magValues_;
   std::array<bool, 4> geoValues_;
   std::array<unsigned int, 4> regValues_;

   // Current excitation position and velocity
   real ext_pos_;
   real ext_vel_;

   // Total distance travelled by excitation
   real total_dist_;

   static constexpr size_t idx(Boundary b) { return static_cast<size_t>(b); }
};