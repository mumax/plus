#include "shift.hpp"
#include "timesolver.hpp"
#include "window.hpp"

Window::Window()
    : ext_pos_(real3{0, 0, 0}),
      ext_vel_(real3{0, 0, 0}),
      total_dist_(real3{0, 0, 0}) {
        magValues_.fill({0, 0, 0});
    }

Field Window::centerOnExcitation(const Field& field, int dir, int axis, int comp) {
    real3 cellsize = field.world()->cellsize();
    real cs = (&cellsize.x)[axis];

    // Multiply dir by -1 because the field is moved to dir iff the window is moved to -dir
    (&ext_pos_.x)[axis] += -1. * dir * cs;
    (&ext_vel_.x)[axis] = -1. * dir * cs / field.world()->timesolver().timestep();
    (&total_dist_.x)[axis] += cs;
    return shift(field, dir, axis, comp, magValues_[0], magValues_[1]);
}