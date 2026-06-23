#include "datatypes.hpp"
#include "mumaxworld.hpp"
#include "shift.hpp"
#include "timesolver.hpp"
#include "window.hpp"

Window::Window(MumaxWorld& world)
    : world_(world),
      position_(real3{0, 0, 0}),
      velocity_(real3{0, 0, 0}),
      total_dist_(real3{0, 0, 0}) {
        magValues_.fill({0, 0, 0});
    }

void Window::move(int dir, int axis, int comp) {
    real3 cs = world_.cellsize();
    // Multiply dir by -1 because the field is moved to dir iff the window is moved to -dir
    (&position_.x)[axis] += -1. * dir * (&cs.x)[axis];
    (&velocity_.x)[axis] = -1. * dir * (&cs.x)[axis] / world_.timesolver().timestep();
    (&total_dist_.x)[axis] += (&cs.x)[axis];
}
Field Window::centerOnExcitation(const Field& field, int dir, int axis, int comp) {
    return shift(field, dir, comp, axis, magValues_[0], magValues_[1]);
}