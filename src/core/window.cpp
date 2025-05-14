#include "shift.hpp"
#include "timesolver.hpp"
#include "window.hpp"

#include <type_traits>

Window::Window() {
        magValues_.fill({0, 0, 0});
        geoValues_.fill(false);
        regValues_.fill(0);
      }


template <typename T>
GpuBuffer<T> Window::centerOnExcitation(const GpuBuffer<T>& data, int dir, int ncells, int nx, int ny) const {
    // Determine boundaries based on dir
    // TODO: is this right?
    T left, right;
    Boundary leftSide, rightSide;
    switch (dir) {
        case -1:
            leftSide = Boundary::Left;
            rightSide = Boundary::Right;
            break;
        case 1:
            leftSide = Boundary::Right;
            rightSide = Boundary::Left;
            break;
        default:
            throw std::invalid_argument("centerOnExcitation: direction must be ±1.");
    }

    if constexpr (std::is_same<T, bool>::value) {
        left = geoValues_[idx(leftSide)];
        right = geoValues_[idx(rightSide)];
    }
    else if constexpr (std::is_same<T, unsigned int>::value) {
        left = regValues_[idx(leftSide)];
        right = regValues_[idx(rightSide)];
    }
    else {
        throw std::invalid_argument("Unsupported type in centerOnExcitation. "
                                    "Supported types are bool and unsigned int.");
    }
    
    return shift(data, dir, ncells, nx, ny, left, right);
}
// Explicit instantiations
template GpuBuffer<bool> Window::centerOnExcitation<bool>(const GpuBuffer<bool>&, int, int, int, int) const;
template GpuBuffer<unsigned int> Window::centerOnExcitation<unsigned int>(const GpuBuffer<unsigned int>&, int, int, int, int) const;


Field Window::centerOnExcitation(const Field& field, int dir, int comp) {
    // TODO: this is only x
    real cs = field.world()->cellsize().x;
    ext_pos_ += cs;
    ext_vel_ = cs / field.world()->timesolver().timestep();
    // TODO: this "comp+1" doesn't seem right (because it's not)
    return shift(field, dir, comp, magValues_[comp], magValues_[comp + 1]);
}

real Window::getDWPositionX(const Field& field) const {
    real av = field.average()[0];
    real cs = field.world()->cellsize().x;
    int n = field.grid().size().x;
    return GetTotalShift() + av * cs * n / 2;
}