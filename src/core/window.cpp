#include "shift.hpp"
#include "window.hpp"

#include <type_traits>

Window::Window() {
        magValues_.fill({0, 0, 0});
        geoValues_.fill(false);
        regValues_.fill(0);
      }




template <typename T>
GpuBuffer<T> Window::centerOnExcitation(const GpuBuffer<T>& data, int dir, int ncells) const {
    // Determine boundaries based on dir
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
    } else if constexpr (std::is_same<T, unsigned int>::value) {
        left = regValues_[idx(leftSide)];
        right = regValues_[idx(rightSide)];
    } else {
        throw std::invalid_argument("Unsupported type in centerOnExcitation. "
                                    "Supported types are bool and unsigned int.");
    }
    
    return shift(data, dir, ncells, left, right);
}
// Explicit instantiations
template GpuBuffer<bool> Window::centerOnExcitation<bool>(const GpuBuffer<bool>&, int, int) const;
template GpuBuffer<unsigned int> Window::centerOnExcitation<unsigned int>(const GpuBuffer<unsigned int>&, int, int) const;


Field Window::centerOnExcitation(const Field& field, int dir, int comp) const {
    return shift(field, dir, comp, magValues_[comp], magValues_[comp + 1]);
}