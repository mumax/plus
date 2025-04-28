#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <memory>

#include "grid.hpp"
#include "voronoi.hpp"
#include "wrappers.hpp"


void wrap_voronoi(py::module& m) {
    py::class_<VoronoiTessellator>(m, "VoronoiTessellator")


        .def(py::init<real, int, unsigned int, std::function<unsigned int(real3)>>(),
                py::arg("grainsize"),
                py::arg("seed"),
                py::arg("max_idx"),
                py::arg("region_of_center"))
        .def("coo_to_idx", &VoronoiTessellator::regionOf)
        // TODO: create template function (wrap_system.cpp)
        .def("generate", [](VoronoiTessellator& t, Grid grid, real3 cellsize, const bool pbc) {
            std::vector<unsigned int> tess = t.generate(grid, cellsize, pbc);

            size_t n = tess.size();
            unsigned int* raw = new unsigned int[n];
            std::memcpy(raw, tess.data(), n * sizeof(unsigned int));

            // Create python capsule which will free tess
            py::capsule free_when_done(raw, [](void* p) {
                delete[] reinterpret_cast<unsigned int*>(p);
            });

            int3 size = grid.size();
            ssize_t shape[3] = {size.z, size.y, size.x};
            ssize_t strides[3] = {
                static_cast<ssize_t>(sizeof(unsigned int)) * size.x * size.y,
                static_cast<ssize_t>(sizeof(unsigned int)) * size.x,
                static_cast<ssize_t>(sizeof(unsigned int))
            };
            return py::array_t<unsigned int>(shape, strides, raw, free_when_done);
        });
}