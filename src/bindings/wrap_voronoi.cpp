#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <memory>

#include "grid.hpp"
#include "voronoi.hpp"
#include "wrappers.hpp"


void wrap_voronoi(py::module& m) {
    py::class_<VoronoiTessellator>(m, "VoronoiTessellator")

        .def(py::init<real, unsigned int, int>(),
                py::arg("grainsize"),
                py::arg("max_idx"),
                py::arg("seed"))
        .def("coo_to_idx", &VoronoiTessellator::regionOf)
        // TODO: create template function (wrap_system.cpp)
        .def("generate", [](VoronoiTessellator& t, Grid grid, real3 cellsize, const bool pbc) {
            unsigned int* tess = t.generate(grid, cellsize, pbc).getHostCopy();

            // Create python capsule which will free tess
            py::capsule free_when_done(tess, [](void* p) {
            unsigned int* tess = reinterpret_cast<unsigned int*>(p);
            delete[] tess;
            });

            int3 size = grid.size();
            int shape[3] = {size.z, size.y, size.x};
            int strides[3];
            strides[0] = sizeof(unsigned int) * size.x * size.y;
            strides[1] = sizeof(unsigned int) * size.x;
            strides[2] = sizeof(unsigned int);

            return py::array_t<unsigned int>(shape, strides, tess, free_when_done);
            });
}