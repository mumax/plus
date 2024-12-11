#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <memory>

#include "grid.hpp"
#include "voronoi.hpp"
#include "wrappers.hpp"


void wrap_voronoi(py::module& m) {
    py::class_<VoronoiTessellator>(m, "VoronoiTessellator")

        .def(py::init<Grid, real, real3>(), py::arg("grid"), py::arg("grainsize"), py::arg("cellsize"))
        // TODO: create template function (wrap_system.cpp)
        .def_property_readonly("tessellation", [](VoronoiTessellator t) {
            unsigned int* tess = t.generate().getHostCopy();

            // Create python capsule which will free tess
            py::capsule free_when_done(tess, [](void* p) {
            unsigned int* tess = reinterpret_cast<unsigned int*>(p);
            delete[] tess;
            });

            int3 size = t.grid.size();
            int shape[3] = {size.z, size.y, size.x};
            int strides[3];
            strides[0] = sizeof(unsigned int) * size.x * size.y;
            strides[1] = sizeof(unsigned int) * size.x;
            strides[2] = sizeof(unsigned int);

            return py::array_t<unsigned int>(shape, strides, tess, free_when_done);
            });
}