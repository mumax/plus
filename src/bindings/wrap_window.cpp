#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <memory>

#include "window.hpp"
#include "wrappers.hpp"

void wrap_window(py::module& m) {

    py::class_<Window>(m, "Window")
        .def(py::init<>())
        .def("insert_magnetization", [](Window& self, int side, real3 value) {
            self.setMagValue(static_cast<Boundary>(side), value);
        }, py::arg("side"), py::arg("value"))

        .def("insert_geometry", [](Window& self, int side, bool value) {
            self.setGeoValue(static_cast<Boundary>(side), value);
        }, py::arg("side"), py::arg("value"))

        .def("insert_region_index", [](Window& self, int side, unsigned int value) {
            self.setRegValue(static_cast<Boundary>(side), value);
        }, py::arg("side"), py::arg("value"))

        .def("total_shift", &Window::GetTotalShift)
        .def("velocity", &Window::velocity);
}