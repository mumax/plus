#include "window.hpp"
#include "wrappers.hpp"

void wrap_window(py::module& m) {

    py::class_<Window>(m, "Window")
        .def("insert_magnetization", [](Window& self, int side, real3 value) {
            self.setMagValue(static_cast<Boundary>(side), value);
        }, py::arg("side"), py::arg("value"))

        .def_property_readonly("position", &Window::position)
        .def_property_readonly("velocity", &Window::velocity)
        .def_property_readonly("total_shift", &Window::totalShift);

    }