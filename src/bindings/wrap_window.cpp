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

        .def("position", &Window::position)
        .def("velocity", &Window::velocity)
        .def("total_shift", &Window::GetTotalShift);

    }