#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include <memory>

#include "mfm.hpp"
#include "wrappers.hpp"

void wrap_mfm(py::module& m) {

    py::class_<MFM, FieldQuantity>(m, "MFM")
        .def_property_readonly("name", &MFM::name)
        .def_property_readonly("unit", &MFM::unit)

        .def(py::init<Magnet*, const Grid>(),
             py::arg("magnet"),
             py::arg("grid"))
        
        .def(py::init<const MumaxWorld*, const Grid>(),
            py::arg("mumaxworld"),
            py::arg("grid"))

        .def_readwrite("lift", &MFM::lift)
        .def_readwrite("tipsize", &MFM::tipsize);
}
