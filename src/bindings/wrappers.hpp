#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

#include "cast.hpp"
#include "fieldToArray.hpp" // Seperate header file for template declaration

namespace py = pybind11;

class Field;

void setArrayInField(Field&, py::array_t<real>);

void wrap_antiferromagnet(py::module& m);
void wrap_antiferromagnetfieldquantity(py::module& m);
void wrap_antiferromagnetscalarquantity(py::module& m);
void wrap_ferromagnet(py::module& m);
void wrap_ncafm(py::module& m);
void wrap_ncafmfieldquantity(py::module& m);
void wrap_ncafmscalarquantity(py::module& m);
void wrap_magnet(py::module& m);
void wrap_magnetfieldquantity(py::module& m);
void wrap_magnetscalarquantity(py::module& m);
void wrap_mfm(py::module& m);
void wrap_field(py::module& m);
void wrap_grid(py::module& m);
void wrap_parameter(py::module& m);
void wrap_fieldquantity(py::module& m);
void wrap_scalarquantity(py::module& m);
void wrap_timesolver(py::module& m);
void wrap_variable(py::module& m);
void wrap_world(py::module& m);
void wrap_strayfield(py::module& m);
void wrap_ferromagnetfieldquantity(py::module& m);
void wrap_ferromagnetscalarquantity(py::module& m);
void wrap_poissonsolver(py::module& m);
void wrap_linsolver(py::module& m);
void wrap_system(py::module& m);
void wrap_dmitensor(py::module& m);
void wrap_voronoi(py::module& m);
void wrap_traction(py::module& m);