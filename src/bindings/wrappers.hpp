#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "cast.hpp"

namespace py = pybind11;

class Field;
py::array_t<real> fieldToArray(const Field&);
void setArrayInField(Field&, py::array_t<real>);

void wrap_ferromagnet(py::module& m);
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
void wrap_system(py::module& m);
void wrap_dmitensor(py::module& m);
