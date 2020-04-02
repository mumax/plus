#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "cast.hpp"

namespace py = pybind11;

class Field;
py::array_t<real> fieldToArray(const Field*);
void setArrayInField(Field*, py::array_t<real>);

void wrap_ferromagnet(py::module& m);
void wrap_field(py::module& m);
void wrap_grid(py::module& m);
void wrap_quantity(py::module& m);
void wrap_timesolver(py::module& m);
void wrap_variable(py::module& m);
void wrap_world(py::module& m);