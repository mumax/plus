#include <memory>

#include "system.hpp"
#include "wrappers.hpp"

void wrap_system(py::module& m) {
  py::class_<System, std::shared_ptr<System>>(m, "System")
      .def_property_readonly("grid", &System::grid)
      .def_property_readonly("cellsize", &System::cellsize)
      .def("cell_position", &System::cellPosition)
      .def_property_readonly("center", &System::center)
      .def_property_readonly("geometry", [](const System* system) {
        bool* geometry;
        if (system->geometry().size() == 0) {
          int ncells = system->grid().ncells();
          geometry = new bool[ncells];
          for (int i = 0; i < ncells; i++) {
            geometry[i] = true;
          }
        } else {
          geometry = system->geometry().getHostCopy();
        }

        // Create python capsule which will free geometry
        py::capsule free_when_done(geometry, [](void* p) {
          bool* geometry = reinterpret_cast<bool*>(p);
          delete[] geometry;
        });

        int3 size = system->grid().size();
        int shape[3] = {size.z, size.y, size.x};
        int strides[3];
        strides[0] = sizeof(bool) * size.x * size.y;
        strides[1] = sizeof(bool) * size.x;
        strides[2] = sizeof(bool);

        return py::array_t<bool>(shape, strides, geometry, free_when_done);
      });
}
