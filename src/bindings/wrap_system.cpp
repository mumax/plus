#include <memory>

#include "system.hpp"
#include "wrappers.hpp"

// Helper function to create a py::array with capsule-based memory management
// TODO: find common ground with fieldToArray
template <typename T>
py::array_t<T> create_py_array(const System* system, const T* data, int3 size) {
    int shape[3] = {size.z, size.y, size.x};
    int strides[3];
    strides[0] = sizeof(T) * size.x * size.y;
    strides[1] = sizeof(T) * size.x;
    strides[2] = sizeof(T);

    // Create Python capsule to manage memory cleanup
    py::capsule free_when_done(data, [](void* p) {
        T* arr = reinterpret_cast<T*>(p);
        delete[] arr;
    });
    return py::array_t<T>(shape, strides, data, free_when_done);
}

template <typename T, typename Getter>
py::array_t<T> get_data(const System* system, Getter getter, T default_value) {
    T* data;
    int ncells = system->grid().ncells();

    auto system_data = (system->*getter)();
    if (system_data.size() == 0) {
        data = new T[ncells];
        for (int i = 0; i < ncells; ++i) {
            data[i] = default_value;
        }
    } else
        data = system_data.getHostCopy();
    return create_py_array(system, data, system->grid().size());
}

void wrap_system(py::module& m) {
  py::class_<System, std::shared_ptr<System>>(m, "System")
      .def_property_readonly("grid", &System::grid)
      .def_property_readonly("cellsize", &System::cellsize)
      .def("cell_position", &System::cellPosition)
      .def_property_readonly("origin", &System::origin)
      .def_property_readonly("center", &System::center)
      .def_property_readonly("geometry", [](const System* system) {
          return get_data<bool>(system, &System::geometry, true); })
      .def_property_readonly("regions", [](const System* system) {
          return get_data<unsigned int>(system, &System::regions, 0); });
}
