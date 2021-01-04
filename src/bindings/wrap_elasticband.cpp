#include <memory>
#include <vector>

#include "elasticband.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "wrappers.hpp"

void wrap_elasticband(py::module& m) {
  py::class_<ElasticBand>(m, "ElasticBand")
      .def(py::init([](Ferromagnet* magnet, py::list py_imagelist) {
        int nImages = py_imagelist.size();
        std::vector<Field> images(nImages);
        for (int i = 0; i < nImages; i++) {
          images[i] = Field(magnet->system(), 3);
          setArrayInField(images[i], py_imagelist[i].cast<py::array_t<real>>());
        }
        std::unique_ptr<ElasticBand> eband(new ElasticBand(magnet, images));
        return eband;
      }))
      .def("select_image", &ElasticBand::selectImage)
      .def("step", &ElasticBand::step)
      .def("n_images", &ElasticBand::nImages)
      .def("relax_endpoints", &ElasticBand::relaxEndPoints)
      .def("geodesic_distance_images", &ElasticBand::geodesicDistanceImages);
}
