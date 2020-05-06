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
        std::vector<Field*> images(nImages, nullptr);

        for (int i = 0; i < nImages; i++) {
          images[i] = new Field(magnet->grid(), 3);
          setArrayInField(images[i], py_imagelist[i].cast<py::array_t<real>>());
        }

        std::unique_ptr<ElasticBand> eband(new ElasticBand(magnet, images));

        for (auto image : images)
          delete image;

        return eband;
      }))
      .def("select_image", &ElasticBand::selectImage)
      .def("solve", &ElasticBand::solve)
      .def("step", &ElasticBand::step)
      .def("n_images", &ElasticBand::nImages)
      .def("relax_endpoints", &ElasticBand::relaxEndPoints);
}