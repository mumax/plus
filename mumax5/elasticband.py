"""Implementation of the elastic band approach to find the minimal energy path."""

import _mumax5cpp as _cpp


class ElasticBand:
    """Elastic band."""

    def __init__(self, ferromagnet, images):
        """Construct an elastic band."""
        self._impl = _cpp.ElasticBand(ferromagnet._impl, images)

    @property
    def n_images(self):
        """Return the number of images in the elastic band."""
        return self._impl.n_images()

    @property
    def spring_constant(self):
        """Spring constant of the elastic band."""
        return self._impl.spring_constant

    @spring_constant.setter
    def spring_constant(self, value):
        self._impl.spring_constant = value

    def relax_endpoints(self):
        """Relax the endpoints of the elastic band."""
        return self._impl.relax_endpoints()

    def step(self, dt):
        """Take a single Euler step to relax the elastic band."""
        return self._impl.step(dt)

    def select_image(self, idx):
        """Select an image to be the actual magnetization of the underlying magnet."""
        return self._impl.select_image(idx)

    def geodesic_distance_images(self, image_idx1, image_idx2):
        """Return the geodesic distance between two images if the elastic band."""
        return self._impl.geodesic_distance_images(image_idx1, image_idx2)
