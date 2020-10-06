import _mumax5cpp as _cpp


class ElasticBand:
    """ Elastic band
    """

    def __init__(self, ferromagnet, images):
        """ Construct an elastic band """
        self._impl = _cpp.ElasticBand(ferromagnet._impl, images)

    @property
    def n_images(self):
        """ The number of images in the elastic band
        """
        return self._impl.n_images()

    def relax_endpoints(self):
        return self._impl.relax_endpoints()

    def step(self, dt):
        return self._impl.step(dt)

    def select_image(self, idx):
        return self._impl.select_image(idx)

    def geodesic_distance_images(self, image_idx1, image_idx2):
        return self._impl.geodesic_distance_images(image_idx1, image_idx2)

