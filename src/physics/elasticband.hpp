#pragma once

#include <memory>
#include <vector>

#include "datatypes.hpp"

class Ferromagnet;
class Field;

/** An elastic band can be used to compute the minimal energy path (MEP)
 *  between two stable states sitting in local minima of the energy landscape.
 *
 *  The elastic band consist out of a finite number of images. The images are
 *  updated stepwise (using a Euler stepper) to 'span the elastic band' and
 *  converge towards the MEP. The first image and the last image are fixed
 *  during this optimization scheme.
 *
 *  The elastic band implementation here is based on the work of Bessarab et al:
 *
 *     Computer Physics Communication 196, 335–347 (2015).
 *     https://doi.org/10.1016/j.cpc.2015.07.001
 */
class ElasticBand {
 public:
  /** Construct an elastic band for a ferromagnet from a series of magnetization
   *  images.
   */
  ElasticBand(Ferromagnet*, const std::vector<Field>& images);

  /** Return the number of images in the elastic band. */
  int nImages() const;

  /** Relax the first and last image of the elastic band. */
  void relaxEndPoints();

  /** Make a single Euler step with a given stepsize towards the MEP. */
  void step(real stepsize);

  /** Set the magnetization of the ferromagnet to be one of the images. */
  void selectImage(int imageIdx);

  /** Return the geodesic distance between two images. */
  real geodesicDistanceImages(int, int) const;

  /** Set the spring constant which is used to compute the spring forces. */
  void setSpringConstant(real);

  /** Return the spring constant value used to compute the spring forces. */
  real springConstant() const;

  /** Returns the energy of each image. */
  std::vector<real> energies();

  /** Return the total force on each image.
   *  @see Eq. 17 in Comp. Phys. Comm. 196, 335–347 (2015).
   */
  std::vector<Field> forceFields();

  /** Return the spring force on each image. */
  std::vector<real> springForces() const;

 private:
  std::vector<Field> images_;
  std::vector<Field> velocities_;
  Ferromagnet* magnet_;
  real springConstant_;
};
