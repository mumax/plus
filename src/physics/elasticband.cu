#include <stdexcept>

#include "cudalaunch.hpp"
#include "datatypes.hpp"
#include "effectivefield.hpp"
#include "elasticband.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "reduce.hpp"
#include "scalarquantity.hpp"

/** ElasticBandSection holds the data needed to compute the force on a section
 *  of the elastic band according to the work of Bessarab et al:
 *
 *    Computer Physics Communication 196, 335–347 (2015).
 *    https://doi.org/10.1016/j.cpc.2015.07.001
 *
 *  ElasticBandSection is meant to be used only in the cuda kernel k_force which
 *  computes the force on a section of the elastic band.
 **/
struct ElasticBandSection {
  CuField imageLeft;
  CuField imageCenter;
  CuField imageRight;
  real energyLeft;
  real energyCenter;
  real energyRight;
  real springForce;
  CuField gradient;

  __device__ real maxEnergyDiff() const {
    return max(abs(energyCenter - energyLeft),  // NOLINT[4]
               abs(energyCenter - energyRight));
  }

  __device__ real minEnergyDiff() const {
    return min(abs(energyCenter - energyLeft),  // NOLINT[4]
               abs(energyCenter - energyRight));
  }

  /** Compute the tangent of the elastic band section in a specific cell.
   *  Result is not normalized and not projected on the tangent space of the
   *  image.
   *  @see Appendix A in Comp. Phys. Comm. 196, 335–347 (2015).
   */
  __device__ real3 tangent(int idx) const {
    real3 leftTangent = imageCenter.vectorAt(idx) - imageLeft.vectorAt(idx);
    real3 rightTangent = imageRight.vectorAt(idx) - imageCenter.vectorAt(idx);

    if (energyLeft < energyCenter && energyCenter < energyRight)
      return rightTangent;

    if (energyLeft > energyCenter && energyCenter > energyRight)
      return leftTangent;

    if (energyRight > energyLeft) {
      return rightTangent * maxEnergyDiff() + leftTangent * minEnergyDiff();
    } else {
      return rightTangent * minEnergyDiff() + leftTangent * maxEnergyDiff();
    }
  }

  /** Compute the force on this elastic band section in a single cell
   *  @see Eq. 17 in Comp. Phys. Comm. 196, 335–347 (2015).
   */
  __device__ inline real3 force(int idx) const {
    real3 m = imageCenter.vectorAt(idx);
    real3 t = normalized(projectOrthogonalOn(tangent(idx), m));
    real3 grad = gradient.vectorAt(idx);
    real3 force = -projectOrthogonalOn(grad, t) + springForce * t;
    return projectOrthogonalOn(force, m);
  }
};

/** Compute for all cells the force on an elastic band section. */
__global__ void k_force(CuField force, ElasticBandSection section) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (force.cellInGeometry(idx))
    force.setVectorInCell(idx, section.force(idx));
}

/** Make a single Euler step (with VPO). */
__global__ static void k_step(CuField xvalues,
                              CuField velocity,
                              CuField force,
                              real dt) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (!xvalues.cellInGeometry(idx))
    return;

  real3 x = xvalues.vectorAt(idx);
  real3 v = velocity.vectorAt(idx);
  real3 f = force.vectorAt(idx);

  x = x + dt * v;
  v = v + dt * f;

  // velocity projection optimization (eq. F.7)
  v = dot(v, f) > 0 ? projectOn(v, f) : real3{0, 0, 0};

  xvalues.setVectorInCell(idx, normalized(x));
  velocity.setVectorInCell(idx, v);
}

ElasticBand::ElasticBand(Ferromagnet* magnet, const std::vector<Field>& images)
    : magnet_(magnet), springConstant_(0.0) {
  for (const auto image : images) {
    if (image.grid() != magnet_->grid()) {
      throw std::runtime_error(
          "Not all images have the same grid as the grid of the magnet");
    }

    if (image.ncomp() != 3) {
      throw std::runtime_error("Not all images have 3 components");
    }
  }

  for (const auto image : images) {
    images_.emplace_back(image);
    velocities_.emplace_back(Field(magnet_->system(), 3, 0.0));
  }
}

int ElasticBand::nImages() const {
  return images_.size();
}

void ElasticBand::relaxEndPoints() {
  Field m0 = magnet_->magnetization()->field();
  int endpoints[2] = {0, nImages() - 1};
  for (int idx : endpoints) {
    magnet_->magnetization()->set(images_.at(idx));
    magnet_->minimize();
    images_[idx] = magnet_->magnetization()->field();
  }
  magnet_->magnetization()->set(m0);
}

void ElasticBand::selectImage(int idx) {
  if (idx < 0 && idx >= images_.size()) {
    std::runtime_error("Not a valid image index");
  }
  magnet_->magnetization()->set(images_.at(idx));
}

std::vector<real> ElasticBand::energies() {
  Field m0 = magnet_->magnetization()->field();

  std::vector<real> energies(nImages());
  for (int i = 0; i < nImages(); i++) {
    magnet_->magnetization()->set(images_.at(i));
    energies[i] = evalTotalEnergy(magnet_);
  }

  magnet_->magnetization()->set(m0);

  return energies;
}

std::vector<real> ElasticBand::springForces() const {
  std::vector<real> distance(nImages() - 1);
  for (int i = 0; i < distance.size(); i++)
    distance[i] = geodesicDistanceImages(i, i + 1);

  std::vector<real> force(nImages());
  for (int i = 1; i < nImages() - 1; i++)  // No spring force on end points
    force[i] = springConstant_ * (distance[i - 1] - distance[i]);

  return force;
}

std::vector<Field> ElasticBand::forceFields() {
  Field m0 = magnet_->magnetization()->field();

  std::vector<real> energy = energies();
  std::vector<real> springForce = springForces();

  std::vector<Field> forces(images_.size(), Field(magnet_->system(), 3, 0.0));
  for (int i = 1; i < nImages() - 1; i++) {  // End points need to stay fixed
    magnet_->magnetization()->set(images_.at(i));
    Field gradient = -evalEffectiveField(magnet_);  // note the negative sign !

    ElasticBandSection section{images_[i - 1].cu(), images_[i].cu(),
                               images_[i + 1].cu(), energy[i - 1],
                               energy[i],           energy[i + 1],
                               springForce[i],      gradient.cu()};

    cudaLaunch(magnet_->grid().ncells(), k_force, forces[i].cu(), section);
  }

  magnet_->magnetization()->set(m0);
  return forces;
}

void ElasticBand::step(real stepsize) {
  std::vector<Field> forces = forceFields();
  for (int i = 1; i < nImages() - 1; i++) {  // End points need to stay fixed
    cudaLaunch(magnet_->grid().ncells(), k_step, images_[i].cu(),
               velocities_[i].cu(), forces[i].cu(), stepsize);
  }
}

real ElasticBand::geodesicDistanceImages(int i, int j) const {
  return geodesicDistance(images_[i], images_[j]);
}

void ElasticBand::setSpringConstant(real newSpringConstant) {
  springConstant_ = newSpringConstant;
}

real ElasticBand::springConstant() const {
  return springConstant_;
}
