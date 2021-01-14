#include <stdexcept>

#include "cudalaunch.hpp"
#include "datatypes.hpp"
#include "effectivefield.hpp"
#include "elasticband.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "reduce.hpp"
#include "scalarquantity.hpp"

// https://aip.scitation.org/doi/10.1063/1.1323224
// https://doi.org/10.1016/j.cpc.2015.07.001
// https://journals.aps.org/prb/abstract/10.1103/PhysRevB.95.214418

ElasticBand::ElasticBand(Ferromagnet* magnet, const std::vector<Field>& images)
    : magnet_(magnet) {
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

/** Computes the (normalized) tangent at image mCenter according to Henkelman.
 *  See https://aip.scitation.org/doi/10.1063/1.1323224
 */
__device__ static inline real3 computeTangent(real3 mLeft,
                                              real3 mCenter,
                                              real3 mRight,
                                              real energyLeft,
                                              real energyCenter,
                                              real energyRight) {
  real3 leftTangent = mCenter - mLeft;
  real3 rightTangent = mRight - mCenter;

  if (energyLeft < energyCenter && energyCenter < energyRight) {
    return normalized(rightTangent);
  }

  if (energyLeft > energyCenter && energyCenter > energyRight) {
    return normalized(leftTangent);
  }

  real leftEnergyDiff = energyCenter - energyLeft;
  real rightEnergyDiff = energyRight - energyCenter;

  real dEnergyMax =
      max(abs(leftEnergyDiff), abs(rightEnergyDiff));  // NOLINT [4]

  real dEnergyMin =
      min(abs(leftEnergyDiff), abs(rightEnergyDiff));  // NOLINT [4]

  if (energyRight > energyLeft) {
    return normalized(rightTangent * dEnergyMax + leftTangent * dEnergyMin);
  } else {
    return normalized(rightTangent * dEnergyMin + leftTangent * dEnergyMax);
  }
}

__device__ real geodesicDistance(real3 v1, real3 v2) {
  return atan2(norm(cross(v1, v2)), dot(v1, v2));
}

__global__ static void k_force(CuField force,
                               CuField hfield,
                               CuField mprev,
                               CuField mcurrent,
                               CuField mnext,
                               real energyPrev,
                               real energyCurrent,
                               real energyNext) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!force.cellInGrid(idx))
    return;

  real3 grad = -hfield.vectorAt(idx);
  real3 prev = mprev.vectorAt(idx);
  real3 next = mnext.vectorAt(idx);
  real3 curr = mcurrent.vectorAt(idx);

  real3 tangent =
      computeTangent(prev, curr, next, energyPrev, energyCurrent, energyNext);
  tangent = tangent - dot(tangent, curr) * curr;
  tangent = normalized(tangent);

  real springConstant = 1.0;
  real3 springForce =
      springConstant *
      (geodesicDistance(next, curr) - geodesicDistance(curr, prev)) * tangent;

  real3 gradPerpendicular = grad - dot(grad, tangent) * tangent;
  // real3 f = -gradPerpendicular + springForce;
  real3 f = springForce;
  f = f - dot(f, curr) * curr;  // project force on tangent space

  force.setVectorInCell(idx, f);
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

std::vector<Field> ElasticBand::perpendicularForces() {
  Field m0 = magnet_->magnetization()->field();

  std::vector<real> energy = energies();

  std::vector<Field> forces(images_.size(), Field(magnet_->system(), 3, 0.0));
  for (int i = 1; i < nImages() - 1; i++) {  // End points need to stay fixed
    magnet_->magnetization()->set(images_.at(i));
    Field hField = evalEffectiveField(magnet_);
    cudaLaunch(magnet_->grid().ncells(), k_force, forces[i].cu(), hField.cu(),
               images_[i - 1].cu(), images_[i].cu(), images_[i + 1].cu(),
               energy[i - 1], energy[i], energy[i + 1]);
  }

  magnet_->magnetization()->set(m0);
  return forces;
}

__global__ static void k_step(CuField xvalues,
                              CuField velocity,
                              CuField force,
                              real dt) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!xvalues.cellInGrid(idx))
    return;

  real3 x = xvalues.vectorAt(idx);
  real3 v = velocity.vectorAt(idx);
  real3 f = force.vectorAt(idx);

  x = x + dt * v;
  v = v + dt * f;

  // velocity projection
  // real3 e_f = normalized(f);
  // v = dot(v, e_f) > 0 ? dot(v, e_f) * e_f : real3{0., 0., 0.};

  xvalues.setVectorInCell(idx, normalized(x));
  velocity.setVectorInCell(idx, v);
}

// __global__ void k_verlet_step(CuField xValues,
//                               CuField xPreviousValues,
//                               CuField force,
//                               real dt) {
//   int idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (!xValues.cellInGrid(idx))
//     return;
//
//   real3 a = force.vectorAt(idx);
//   real3 x0 = xValues.vectorAt(idx);
//   real3 xp = xPreviousValues.vectorAt(idx);
//
//   real3 xn = 2 * x0 - xp + a * dt * dt;
//
//   xValues.setVectorInCell(idx, xn);
//   xPreviousValues.setVectorInCell(idx, x0);
// }

void ElasticBand::step(real stepsize) {
  std::vector<Field> forces = perpendicularForces();
  for (int i = 1; i < nImages() - 1; i++) {  // End points need to stay fixed
    cudaLaunch(magnet_->grid().ncells(), k_step, images_[i].cu(),
               velocities_[i].cu(), forces[i].cu(), stepsize);
  }
}

real ElasticBand::geodesicDistanceImages(int i, int j) {
  return geodesicDistance(images_[i], images_[j]);
}
