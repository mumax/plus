#include <stdexcept>

#include "cudalaunch.hpp"
#include "effectivefield.hpp"
#include "elasticband.hpp"
#include "energy.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "reduce.hpp"
#include "scalarquantity.hpp"

// https://www.sciencedirect.com/science/article/abs/pii/S0304885302003888
// 10.1063/1.1323224

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

__device__ inline real emax(real ep, real ec, real en) {
  return abs(ep - ec) > abs(en - ec) ? abs(ep - ec) : abs(en - ec);
}
__device__ inline real emin(real ep, real ec, real en) {
  return abs(ep - ec) < abs(en - ec) ? abs(ep - ec) : abs(en - ec);
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

  real3 tleft = curr - prev;
  real3 tright = next - curr;
  real3 tangent;

  if (energyPrev < energyCurrent && energyCurrent < energyNext) {
    tangent = tright;
  } else if (energyPrev > energyCurrent && energyCurrent > energyNext) {
    tangent = tleft;
  } else {
    real emax_ = emax(energyPrev, energyCurrent, energyNext);
    real emin_ = emin(energyPrev, energyCurrent, energyNext);
    if (energyNext > energyPrev) {
      tangent = tright * emax_ + tleft * emin_;
    } else {
      tangent = tright * emin_ + tleft * emax_;
    }
  }

  tangent = normalized(tangent);
  tangent = tangent - dot(tangent, curr) * curr;
  tangent = normalized(tangent);

  real3 f = -(grad - dot(grad, tangent) * tangent);
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

  auto energy = energies();

  std::vector<Field> forces;
  for (const auto image : images_) {
    forces.emplace_back(Field(magnet_->system(), 3));
  }

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
  real3 e_f = normalized(f);
  v = dot(v, e_f) > 0 ? dot(v, e_f) * e_f : real3{0., 0., 0.};

  xvalues.setVectorInCell(idx, normalized(x));
  velocity.setVectorInCell(idx, v);
}

void ElasticBand::step(real dt) {
  std::vector<Field> forces = perpendicularForces();
  for (int i = 1; i < nImages() - 1; i++) {  // End points need to stay fixed
    cudaLaunch(magnet_->grid().ncells(), k_step, images_[i].cu(),
               velocities_[i].cu(), forces[i].cu(), dt);
  }
}

real ElasticBand::geodesicDistanceImages(int i, int j) {
  return geodesicDistance(images_[i], images_[j]);
}
