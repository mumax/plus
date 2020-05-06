#include <stdexcept>

#include "cudalaunch.hpp"
#include "elasticband.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "scalarquantity.hpp"

ElasticBand::ElasticBand(Ferromagnet* magnet, std::vector<Field*> images)
    : magnet_(magnet) {
  for (const auto image : images) {
    if (image->grid() != magnet_->grid()) {
      throw std::runtime_error(
          "Not all images have the same grid as the grid of the magnet");
    }
    if (image->ncomp() != 3) {
      throw std::runtime_error("Not all images have 3 components");
    }
  }

  for (const auto image : images) {
    images_.push_back(std::make_unique<Field>(magnet_->grid(), 3));
    images_.back()->copyFrom(image);

    forces_.push_back(std::make_unique<Field>(magnet_->grid(), 3));

    velocities_.push_back(std::make_unique<Field>(magnet_->grid(), 3));
    for (int comp = 0; comp < 3; comp++)
      velocities_.back()->setUniformComponent(0.0, comp);
  }
}

int ElasticBand::nImages() const {
  return images_.size();
}

void ElasticBand::relaxEndPoints() {
  int endpoints[2] = {0, nImages() - 1};
  for (int idx : endpoints) {
    selectImage(idx);
    magnet_->minimize();
    images_[idx]->copyFrom(magnet_->magnetization()->field());
  }
}

void ElasticBand::selectImage(int idx) {
  if (idx < 0 && idx >= images_.size()) {
    std::runtime_error("Not a valid image index");
  }
  magnet_->magnetization()->set(images_.at(idx).get());
}

__device__ real emax(real ep, real ec, real en) {
  return abs(ep - ec) > abs(en - ec) ? abs(ep - ec) : abs(en - ec);
}
__device__ real emin(real ep, real ec, real en) {
  return abs(ep - ec) < abs(en - ec) ? abs(ep - ec) : abs(en - ec);
}

__global__ void k_dir(CuField dir,
                      CuField fgradient,
                      CuField fprev,
                      CuField fcurrent,
                      CuField fnext,
                      real energyPrev,
                      real energyCurrent,
                      real energyNext) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!dir.cellInGrid(idx))
    return;

  real3 gradient = fgradient.vectorAt(idx);
  real3 prev = fprev.vectorAt(idx);
  real3 next = fnext.vectorAt(idx);
  real3 curr = fcurrent.vectorAt(idx);

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

  real3 d = gradient - dot(gradient, tangent) * tangent;

  dir.setVectorInCell(idx, d);
}

__global__ void k_step(CuField xvalues, CuField velocity, CuField force, real dt) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (!xvalues.cellInGrid(idx))
    return;

  real3 x = xvalues.vectorAt(idx);
  real3 v = velocity.vectorAt(idx);
  real3 f = force.vectorAt(idx);

  // velocity projection
  real3 e_f = normalized(f);
  v = dot(v,e_f) > 0 ? dot(v,e_f)*e_f : real3{0.,0.,0.};

  xvalues.setVectorInCell(idx, normalized(x+dt*v));
  velocity.setVectorInCell(idx, v+dt*f);
}

void ElasticBand::step(real dt) {
  std::vector<real> energies(nImages());

  for (int i = 0; i < nImages(); i++) {
    selectImage(i);
    energies[i] = magnet_->totalEnergy()->eval();
  }

  for (int i = 1; i < nImages() - 1; i++) {  // End points need to stay fixed
    selectImage(i);
    auto torque = magnet_->effectiveField()->eval();
    cudaLaunch(magnet_->grid().ncells(), k_dir, forces_[i]->cu(), torque->cu(),
               images_[i - 1]->cu(), images_[i]->cu(), images_[i + 1]->cu(),
               energies[i - 1], energies[i], energies[i + 1]);
  }

  for (int i = 1; i < nImages() - 1; i++) {  // End points need to stay fixed
    cudaLaunch(magnet_->grid().ncells(), k_step, images_[i]->cu(),
               velocities_[i]->cu(), forces_[i]->cu(), dt);
  }
}

void ElasticBand::solve() {}