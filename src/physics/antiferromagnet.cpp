#include "antiferromagnet.hpp"

#include <algorithm>
#include <memory>
#include <math.h>
#include <cfloat>
#include <vector>

#include "fieldquantity.hpp"
#include "gpubuffer.hpp"
#include "minimizer.hpp"
#include "mumaxworld.hpp"
#include "relaxer.hpp"

Antiferromagnet::Antiferromagnet(std::shared_ptr<System> system_ptr,
                                 std::string name)
    : Magnet(system_ptr, name),
      afmex_cell(system(), 0.0, name + ":afmex_cell", "J/m"),
      afmex_nn(system(), 0.0, name + ":afmex_nn", "J/m"),
      latcon(system(), 0.35e-9, name + ":latcon", "m"),
      dmiTensor(system()),
      interAfmExchNN(system(), 0.0, name + ":inter_afmex_nn", "J/m"),
      scaleAfmExchNN(system(), 1.0, name + ":scale_afmex_nn", ""),
      sub1_(Ferromagnet(system_ptr, name + ":sublattice_1", this)),
      sub2_(Ferromagnet(system_ptr, name + ":sublattice_2", this)) {}
      
Antiferromagnet::Antiferromagnet(MumaxWorld* world,
                         Grid grid,
                         std::string name,
                         GpuBuffer<bool> geometry,
                         GpuBuffer<uint> regions)
    : Antiferromagnet(std::make_shared<System>(world, grid, geometry, regions), name) {}

const Ferromagnet* Antiferromagnet::sub1() const {
  return &sub1_;
}

const Ferromagnet* Antiferromagnet::sub2() const {
  return &sub2_;
}

const Ferromagnet* Antiferromagnet::getOtherSublattice(const Ferromagnet* sub) const {
  if (std::find(sublattices_.begin(), sublattices_.end(), sub) == sublattices_.end())
    throw std::out_of_range("Sublattice not found in Antiferromagnet.");
  return sublattices_[0] == sub ? &sub2_ : &sub1_;
}

std::vector<const Ferromagnet*> Antiferromagnet::sublattices() const {
  return sublattices_;
}

void Antiferromagnet::minimize(real tol, int nSamples) {
  Minimizer minimizer(this, tol, nSamples);
  minimizer.exec();
}

void Antiferromagnet::relax(real tol) {
  std::vector<real> threshold = {sub1()->RelaxTorqueThreshold,
                                 sub2()->RelaxTorqueThreshold};
    // If only one sublattice has a user-set threshold, then both
    // sublattices are relaxed using the same threshold.
    if (threshold[0] > 0.0 && threshold[1] <= 0.0)
      threshold[1] = threshold[0];
    else if (threshold[0] <= 0.0 && threshold[1] > 0.0)
      threshold[0] == threshold[1];

    Relaxer relaxer(this, threshold, tol);
    relaxer.exec();
}