#pragma once

#include <memory>
#include <vector>

#include "dmitensor.hpp"
#include "ferromagnet.hpp"
#include "inter_parameter.hpp"
#include "magnet.hpp"

class HostMagnet : public Magnet {
 public:
  HostMagnet(std::shared_ptr<System> system_ptr, std::string name);

  // Return all sublattices
  std::vector<const Ferromagnet*> sublattices() const;

  void addSublattice(const Ferromagnet* sub);
  std::vector<const Ferromagnet*> getOtherSublattices(const Ferromagnet*) const;
  int getSublatticeIndex(const Ferromagnet*) const;

 public:
  Parameter afmex_cell;
  Parameter afmex_nn;
  Parameter latcon;
  InterParameter interAfmExchNN;
  InterParameter scaleAfmExchNN;

  DmiTensor dmiTensor;
  VectorParameter dmiVector;

 private:
  std::vector<const Ferromagnet*> sublattices_;
};
