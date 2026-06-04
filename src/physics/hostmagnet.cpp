#include "hostmagnet.hpp"

HostMagnet::HostMagnet(std::shared_ptr<System> system_ptr, std::string name)
    : Magnet(system_ptr, name),
      afmex_cell(system(), 0.0, name + ":afmex_cell", "J/m"),
      afmex_nn(system(), 0.0, name + ":afmex_nn", "J/m"),
      latcon(system(), 0.35e-9, name + ":latcon", "m"),
      interAfmExchNN(system(), 0.0, name + ":inter_afmex_nn", "J/m"),
      scaleAfmExchNN(system(), 1.0, name + ":scale_afmex_nn", ""),
      dmiTensor(system()),
      dmiVector(system(), real3{0,0,0}, name + ":dmi_vector", "J/m³") {}

std::vector<const Ferromagnet*> HostMagnet::sublattices() const {
    return sublattices_;
}

void HostMagnet::addSublattice(const Ferromagnet* sub) {
    sublattices_.push_back(sub);
}

std::vector<const Ferromagnet*> HostMagnet::getOtherSublattices(const Ferromagnet* sub) const {
    bool found = false;
    int j = 0;
    std::vector<const Ferromagnet*> result(sublattices_.size() - 1);
    for (int i = 0; i < sublattices_.size(); i++) {
    if (sublattices_[i] != sub) { result[j++] = sublattices_[i]; }
    else { found = true; }
    }
    if (!found)
        throw std::out_of_range("Sublattice not found in HostMagnet.");
    return result;
}

int HostMagnet::getSublatticeIndex(const Ferromagnet* magnet) const {
    for (int i = 0; i < sublattices_.size(); ++i) {
      if (sublattices_[i] == magnet)
        return i;
    }
    throw std::runtime_error("Sublattice not found in host magnet.");
}