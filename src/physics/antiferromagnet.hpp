#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "dmitensor.hpp"
#include "ferromagnet.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "inter_parameter.hpp"
#include "magnet.hpp"
#include "parameter.hpp"
#include "world.hpp"
#include "system.hpp"

class Antiferromagnet : public Magnet {
 public:
  Antiferromagnet(std::shared_ptr<System> system_ptr,
         std::string name);

  Antiferromagnet(MumaxWorld* world,
         Grid grid,
         std::string name,
         GpuBuffer<bool> geometry,
         GpuBuffer<unsigned int> regions);
         
  /** Empty destructor
   *
   * Sublattices are destroyed automatically. They are not pointers.
   */
  ~Antiferromagnet() override {};
  
 std::vector<const Ferromagnet*> sublattices() const;
 const Ferromagnet* sub1() const;
 const Ferromagnet* sub2() const;
 const Ferromagnet* getOtherSublattice(const Ferromagnet* sub) const;
 
 void minimize(real tol = 1e-6, int nSamples = 20);
 void relax(real tol);

 public:
  Parameter afmex_cell;
  Parameter afmex_nn;
  InterParameter interAfmExchNN;
  InterParameter scaleAfmExchNN;
  Parameter latcon;
  Ferromagnet sub1_;
  Ferromagnet sub2_;

  DmiTensor dmiTensor;
 
 private:
  std::vector<const Ferromagnet*> sublattices_ = {&sub1_, &sub2_};
};