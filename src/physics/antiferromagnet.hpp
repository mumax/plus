#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ferromagnet.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
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
         GpuBuffer<bool> geometry);
         
  ~Antiferromagnet() override = default;
  
 std::vector<const Ferromagnet*> sublattices() const;
 const Ferromagnet* sub1() const;
 const Ferromagnet* sub2() const;
 const Ferromagnet* getOtherSublattice(const Ferromagnet* sub) const;

 public:
  Parameter afmex_cell;
  Parameter afmex_nn;
  Parameter latcon;
  Ferromagnet sub1_;
  Ferromagnet sub2_;
 
 private:
  std::vector<const Ferromagnet*> sublattices_ = {&sub1_, &sub2_};
};