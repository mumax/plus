#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ferromagnet.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "hostmagnet.hpp"
#include "parameter.hpp"
#include "world.hpp"
#include "system.hpp"

class Antiferromagnet : public HostMagnet {
 public:
  Antiferromagnet(std::shared_ptr<System> system_ptr,
                  std::string name);

  Antiferromagnet(MumaxWorld* world,
         Grid grid,
         std::string name,
         GpuBuffer<bool> geometry,
         GpuBuffer<unsigned int> regions);
         
  /** Empty destructor
   * Sublattices are destroyed automatically. They are not pointers.
   */
  ~Antiferromagnet() override {};
  
 const Ferromagnet* sub1() const;
 const Ferromagnet* sub2() const;
 
 void minimize(real tol = 1e-6, int nSamples = 20);
 void relax(real tol);

 private:
  Ferromagnet sub1_;
  Ferromagnet sub2_;
};