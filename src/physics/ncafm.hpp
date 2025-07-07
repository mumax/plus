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

class NcAfm : public HostMagnet {
 public:
  NcAfm(std::shared_ptr<System> system_ptr,
         std::string name);

  NcAfm(MumaxWorld* world,
         Grid grid,
         std::string name,
         GpuBuffer<bool> geometry,
         GpuBuffer<unsigned int> regions);
         
  /** Empty destructor
   * Sublattices are destroyed automatically. They are not pointers.
   */
  ~NcAfm() override {};
  
 const Ferromagnet* sub1() const;
 const Ferromagnet* sub2() const;
 const Ferromagnet* sub3() const;

 void minimize(real tol = 1e-6, int nsamples = 30);
 void relax(real tol);

 private:
  Ferromagnet sub1_;
  Ferromagnet sub2_;
  Ferromagnet sub3_;
};