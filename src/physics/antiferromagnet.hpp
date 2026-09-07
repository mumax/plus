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
 
void minimize(real tol = 1e-6, int nSamples = 20,
              real tolEl = 1e-6, int nSamplesEl = 20,
              real stepsizeEl = 1e-30, real stepsizeElFallback = 1e-30,
              int maxSteps = 200000, int rigidBodyModesInterval = 1, int rigidBodyModesDelay = 0,
              int rigidBodyModesMethod = 0);
void relax(real tol);

 private:
  Ferromagnet sub1_;
  Ferromagnet sub2_;
};
