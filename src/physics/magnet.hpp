#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "field.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "parameter.hpp"
#include "strayfield.hpp"
#include "variable.hpp"
#include "world.hpp"
#include "system.hpp"

class Antiferromagnet;
class Ferromagnet;
class FieldQuantity;
class HostMagnet;
class MumaxWorld;
class NcAfm;
class System;

class Magnet {
 friend class MumaxWorld;
 public:
  explicit Magnet(std::shared_ptr<System> system_ptr,
                  std::string name);
  /*explicit Magnet(MumaxWorld* world,
                  Grid grid,
                  std::string name,
                  GpuBuffer<bool> geometry = GpuBuffer<bool>());
*/
  virtual ~Magnet();

  std::string name() const;
  std::shared_ptr<const System> system() const;
  const World* world() const;
  const MumaxWorld* mumaxWorld() const;
  Grid grid() const;
  real3 cellsize() const;
  const GpuBuffer<bool>& getGeometry() const;

  // Cast Magnet instance to child instances
  const Ferromagnet* asFM() const;
  const HostMagnet* asHost() const;
  const Antiferromagnet* asAFM() const;
  const NcAfm* asNcAfm() const;

  const StrayField* getStrayField(const Magnet*) const;
  std::vector<const StrayField*> getStrayFields() const;
  void addStrayField(
      const Magnet*,
      StrayFieldExecutor::Method method = StrayFieldExecutor::METHOD_AUTO);
  void removeStrayField(const Magnet*);

 private:
  std::shared_ptr<System> system_;  // the system_ has to be initialized first,
                                    // hence its listed as the first datamember here
  std::string name_;
  std::map<const Magnet*, StrayField*> strayFields_;

  // these take a lot of memory. Don't initialize unless wanted!
  std::unique_ptr<Variable> elasticDisplacement_;
  std::unique_ptr<Variable> elasticVelocity_;
  bool enableElastodynamics_;

 public:
  bool enableAsStrayFieldSource;
  bool enableAsStrayFieldDestination;
  bool enableElastodynamics() const {return enableElastodynamics_;}
  void setEnableElastodynamics(bool);

  // Elasticity
  const Variable* elasticDisplacement() const;
  const Variable* elasticVelocity() const;

  VectorParameter externalBodyForce;  // Externally applied force density
  VectorParameter rigidNormStrain;
  VectorParameter rigidShearStrain;

  // stiffness constants; TODO: can this be generalized to a 6x6 tensor?
  Parameter C11;  // C11 = C22 = C33
  Parameter C12;  // C12 = C13 = C23
  Parameter C44;  // C44 = C55 = C66

  Parameter eta;  // Phenomenological elastic damping constant
  Parameter stiffnessDamping;  // eta_ij = stiffnessDamping * Cij
  Parameter eta11;  // viscosity tensor
  Parameter eta12;
  Parameter eta44;
  Parameter rho;  // Mass density


  // Delete copy constructor and copy assignment operator to prevent shallow copies
  Magnet(const Magnet&) = delete;
  Magnet& operator=(const Magnet&) = delete;

  Magnet(Magnet&& other) noexcept;

  // Provide move constructor and move assignment operator
  Magnet& operator=(Magnet&& other) noexcept;
};