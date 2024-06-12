#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "field.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"
#include "strayfield.hpp"
#include "world.hpp"

class FieldQuantity;
class MumaxWorld;
class System;

class Magnet {
 public:
  explicit Magnet(MumaxWorld* world,
                  Grid grid,
                  std::string name,
                  GpuBuffer<bool> geometry = GpuBuffer<bool>());

  virtual ~Magnet() = default; // default destructor, replace later (remove strayfield)

  std::string name() const;
  std::shared_ptr<const System> system() const;
  const World* world() const;
  Grid grid() const;
  real3 cellsize() const;
  const GpuBuffer<bool>& getGeometry() const;


 public:
  std::shared_ptr<System> system_;  // the system_ has to be initialized first,
                                    // hence its listed as the first datamember here
  std::string name_;

  // Delete copy constructor and copy assignment operator to prevent shallow copies
  Magnet(const Magnet&) = delete;
  Magnet& operator=(const Magnet&) = delete;

  Magnet(Magnet&& other) noexcept : system_(other.system_), name_(other.name_) {
        other.system_ = nullptr;
        other.name_ = "";
    }

  // Provide move constructor and move assignment operator
  Magnet& operator=(Magnet&& other) noexcept {
        if (this != &other) {
            system_ = other.system_;
            name_ = other.name_;
            other.system_ = nullptr;
            other.name_ = "";
        }
        return *this;
    }
};

/* TO DO - global properties / parameters / ...
* Poissonsystem?
* STT-parameters?
* Check if "includes" in ferromagnet.* are still needed
* Delete int comp in Magnet construction
*/