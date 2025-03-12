#include "strayfield.hpp"

#include <memory>

#include "antiferromagnet.hpp"
#include "magnet.hpp"
#include "ferromagnet.hpp"
#include "ncafm.hpp"
#include "parameter.hpp"
#include "strayfieldbrute.hpp"
#include "strayfieldfft.hpp"
#include "system.hpp"
#include "world.hpp"

std::unique_ptr<StrayFieldExecutor> StrayFieldExecutor::create(
    const Magnet* magnet,
    std::shared_ptr<const System> system,
    Method method) {
  switch (method) {
    case StrayFieldExecutor::METHOD_AUTO:
      // TODO: make smart choice (dependent on the
      // grid sizes) when choosing between fft or
      // brute method. For now, we choose fft method
      return std::make_unique<StrayFieldFFTExecutor>(magnet, system);
    case StrayFieldExecutor::METHOD_FFT:
      return std::make_unique<StrayFieldFFTExecutor>(magnet, system);
    case StrayFieldExecutor::METHOD_BRUTE:
      return std::make_unique<StrayFieldBruteExecutor>(magnet, system);
    default:  // TODO: should it throw an error or default to METHOD_AUTO?
      throw std::invalid_argument("Stray field executor method number '"
                       + std::to_string(method) + "' does not exist");
  }
}

StrayFieldExecutor::StrayFieldExecutor(const Magnet* magnet,
                                       std::shared_ptr<const System> system)
    : magnet_(magnet), system_(system) {}

StrayField::StrayField(const Magnet* magnet,
                       std::shared_ptr<const System> system,
                       StrayFieldExecutor::Method method)
    : magnet_(magnet), system_(system), executor_(nullptr) {
  setMethod(method);
}

StrayField::StrayField(const Magnet* magnet,
                       Grid grid,
                       StrayFieldExecutor::Method method)
    : magnet_(magnet), executor_(nullptr) {
  system_ = std::make_shared<System>(magnet->world(), grid);
  setMethod(method);
}

StrayField::~StrayField() {}

void StrayField::setMethod(StrayFieldExecutor::Method method) {
  if (!executor_ || executor_->method() != method) {
    executor_ = StrayFieldExecutor::create(magnet_, system_, method);
  }
}

void StrayField::recreateStrayFieldExecutor() {
  executor_ = StrayFieldExecutor::create(magnet_, system_, executor_->method());
}

const Magnet* StrayField::source() const {
  return magnet_;
}

int StrayField::ncomp() const {
  return 3;
}

std::shared_ptr<const System> StrayField::system() const {
  return system_;
}

Field StrayField::eval() const {
  if (assuredZero()) {
    return Field(system(), ncomp(), 0.0);
  }
  return executor_->exec();
}

std::string StrayField::unit() const {
  return "T";
}

bool StrayField::assuredZero() const {
  if(const Ferromagnet* mag = dynamic_cast<const Ferromagnet*>(magnet_))
    return mag->msat.assuredZero();
  else if (const Antiferromagnet* mag = dynamic_cast<const Antiferromagnet*>(magnet_))
    return mag->sub1()->msat.assuredZero() && mag->sub2()->msat.assuredZero();
  else if (const NCAFM* mag = dynamic_cast<const NCAFM*>(magnet_))
    return true;
  else 
    throw std::invalid_argument("Cannot calculate strayfield since magnet is neither "
                                "a Ferromagnet nor an Antiferromagnet/Ferrimagnet.");
}
