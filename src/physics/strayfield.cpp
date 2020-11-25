#include "strayfield.hpp"

#include <memory>

#include "ferromagnet.hpp"
#include "parameter.hpp"
#include "strayfieldbrute.hpp"
#include "strayfieldfft.hpp"
#include "system.hpp"
#include "world.hpp"

std::unique_ptr<StrayFieldExecutor> StrayFieldExecutor::create(Method method,
                                                               Grid gridOut,
                                                               Grid gridIn,
                                                               real3 cellsize) {
  switch (method) {
    case StrayFieldExecutor::METHOD_AUTO:
      // TODO: make smart choice (dependent on the
      // grid sizes) when choosing between fft or
      // brute method. For now, we choose fft method
      return std::make_unique<StrayFieldFFTExecutor>(gridOut, gridIn, cellsize);
      break;
    case StrayFieldExecutor::METHOD_FFT:
      return std::make_unique<StrayFieldFFTExecutor>(gridOut, gridIn, cellsize);
      break;
    case StrayFieldExecutor::METHOD_BRUTE:
      return std::make_unique<StrayFieldBruteExecutor>(gridOut, gridIn,
                                                       cellsize);
      break;
  }
}

StrayField::StrayField(const Ferromagnet* magnet,
                       std::shared_ptr<const System> system,
                       StrayFieldExecutor::Method method)
    : magnet_(magnet), system_(system), executor_(nullptr) {
  setMethod(method);
}

StrayField::StrayField(const Ferromagnet* magnet,
                       Grid grid,
                       StrayFieldExecutor::Method method)
    : magnet_(magnet), executor_(nullptr) {
  system_ = std::make_shared<System>(magnet->world(), grid);
  setMethod(method);
}

StrayField::~StrayField() {}

void StrayField::setMethod(StrayFieldExecutor::Method method) {
  if (!executor_ || executor_->method() != method) {
    executor_ = StrayFieldExecutor::create(method, grid(), magnet_->grid(),
                                           magnet_->world()->cellsize());
  }
}

const Ferromagnet* StrayField::source() const {
  return magnet_;
}

int StrayField::ncomp() const {
  return 3;
}

std::shared_ptr<const System> StrayField::system() const {
  return system_;
}

Field StrayField::eval() const {
  Field h(system(), ncomp());
  if (assuredZero()) {
    h.makeZero();
    return h;
  }
  const Parameter* msat = &magnet_->msat;
  const Field& m = magnet_->magnetization()->field();
  executor_->exec(&h, &m, msat);
  return h;
}

std::string StrayField::unit() const {
  return "T";
}

bool StrayField::assuredZero() const {
  return magnet_->msat.assuredZero();
}