#include "dmitensor.hpp"

#include <memory>
#include <vector>

#include "parameter.hpp"
#include "system.hpp"

DmiTensor::DmiTensor(std::shared_ptr<const System> system)
    : xxy(system),
      xyz(system),
      xxz(system),
      yxy(system),
      yyz(system),
      yxz(system),
      zxy(system),
      zyz(system),
      zxz(system) {}

CuDmiTensor DmiTensor::cu() const {
  return CuDmiTensor{
      xxy.cu(), xyz.cu(), xxz.cu(), yxy.cu(), yyz.cu(),
      yxz.cu(), zxy.cu(), zyz.cu(), zxz.cu(),
  };
}

bool DmiTensor::assuredZero() const {
  return xxy.assuredZero() && xyz.assuredZero() && xxz.assuredZero() &&
         yxy.assuredZero() && yyz.assuredZero() && yxz.assuredZero() &&
         zxy.assuredZero() && zyz.assuredZero() && zxz.assuredZero();
}
