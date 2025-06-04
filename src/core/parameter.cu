#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>

#include "datatypes.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "parameter.hpp"
#include "reduce.hpp"

Parameter::Parameter(std::shared_ptr<const System> system, real value,
                     std::string name, std::string unit)
    : system_(system), staticField_(nullptr), uniformValue_(value),
      name_(name), unit_(unit) {}

Parameter::~Parameter() {
  if (staticField_)
    delete staticField_;
}

void Parameter::set(real value) {
  uniformValue_ = value;
  if (staticField_) {
    delete staticField_;
    staticField_ = nullptr;
  }
}

void Parameter::set(const Field& values) {
  if (isUniformField(values)) {
    real* value = values.device_ptr(0);
    checkCudaError(cudaMemcpy(&uniformValue_, value, sizeof(real),
                            cudaMemcpyDeviceToHost));
    if (staticField_) {
      delete staticField_;
      staticField_ = nullptr;
    }
  }
  else
    staticField_ = new Field(values);
}

void Parameter::setInRegion(const unsigned int region_idx, real value) {
  if (isUniform()) {
    if (value == uniformValue_) return;
    staticField_ = new Field(system_, 1, uniformValue_);
  }
  staticField_->setUniformValueInRegion(region_idx, value);
}

bool Parameter::isUniform() const {
  return !staticField_ && DynamicParameter<real>::isUniform();
}

bool Parameter::assuredZero() const {
  return !isDynamic() && isUniform() && uniformValue_ == 0.0;
}

int Parameter::ncomp() const {
  return 1;
}

std::shared_ptr<const System> Parameter::system() const {
  return system_;
}

Field Parameter::eval() const {
  Field staticField(system_, ncomp());

  if (staticField_) {
    staticField = *staticField_;
  } else {
    staticField.setUniformValue(uniformValue_);
  }

  if (isDynamic()) {
    auto t = system_->world()->time();
    Field dynamicField(system_, ncomp());

    evalTimeDependentTerms(t, dynamicField);

    staticField += dynamicField;
  }

  return staticField;
}

real Parameter::getUniformValue() const {
  if (!isUniform()) {
    throw std::invalid_argument("Cannot get uniform value of non-uniform Parameter.");
  }
  return uniformValue_;
}

CuParameter Parameter::cu() const {
  if (isDynamic()) {
    auto t = system_->world()->time();
    dynamicField_.reset(new Field(system_, ncomp()));

    evalTimeDependentTerms(t, *dynamicField_);
  }

  return CuParameter(this);
}

void Parameter::loadFile(std::string file) {
  std::ifstream in(file);
  if (!in.is_open()) {
      throw std::runtime_error("Cannot open file " + file);
  }

  std::string line;
  bool inHeader = false;
  real controlnumber;
  real realControlnumber = 1234567.0;

  while (std::getline(in, line)) {
    // Skip comment lines
    if (line.empty()) continue;

    if (line.find("# Begin: Header") != std::string::npos) {
      inHeader = true;
    } else if (line.find("# End: Header") != std::string::npos) {
      inHeader = false;
    } else if (line.find("# Begin: Data Text") != std::string::npos) {
      break;
    } else if (inHeader) {
      std::istringstream ss(line);
      std::string tag;
      ss >> tag; // skip '#'
      ss >> tag;

      if (tag == "Title:") {
        std::getline(ss, tag);
        tag = tag.substr(1); // remove leading space
        // TODO: change name
      } else if (tag == "valuedim:") {
        ss >> tag;
        if (ncomp() != stoi(tag)) {
          throw std::invalid_argument("The number of components does not match.");
        }
      } else if (tag == "valueunit:") {
        ss >> tag;
        if (tag == "1") {tag = "";}
        if (unit() != tag) {
          throw std::invalid_argument("The units don't match.");
        }
      } else if (tag == "xnodes:") {
        ss >> tag;
        if (system()->grid().size().x != stoi(tag)) {
          throw std::invalid_argument("The number of x-cells don't match.");
        }
      } else if (tag == "ynodes:") {
        ss >> tag;
        if (system()->grid().size().y != stoi(tag)) {
          throw std::invalid_argument("The number of y-cells don't match.");
        }
      } else if (tag == "znodes:") {
        ss >> tag;
        if (system()->grid().size().z != stoi(tag)) {
          throw std::invalid_argument("The number of z-cells don't match.");
        }
      } else if (tag == "xstepsize:") {
        ss >> tag;
        if (system().get()->world()->cellsize().x != stor(tag)) {
          std::cout << std::to_string(system().get()->world()->cellsize().x) << " " << stod(tag);
          throw std::invalid_argument("The cell size x-values don't match.");
        }
      } else if (tag == "ystepsize:") {
        ss >> tag;
        if (system().get()->world()->cellsize().y != stor(tag)) {
          throw std::invalid_argument("The cell size y-values don't match.");
        }
      } else if (tag == "zstepsize:") {
        ss >> tag;
        if (system().get()->world()->cellsize().z != stor(tag)) {
          throw std::invalid_argument("The cell size z-values don't match.");
        }
      }
    }
  }
  size_t totalValues = system()->grid().size().x * system()->grid().size().y * system()->grid().size().z * ncomp();
  std::vector<real> data(totalValues);
  in.read(reinterpret_cast<char*>(&controlnumber), sizeof(real));
  if (controlnumber != realControlnumber) {
    throw std::runtime_error("Unexpected control number: " + std::to_string(controlnumber));
  }
  in.read(reinterpret_cast<char*>(data.data()), totalValues * sizeof(real));

  Field staticField = Field(system(), ncomp());
  staticField.setData(data);
  set(staticField);
}

VectorParameter::VectorParameter(std::shared_ptr<const System> system,
                                 real3 value,
                                 std::string name, std::string unit)
    : system_(system), staticField_(nullptr), uniformValue_(value),
      name_(name), unit_(unit) {}

VectorParameter::~VectorParameter() {
  if (staticField_)
    delete staticField_;
}

void VectorParameter::set(real3 value) {
  uniformValue_ = value;
  if (staticField_) {
    delete staticField_;
    staticField_ = nullptr;
  }
}

void VectorParameter::set(const Field& values) {
  if (isUniformField(values)) {
    real* valueX = values.device_ptr(0);
    real* valueY = values.device_ptr(1);
    real* valueZ = values.device_ptr(2);

    checkCudaError(cudaMemcpy(&uniformValue_.x, valueX, sizeof(real),
                            cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(&uniformValue_.y, valueY, sizeof(real),
                            cudaMemcpyDeviceToHost));
    checkCudaError(cudaMemcpy(&uniformValue_.z, valueZ, sizeof(real),
                            cudaMemcpyDeviceToHost));
    if (staticField_) {
      delete staticField_;
      staticField_ = nullptr;
    }
  }
  else
    staticField_ = new Field(values);
}

void VectorParameter::setInRegion(const unsigned int region_idx, real3 value) {
  if (isUniform()) {
    if (value == uniformValue_) return;
    staticField_ = new Field(system_, 3, uniformValue_);
  }
  staticField_->setUniformValueInRegion(region_idx, value);
}

bool VectorParameter::isUniform() const {
  return !staticField_ && DynamicParameter<real3>::isUniform();
}

bool VectorParameter::assuredZero() const {
  return !isDynamic() && isUniform() && uniformValue_ == real3{0.0, 0.0, 0.0};
}

int VectorParameter::ncomp() const {
  return 3;
}

std::shared_ptr<const System> VectorParameter::system() const {
  return system_;
}

Field VectorParameter::eval() const {
  Field staticField(system_, ncomp());

  if (staticField_) {
    staticField = *staticField_;
  } else {
    staticField.setUniformValue(uniformValue_);
  }

  if (isDynamic()) {
    auto t = system_->world()->time();
    Field dynamicField(system_, ncomp());

    evalTimeDependentTerms(t, dynamicField);

    staticField += dynamicField;
  }

  return staticField;
}

real3 VectorParameter::getUniformValue() const {
  if (!isUniform()) {
    throw std::invalid_argument("Cannot get uniform value of non-uniform Parameter.");
  }
  return uniformValue_;
}

void VectorParameter::loadFile(std::string file) {
  std::ifstream in(file);
  if (!in.is_open()) {
      throw std::runtime_error("Cannot open file " + file);
  }

  std::string line;
  bool inHeader = false;
  real controlnumber;
  real realControlnumber = 1234567.0;

  while (std::getline(in, line)) {
    // Skip comment lines
    if (line.empty()) continue;

    if (line.find("# Begin: Header") != std::string::npos) {
      inHeader = true;
    } else if (line.find("# End: Header") != std::string::npos) {
      inHeader = false;
    } else if (line.find("# Begin: Data Text") != std::string::npos) {
      break;
    } else if (inHeader) {
      std::istringstream ss(line);
      std::string tag;
      ss >> tag; // skip '#'
      ss >> tag;

      if (tag == "Title:") {
        std::getline(ss, tag);
        tag = tag.substr(1); // remove leading space
        // TODO: change name
      } else if (tag == "valuedim:") {
        ss >> tag;
        if (ncomp() != stoi(tag)) {
          throw std::invalid_argument("The number of components does not match.");
        }
      } else if (tag == "valueunit:") {
        ss >> tag;
        if (tag == "1") {tag = "";}
        if (unit() != tag) {
          throw std::invalid_argument("The units don't match.");
        }
      } else if (tag == "xnodes:") {
        ss >> tag;
        if (system()->grid().size().x != stoi(tag)) {
          throw std::invalid_argument("The number of x-cells don't match.");
        }
      } else if (tag == "ynodes:") {
        ss >> tag;
        if (system()->grid().size().y != stoi(tag)) {
          throw std::invalid_argument("The number of y-cells don't match.");
        }
      } else if (tag == "znodes:") {
        ss >> tag;
        if (system()->grid().size().z != stoi(tag)) {
          throw std::invalid_argument("The number of z-cells don't match.");
        }
      } else if (tag == "xstepsize:") {
        ss >> tag;
        if (system().get()->world()->cellsize().x != stor(tag)) {
          std::cout << std::to_string(system().get()->world()->cellsize().x) << " " << stod(tag);
          throw std::invalid_argument("The cell size x-values don't match.");
        }
      } else if (tag == "ystepsize:") {
        ss >> tag;
        if (system().get()->world()->cellsize().y != stor(tag)) {
          throw std::invalid_argument("The cell size y-values don't match.");
        }
      } else if (tag == "zstepsize:") {
        ss >> tag;
        if (system().get()->world()->cellsize().z != stor(tag)) {
          throw std::invalid_argument("The cell size z-values don't match.");
        }
      }
    }
  }
  size_t totalValues = system()->grid().size().x * system()->grid().size().y * system()->grid().size().z * ncomp();
  std::vector<real> data(totalValues);
  in.read(reinterpret_cast<char*>(&controlnumber), sizeof(real));
  if (controlnumber != realControlnumber) {
    throw std::runtime_error("Unexpected control number: " + std::to_string(controlnumber));
  }
  in.read(reinterpret_cast<char*>(data.data()), totalValues * sizeof(real));

  Field staticField = Field(system(), ncomp());
  staticField.setData(data);
  set(staticField);
}

CuVectorParameter VectorParameter::cu() const {
  if (isDynamic()) {
    auto t = system_->world()->time();
    dynamicField_.reset(new Field(system_, ncomp()));

    evalTimeDependentTerms(t, *dynamicField_);
  }

  return CuVectorParameter(this);
}
