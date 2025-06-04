#include "variable.hpp"

#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <sstream>

#include "field.hpp"
#include "fieldops.hpp"
#include "system.hpp"

Variable::Variable(std::shared_ptr<const System> system, int ncomp,
                   std::string name, std::string unit)
    : name_(name), unit_(unit) {
  field_ = new Field(system, ncomp);
}

Variable::~Variable() {
  delete field_;
}

int Variable::ncomp() const {
  return field_->ncomp();
}

std::shared_ptr<const System> Variable::system() const {
  return field_->system();
}

std::string Variable::name() const {
  return name_;
}

std::string Variable::unit() const {
  return unit_;
}

Field Variable::eval() const {
  return field_->eval();
}

const Field& Variable::field() const {
  return *field_;
}

void Variable::set(const Field& src) const {
  if (src.system() != field_->system()) {
    throw std::runtime_error(
        "Can not set the variable because the given field variable is defined "
        "on another system.");
  }
  *field_ = src;
}

void Variable::set(real value) const {
  if (ncomp() != 1)
    throw std::runtime_error("Variable has " + std::to_string(ncomp()) +
                             "components instead of 1");
  field_->setUniformComponent(0, value);
}

void Variable::set(real3 value) const {
  if (ncomp() != 3)
    throw std::runtime_error("Variable has " + std::to_string(ncomp()) +
                             "components instead of 3");
  field_->setUniformComponent(0, value.x);
  field_->setUniformComponent(1, value.y);
  field_->setUniformComponent(2, value.z);
}

// Read ovf2 binary file
void Variable::loadFile(std::string file) {
  std::ifstream in(file, std::ios::binary);
  if (!in.is_open()) {
      throw std::runtime_error("Cannot open file " + file);
  }

  std::string line;
  bool inHeader = false;
  bool firstLine = true;
  real controlnumber;

  // Check the file and make sure it is compatible
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
        // TODO: change name?
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

  // Read the actual data
  size_t totalValues = system()->grid().size().x * system()->grid().size().y * system()->grid().size().z * ncomp();
  std::vector<real> data(totalValues);
  in.read(reinterpret_cast<char*>(&controlnumber), sizeof(real));
  if (controlnumber != realControlnumber) {
    throw std::runtime_error("Unexpected control number: " + std::to_string(controlnumber));
  }
  in.read(reinterpret_cast<char*>(data.data()), totalValues * sizeof(real));
  field_->setData(data);
}

NormalizedVariable::NormalizedVariable(std::shared_ptr<const System> system,
                                       int ncomp,
                                       std::string name,
                                       std::string unit)
    : Variable(system, ncomp, name, unit) {}

void NormalizedVariable::set(const Field& src) const {
  // TODO: check if this is possible without the extra copy
  Variable::set(normalized(src));
}

void NormalizedVariable::set(real value) const {
  Variable::set(1);
}

void NormalizedVariable::set(real3 value) const {
  Variable::set(normalized(value));
}