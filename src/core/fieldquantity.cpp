#include "fieldquantity.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "datatypes.hpp"
#include "field.hpp"
#include "fieldops.hpp"
#include "reduce.hpp"
#include "system.hpp"
#include "world.hpp"

Grid FieldQuantity::grid() const {
  return system()->grid();
}

void FieldQuantity::addToField(Field& f) const {
  if (!sameFieldDimensions(*this, f))
    throw std::invalid_argument(
        "Can not add the quantity to given field because the fields are "
        "incompatible.");
  f += *this;  // += checks assuredZero before calling eval()
}

std::vector<real> FieldQuantity::average() const {
  return fieldAverage(eval());
}

Field FieldQuantity::getRGB() const {
  return fieldGetRGB(eval());
}

const World* FieldQuantity::world() const {
  const System* sys = system().get();
  if (sys)
    return sys->world();
  return nullptr;
}

// Create an ovf2 text file
void FieldQuantity::writeOvf(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open file " + filename);
    }
    
    int3 gridSize = system()->grid().size();
    real3 cellSize = system().get()->world()->cellsize();

    // Write header
    out << "# OOMMF OVF 2.0\n";
    out << "# Segment count: 1\n";
    out << "# Begin: Segment\n";
    out << "# Begin: Header\n";
    out << "# Title: " << name() << "\n";
    out << "# meshtype: rectangular\n";
    out << "# meshunit: m\n";
    out << "# xmin: 0\n";
    out << "# ymin: 0\n";
    out << "# zmin: 0\n";
    out << "# xmax: " << cellSize.x * gridSize.x << "\n";
    out << "# ymax: " << cellSize.y * gridSize.y << "\n";
    out << "# zmax: " << cellSize.z * gridSize.z << "\n";
    out << "# valuedim: " << ncomp() << "\n";
    if (ncomp() == 1) {out << "# valuelabels: " << name() << "\n";}
    else if (ncomp() == 3) {
      out << "# valuelabels: " << name() << "_x " << name() << "_y " << name() << "_z\n";
    }
    if (unit() == "") {out << "# valueunit: 1\n";}
    else {out << "# valueunit: " << unit() << "\n";}
    out << "# Desc: Total simulation time: " << system()->world()->time() << " s\n";
    out << "# xbase: " << cellSize.x/2 << "\n";
    out << "# ybase: " << cellSize.y/2 << "\n";
    out << "# zbase: " << cellSize.z/2 << "\n";
    out << "# xnodes: " << gridSize.x << "\n";
    out << "# ynodes: " << gridSize.y << "\n";
    out << "# znodes: " << gridSize.z << "\n";
    out << "# xstepsize: " << cellSize.x << "\n";
    out << "# ystepsize: " << cellSize.y << "\n";
    out << "# zstepsize: " << cellSize.z << "\n";
    out << "# End: Header\n";
    out << "# Begin: Data Text\n";

    int nc = ncomp();
    std::vector<real> data = eval().getData();

    // Write data
    real controlnumber = 1234567.0; // ovf control number
    out.write(reinterpret_cast<const char*>(&controlnumber), sizeof(real));
    int i = 0;
    for (real val : data) {
      out.write(reinterpret_cast<const char*>(&val), sizeof(real));
    }

    out << "# End: Data Text\n";
    out << "# End: Segment\n";

    out.close();
}

