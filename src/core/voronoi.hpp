#include <functional>
#include <random>
#include <vector>
#include <unordered_map>

#include "datatypes.hpp"
#include "field.hpp"
#include "gpubuffer.hpp"
#include "grid.hpp"

struct Center {
  real3 pos;
  unsigned int ridx;
  Center() : pos{0, 0, 0}, ridx(0) {}
  Center(real3 position, unsigned int region_idx)
      : pos(position), ridx(region_idx) {}
};

struct Tile {
  int3 pos;
  std::vector<Center> centers;
};

struct Int3Hash {
  // Hash function to allow int3 to be used as a key
    std::size_t operator()(const int3& k) const {
        return std::hash<int>()(k.x) ^ std::hash<int>()(k.y) ^ std::hash<int>()(k.z);
    }
};

class VoronoiTessellator {
 public:
  VoronoiTessellator(real grainsize,
                     int seed,
                     unsigned int maxIdx=255,
                     const std::function<unsigned int(real3)>& centerIdx = nullptr);
  ~VoronoiTessellator() = default;

  // * Generate a Voronoi tessellation
  std::vector<unsigned int> generate(Grid grid, real3 cellsize, const bool pbc);
  
  real3 getTileSize(real3 griddims);

  // * Calc nearest center and assign center index to coo
  unsigned int regionOf(real3 coo);

  real3 periodicShift(real3 coo, real3 center);
  
  private:
  // * Calculate position and index of centers in tile
  std::vector<Center> centersInTile(int3 pos);

  // * Poisson distribution
  int Poisson(real lambda);
  
  // * Calculate to which tile the given cell belongs
  Tile tileOfCell(real3 coo);

public:
  GpuBuffer<unsigned int> tessellation;
private:
  real grainsize_;
  real3 grid_dims_;
  bool pbc_;

  real3 tilesize_;
  int numTiles_x;
  int numTiles_y;

  int seed_;
  std::unordered_map<int3, Tile, Int3Hash> tileCache_;

 // RNG related members
  real lambda_; // Poisson parameter
  std::default_random_engine engine_;
  std::uniform_real_distribution<> distReal_;
  std::uniform_int_distribution<> distInt_;
  std::function<unsigned int(real3)> centerIdx_;
};