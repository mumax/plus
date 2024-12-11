#include <math.h>
#include <random>

#include "field.hpp"
#include "gpubuffer.hpp"
#include "voronoi.hpp"

VoronoiTessellator::VoronoiTessellator(Grid grid, real grainsize, real3 cellsize)
    : grid(grid),
      grainsize_(grainsize),
      cellsize_(cellsize),
      distReal_(0.0, 1.0) {
        real tilesize_in_grains = 2; // tile size in unit grains
        tilesize_ = tilesize_in_grains * grainsize;
        lambda_ = tilesize_in_grains * tilesize_in_grains;
        tessellation = this->generate();
    }

GpuBuffer<unsigned int> VoronoiTessellator::generate() {

   std::vector<unsigned int> data(grid.ncells());
   for (int nx = 0; nx < grid.size().x; nx++) {
        for (int ny = 0; ny < grid.size().y; ny++) {
            real3 coo = real3{nx * cellsize_.x,
                              ny * cellsize_.y,
                              0};
            data[nx * grid.size().y + ny] = regionOf(coo);
        }
    }
    return GpuBuffer<unsigned int>(data);
}

unsigned int VoronoiTessellator::regionOf(real3 coo) {
    Tile t = tileOfCell(coo);
    Center nearest = Center{coo, 0};
    real mindist = INFINITY;
    for (int tx = t.pos.x-1; tx <= t.pos.x+1; tx++) {
        for (int ty = t.pos.y-1; ty <= t.pos.y+1; ty++) {
            std::vector<Center> centers = centersInTile(int3{tx, ty, 0});
            for (auto c : centers) {
                real dist = (coo.x - c.pos.x) * (coo.x - c.pos.x)
                          + (coo.y - c.pos.y) * (coo.y - c.pos.y);
                if (dist < mindist) {
                    nearest = c;
                    mindist = dist;
                }
            }
        }
    }
    return nearest.ridx;
}

std::vector<Center> VoronoiTessellator::centersInTile(int3 pos) {

    // Check if centers in this tile are already cached
    auto it = tileCache_.find(pos);
    if (it != tileCache_.end()) {
        return it->second.centers;
    }

    int64_t seed = (int64_t(pos.y) + (1<<24)) * (1<<24)
                 + (int64_t(pos.x) + (1<<24));
    engine_.seed (seed ^ 1234567);

    int N = Poisson(lambda_);

    std::vector<Center> centers(N);

    for (int n = 0; n < N; n++) {
        real cx = (pos.x + distReal_(engine_)) * tilesize_;
        real cy = (pos.y + distReal_(engine_)) * tilesize_;

        centers[n] = Center(real3{cx, cy, 0}, centerIdx_);
        centerIdx_ += 1;
    }
    
    // Cache centers belonging to this tile
    Tile newTile;
    newTile.pos = pos;
    newTile.centers = centers;
    tileCache_[pos] = newTile;

    return centers;
}

int VoronoiTessellator::Poisson(real lambda) {
    std::poisson_distribution<int> dist(lambda);
    return dist(engine_);
}

Tile VoronoiTessellator::tileOfCell(real3 coo) {
    return Tile{int3{static_cast<int>(std::floor(coo.x / tilesize_)), // This cannot be the best way...
                     static_cast<int>(std::floor(coo.y / tilesize_)),
                     0}
            };
}

