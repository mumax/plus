#include <math.h>
#include <random>
#include <cmath>

#include "field.hpp"
#include "gpubuffer.hpp"
#include "voronoi.hpp"

VoronoiTessellator::VoronoiTessellator(real grainsize, unsigned int maxIdx, int seed)
    : grainsize_(grainsize),
      seed_(seed),
      distReal_(0.0, 1.0),
      distInt_(0, maxIdx) {
        pbc_ = false;
      }

real3 VoronoiTessellator::getTileSize(real3 griddims) {

    if (pbc_)
        return real3{2 * grainsize_, 2 * grainsize_ , 0};

    return griddims / real3{std::max(1.0f, std::ceil(griddims.x / (2 * grainsize_))),
                            std::max(1.0f, std::ceil(griddims.y / (2 * grainsize_))),
                            0};
}

GpuBuffer<unsigned int> VoronoiTessellator::generate(Grid grid, real3 cellsize, const bool pbc) {

   pbc_ = pbc;
   grid_dims_ = real3{grid.size().x * cellsize.x, grid.size().y * cellsize.y, 0};

   tilesize_ = getTileSize(grid_dims_);

   lambda_ = (tilesize_.x / grainsize_) * (tilesize_.y / grainsize_);

   if (pbc) {
    numTiles_x = static_cast<int>(std::ceil(grid_dims_.x / tilesize_.x));
    numTiles_y = static_cast<int>(std::ceil(grid_dims_.y / tilesize_.y));
   }

   std::vector<unsigned int> data(grid.ncells());

   for (int nx = 0; nx < grid.size().x; nx++) {
        for (int ny = 0; ny < grid.size().y; ny++) {
            real3 coo = real3{nx * cellsize.x,
                              ny * cellsize.y,
                              0};
            data[nx + grid.size().x * ny] = regionOf(coo);
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

            int wrapped_tx = pbc_ ? (tx + numTiles_x) % numTiles_x : tx;
            int wrapped_ty = pbc_ ? (ty + numTiles_y) % numTiles_y : ty;

            std::vector<Center> centers = centersInTile(int3{wrapped_tx, wrapped_ty, 0});

            for (auto c : centers) {
                real3 center = pbc_ ? (periodicShift(coo, c.pos)) : c.pos;

                real dist = (coo.x - center.x) * (coo.x - center.x)
                          + (coo.y - center.y) * (coo.y - center.y);

                if (dist < mindist) {
                    nearest = c;
                    mindist = dist;
                }
            }
        }
    }
    return nearest.ridx;
}

real3 VoronoiTessellator::periodicShift(real3 coo, real3 center) {
    real3 d = center - coo;
    d.x -= grid_dims_.x * (std::abs(d.x) > 0.5 * grid_dims_.x ? std::copysign(1.0, d.x) : 0);
    d.y -= grid_dims_.y * (std::abs(d.y) > 0.5 * grid_dims_.y ? std::copysign(1.0, d.y) : 0);
    return coo + d;
}

std::vector<Center> VoronoiTessellator::centersInTile(int3 pos) {

    // Check if centers in this tile are already cached
    auto it = tileCache_.find(pos);
    if (it != tileCache_.end()) {
        return it->second.centers;
    }

    int64_t seed = (int64_t(pos.y) + (1<<24)) * (1<<24)
                 + (int64_t(pos.x) + (1<<24));
    engine_.seed (seed ^ seed_);

    int N = Poisson(lambda_);

    std::vector<Center> centers(N);

    for (int n = 0; n < N; n++) {
        real cx = (pos.x + distReal_(engine_)) * tilesize_.x;
        real cy = (pos.y + distReal_(engine_)) * tilesize_.y;

        centers[n] = Center(real3{cx, cy, 0}, distInt_(engine_));
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
    return Tile{int3{static_cast<int>(std::floor(coo.x / tilesize_.x)), // This cannot be the best way...
                     static_cast<int>(std::floor(coo.y / tilesize_.y)),
                     0}
            };
}