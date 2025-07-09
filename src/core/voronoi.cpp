#include <math.h>
#include <random>
#include <cmath>

#include "field.hpp"
#include "gpubuffer.hpp"
#include "voronoi.hpp"

VoronoiTessellator::VoronoiTessellator(real grainsize,
                                       int seed,
                                       unsigned int maxIdx,
                                       const std::function<unsigned int(real3)>& centerIdx)
    : grainsize_(grainsize),
      seed_(seed),
      distReal_(0.0, 1.0),
      distInt_(0, maxIdx),
      numTiles_x(1),
      numTiles_y(1),
      numTiles_z(1),
      pbc_(false),
      is2D_(false) {
        if (centerIdx) { centerIdx_ = centerIdx; }
        else
            centerIdx_ = [this](real3 coo) -> unsigned int { return distInt_(engine_); };
    }
    
real3 VoronoiTessellator::getTileSize(const real3 griddims) const {
    if (!pbc_)
        return real3{2 * grainsize_, 2 * grainsize_ , 2 * grainsize_ * (!is2D_)};

    return griddims / real3{std::max(real(1), std::ceil(griddims.x / (2 * grainsize_))),
                            std::max(real(1), std::ceil(griddims.y / (2 * grainsize_))),
                            std::max(real(1), std::ceil(griddims.z / (2 * grainsize_)))};
}

std::vector<unsigned int> VoronoiTessellator::generate(const Grid grid,
                                                       const real3 cellsize,
                                                       const bool pbc,
                                                       const bool make2D) {
    pbc_ = pbc;
    is2D_ = make2D ? make2D : (grid.size().z == 1);

    grid_dims_ = real3{grid.size().x * cellsize.x,
                       grid.size().y * cellsize.y,
                       (!is2D_) * grid.size().z * cellsize.z};

    tilesize_ = getTileSize(grid_dims_);

    lambda_ =   (tilesize_.x / grainsize_)
              * (tilesize_.y / grainsize_)
              * (is2D_ ? 1 : (tilesize_.z / grainsize_));
    if (pbc) {
        numTiles_x = std::ceil(grid_dims_.x / tilesize_.x);
        numTiles_y = std::ceil(grid_dims_.y / tilesize_.y);
        numTiles_z = is2D_ ? 1 : std::ceil(grid_dims_.z / tilesize_.z);
    }

    std::vector<unsigned int> data(grid.ncells());
    for (int nz = 0; nz < grid.size().z; nz++) {
        for (int ny = 0; ny < grid.size().y; ny++) {
            for (int nx = 0; nx < grid.size().x; nx++) {
                real3 coo = real3{nx * cellsize.x,
                                  ny * cellsize.y,
                                  nz * cellsize.z};
                size_t idx = nx + grid.size().x * (ny + grid.size().y * nz);
                data[idx] = regionOf(coo);
            }
        }
    }
    return data;
}

unsigned int VoronoiTessellator::regionOf(const real3 coo) {
    Tile t = tileOfCell(coo);
    Center nearest = Center{coo, 0};
    real mindist = INFINITY;

    int tz_start = t.pos.z - 1;
    int tz_end   = t.pos.z + 1;
    if (is2D_)
        tz_start = tz_end = t.pos.z;

    for (int tx = t.pos.x-1; tx <= t.pos.x+1; tx++) {
        for (int ty = t.pos.y-1; ty <= t.pos.y+1; ty++) {
            for (int tz = tz_start; tz <= tz_end; tz++) {
                int wrapped_tx = pbc_ ? (tx + numTiles_x) % numTiles_x : tx;
                int wrapped_ty = pbc_ ? (ty + numTiles_y) % numTiles_y : ty;
                int wrapped_tz = pbc_ ? (tz + numTiles_z) % numTiles_z : tz;

                std::vector<Center> centers = centersInTile(int3{wrapped_tx, wrapped_ty, wrapped_tz});
                for (auto c : centers) {
                    real3 center = pbc_ ? (periodicShift(coo, c.pos)) : c.pos;
                    real dist = (coo.x - center.x) * (coo.x - center.x)
                              + (coo.y - center.y) * (coo.y - center.y)
                              + (coo.z - center.z) * (coo.z - center.z);
                    if (dist < mindist) {
                        nearest = c;
                        mindist = dist;
                    }
                }
            }
        }
    }
    return nearest.ridx;
}

real3 VoronoiTessellator::periodicShift(const real3 coo, real3 center) {
    real3 d = center - coo;
    if (std::abs(d.x) > 0.5 * grid_dims_.x) center.x -= std::copysign(grid_dims_.x, d.x);
    if (std::abs(d.y) > 0.5 * grid_dims_.y) center.y -= std::copysign(grid_dims_.y, d.y);
    if (std::abs(d.z) > 0.5 * grid_dims_.z) center.z -= std::copysign(grid_dims_.z, d.z);
    return center;
}

std::vector<Center> VoronoiTessellator::centersInTile(const int3 pos) {

    // Check if centers in this tile are already cached
    auto it = tileCache_.find(pos);
    if (it != tileCache_.end()) {
        return it->second.centers;
    }

    const int64_t seed = ((int64_t(pos.x) + (1LL << 20)) * (1LL << 40)) +
                         ((int64_t(pos.y) + (1LL << 20)) * (1LL << 20)) +
                         ( int64_t(pos.z) + (1LL << 20));

    engine_.seed (seed ^ seed_);
    int N = Poisson(lambda_);
    std::vector<Center> centers(N);

    for (int n = 0; n < N; n++) {
        real cx = (pos.x + distReal_(engine_)) * tilesize_.x;
        real cy = (pos.y + distReal_(engine_)) * tilesize_.y;
        real cz = (pos.z + distReal_(engine_)) * tilesize_.z;

        centers[n] = Center(real3{cx, cy, cz}, centerIdx_(real3{cx, cy, cz}));
    }
    
    // Cache centers belonging to this tile
    Tile newTile;
    newTile.pos = pos;
    newTile.centers = centers;
    tileCache_[pos] = newTile;

    return centers;
}

int VoronoiTessellator::Poisson(const real lambda) {
    std::poisson_distribution<int> dist(lambda);
    return dist(engine_);
}

Tile VoronoiTessellator::tileOfCell(const real3 coo) const {
    return Tile{int3{static_cast<int>(std::floor(coo.x / tilesize_.x)), // This cannot be the best way...
                     static_cast<int>(std::floor(coo.y / tilesize_.y)),
                     is2D_ ? 0 : static_cast<int>(std::floor(coo.z / tilesize_.z))}
               };
}