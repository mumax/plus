#pragma once

#include <memory>
#include <vector>

#include "field.hpp"

/// a1*x1 + a2*x2
Field add(real a1, const Field& x1, real a2, const Field& x2);

/// x1 + x2
Field add(const Field& x1, const Field& x2);

/// y += a*x
void addTo(Field& y, real a, const Field& x);

/// sum_i( weight[i]*x[i] )
Field add(std::vector<const Field*> x, std::vector<real> weights);

Field operator*(real a, const Field& x);

Field operator*(real3 a, const Field& x);

Field normalized(const Field& src);
void normalize(Field&);
