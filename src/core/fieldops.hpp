#pragma once

#include <memory>
#include <vector>

#include "field.hpp"

/// y = a1*x1 + a2*x2
void add(Field& y, real a1, const Field& x1, real a2, const Field& x2);
/// a1*x1 + a2*x2
Field add(real a1, const Field& x1, real a2, const Field& x2);

/// y = a1*x1 + a2*x2
void add(Field& y, real3 a1, const Field& x1, real3 a2, const Field& x2);
/// a1*x1 + a2*x2
Field add(real3 a1, const Field& x1, real3 a2, const Field& x2);

/// positional y = a1*x1 + a2*x2
void add(Field& y, const Field& a1, const Field& x1, const Field& a2, const Field& x2);
/// positional a1*x1 + a2*x2
Field add(const Field& a1, const Field& x1, const Field& a2, const Field& x2);

/// x1 + x2
Field add(const Field& x1, const Field& x2);
/// x1 + x2
Field operator+(const Field& x1, const Field& x2);

/// y += a*x
void addTo(Field& y, real a, const Field& x);

void addTo(Field& y, real3 a, const Field& x);

void addTo(Field& y, const Field& a, const Field& x);

/// sum_i( weight[i]*x[i] )
Field add(std::vector<const Field*> x, std::vector<real> weights);

Field operator*(real a, const Field& x);

Field operator*(real3 a, const Field& x);

/// positional multiplication
Field multiply(const Field& a, const Field& x);
/// positional x*y
Field operator*(const Field& a, const Field& x);

/// add constant to Field
Field addConstant(const Field& x, real value);

/// normalize Field
Field normalized(const Field& src);
void normalize(Field&);

/// Map vector field to RGB field
Field fieldGetRGB(const Field& src);
