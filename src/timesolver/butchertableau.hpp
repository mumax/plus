#pragma once

#include <vector>

#include "datatypes.hpp"

enum RKmethod { HEUN, BOGACKISHAMPINE, CASHKARP, FEHLBERG, DORMANDPRINCE };

/// Extended Butcher Tableau for Adaptive Runge-Kutta methods
class ButcherTableau {
 public:
  ButcherTableau(std::vector<real> nodes,
                 std::vector<std::vector<real>> rkMatrix,
                 std::vector<real> weights1,
                 std::vector<real> weights2,
                 int order1,
                 int order2);

  ButcherTableau(RKmethod method);

  bool isConsistent() const;

  const std::vector<real> nodes;
  const std::vector<std::vector<real>> rkMatrix;
  const std::vector<real> weights1;
  const std::vector<real> weights2;
  const int nStages;
  const int order1;
  const int order2;
};

ButcherTableau constructTableau(RKmethod);
ButcherTableau constructHeunTableau();
ButcherTableau constructBogackiShampineTableau();
ButcherTableau constructCashKarpTableau();
ButcherTableau constructFehlbergTableau();
ButcherTableau constructDormandPrinceTableau();