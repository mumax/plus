#pragma once

#include <vector>

#include "datatypes.hpp"

enum class RKmethod { HEUN, BOGACKI_SHAMPINE, CASH_KARP, FEHLBERG, DORMAND_PRINCE };

/// Extended Butcher Tableau for Adaptive Runge-Kutta methods
class ButcherTableau {
  // I think we should add proper documentation for this class
 public:
  ButcherTableau(std::vector<real> nodes,
                 std::vector<std::vector<real>> rkMatrix,
                 std::vector<real> weights1,
                 std::vector<real> weights2,
                 int order1,
                 int order2);

  explicit ButcherTableau(RKmethod method);

  bool isConsistent() const;

  const std::vector<real> nodes;
  const std::vector<std::vector<real>> rkMatrix;
  const std::vector<real> weights1;
  const std::vector<real> weights2;
  const int nStages;
  const int order1; // what is order for?
  const int order2;
};

// why these are not part of the class?
// the first function and constructor can be merged???
// other functions seem like ideal candidates for private members
ButcherTableau constructTableau(RKmethod);
ButcherTableau constructHeunTableau();
ButcherTableau constructBogackiShampineTableau();
ButcherTableau constructCashKarpTableau();
ButcherTableau constructFehlbergTableau();
ButcherTableau constructDormandPrinceTableau();
