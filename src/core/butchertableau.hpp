#pragma once

#include <map>
#include <string>
#include <vector>

#include "datatypes.hpp"

enum class RKmethod {
  HEUN,
  BOGACKI_SHAMPINE,
  CASH_KARP,
  FEHLBERG,
  DORMAND_PRINCE
};

const std::map<RKmethod, std::string> RungeKuttaMethodNames{
    // clang-format off
    {RKmethod::HEUN, "Heun"},
    {RKmethod::BOGACKI_SHAMPINE, "BogackiShampine"},
    {RKmethod::CASH_KARP, "CashKarp"},
    {RKmethod::FEHLBERG, "Fehlberg"},
    {RKmethod::DORMAND_PRINCE, "DormandPrince"},
    // clang-format on
};

RKmethod getRungeKuttaMethodFromName(const std::string& name);

/// Extended Butcher Tableau for Adaptive Runge-Kutta methods
class ButcherTableau {
 public:
  static const ButcherTableau& get(RKmethod);
  static const ButcherTableau Heun;
  static const ButcherTableau BogackiShampine;
  static const ButcherTableau CashKarp;
  static const ButcherTableau Fehlberg;
  static const ButcherTableau DormandPrince;

 public:
  ButcherTableau(std::vector<real> nodes,
                 std::vector<std::vector<real>> rkMatrix,
                 std::vector<real> weights1,
                 std::vector<real> weights2,
                 int order1,
                 int order2);

  bool isConsistent() const;

  const std::vector<real> nodes;
  const std::vector<std::vector<real>> rkMatrix;
  const std::vector<real> weights1;
  const std::vector<real> weights2;
  const int order1;
  const int order2;
};
