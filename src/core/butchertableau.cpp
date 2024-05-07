#include "butchertableau.hpp"

#include <math.h>

#include <algorithm>
#include <exception>
#include <string>

RKmethod getRungeKuttaMethodFromName(const std::string& name) {
  auto it =
      std::find_if(RungeKuttaMethodNames.begin(), RungeKuttaMethodNames.end(),
                   [name](const auto& kv) { return kv.second == name; });

  if (it == RungeKuttaMethodNames.end())
    throw std::invalid_argument("'" + name +
                                "' is not a valid Runge Kutta method name");

  RKmethod method = it->first;
  return method;
}

std::string getRungeKuttaNameFromMethod(const RKmethod& method) {
  auto pos = RungeKuttaMethodNames.find(method);
  if (pos == RungeKuttaMethodNames.end()) {
    throw std::invalid_argument("Got an invalid RKmethod");
  }
  return pos->second;
}

ButcherTableau::ButcherTableau(std::vector<real> nodes,
                               std::vector<std::vector<real>> rkMatrix,
                               std::vector<real> weights1,
                               std::vector<real> weights2,
                               int order1,
                               int order2)
    : nodes(nodes),
      rkMatrix(rkMatrix),
      weights1(weights1),
      weights2(weights2),
      order1(order1),
      order2(order2) {
  if (!isConsistent())
    throw std::invalid_argument(
        "Arguments to construct the ButcherTableau are inconsistent");
}

bool ButcherTableau::isConsistent() const {
  int N = nodes.size();

  if (rkMatrix.size() != N || weights1.size() != N || weights2.size() != N)
    return false;

  for (int i = 0; i < N; i++) {
    if (rkMatrix[i].size() != i)
      return false;

    real rowSum = 0.;
    for (int j = 0; j < i; j++)
      rowSum += rkMatrix[i][j];
    if (fabs(rowSum - nodes[i]) > 1e-5)  // TODO: do this better
      return false;
  }

  // TODO: check if orders are consistent

  return true;
}

const ButcherTableau& ButcherTableau::get(RKmethod method) {
  switch (method) {
    case RKmethod::HEUN:
      return ButcherTableau::Heun;
    case RKmethod::BOGACKI_SHAMPINE:
      return ButcherTableau::BogackiShampine;
    case RKmethod::CASH_KARP:
      return ButcherTableau::CashKarp;
    case RKmethod::FEHLBERG:
      return ButcherTableau::Fehlberg;
    case RKmethod::DORMAND_PRINCE:
      return ButcherTableau::DormandPrince;
    default:  // TODO: handle this better
      std::cerr << "Method is not implemented, Dormand Prince method is "
                   "used instead"
                << std::endl;
      return ButcherTableau::DormandPrince;
  }
}

const ButcherTableau ButcherTableau::Heun = []() {
  int order1 = 2;
  int order2 = 1;
  std::vector<real> nodes = {0., 1.};
  std::vector<std::vector<real>> rkMatrix = {{}, {1.}};
  std::vector<real> weights1 = {1. / 2., 1. / 2.};
  std::vector<real> weights2 = {1., 0.};
  return ButcherTableau(nodes, rkMatrix, weights1, weights2, order1, order2);
}();

const ButcherTableau ButcherTableau::BogackiShampine = []() {
  int N = 4;
  int order1 = 3;
  int order2 = 2;
  std::vector<real> nodes = {0., 1. / 2., 3. / 4., 1.};
  std::vector<std::vector<real>> rkMatrix(N);
  rkMatrix[0] = {};
  rkMatrix[1] = {1. / 2.};
  rkMatrix[2] = {0., 3. / 4.};
  rkMatrix[3] = {2. / 9., 1. / 3, 4. / 9.};
  std::vector<real> weights1 = {2. / 9., 1. / 3., 4. / 9., 0.};
  std::vector<real> weights2 = {7. / 24, 1. / 4., 1. / 3., 1. / 8.};
  return ButcherTableau(nodes, rkMatrix, weights1, weights2, order1, order2);
}();

const ButcherTableau ButcherTableau::Fehlberg = []() {
  int N = 6;
  int order1 = 5;
  int order2 = 4;
  std::vector<real> nodes(N);
  nodes[0] = 0.;
  nodes[1] = 1. / 4.;
  nodes[2] = 3. / 8.;
  nodes[3] = 12. / 13.;
  nodes[4] = 1.;
  nodes[5] = 1. / 2.;
  std::vector<std::vector<real>> rkMatrix(N);
  rkMatrix[0] = {};
  rkMatrix[1] = {1. / 4.};
  rkMatrix[2] = {3. / 32., 9. / 32.};
  rkMatrix[3] = {1932. / 2197., -7200. / 2197., 7296. / 2197.};
  rkMatrix[4] = {439. / 216., -8., 3680. / 513., -845. / 4104.};
  rkMatrix[5] = {-8. / 27., 2., -3544. / 2565., 1859. / 4104., -11. / 40.};
  std::vector<real> weights1(N);
  weights1[0] = 16. / 135.;
  weights1[1] = 0.;
  weights1[2] = 6656. / 12825.;
  weights1[3] = 28561. / 56430.;
  weights1[4] = -9. / 50.;
  weights1[5] = 2. / 55.;
  std::vector<real> weights2(N);
  weights2[0] = 25. / 216.;
  weights2[1] = 0.;
  weights2[2] = 1408. / 2565.;
  weights2[3] = 2197. / 4104.;
  weights2[4] = -1 / 5.;
  weights2[5] = 0.;
  return ButcherTableau(nodes, rkMatrix, weights1, weights2, order1, order2);
}();

const ButcherTableau ButcherTableau::CashKarp = []() {
  int N = 6;
  int order1 = 5;
  int order2 = 4;
  std::vector<real> nodes(N);
  nodes[0] = 0.;
  nodes[1] = 1. / 5.;
  nodes[2] = 3. / 10.;
  nodes[3] = 3. / 5.;
  nodes[4] = 1.;
  nodes[5] = 7. / 8.;
  std::vector<std::vector<real>> rkMatrix(N);
  rkMatrix[0] = {};
  rkMatrix[1] = {1. / 5.};
  rkMatrix[2] = {3. / 40., 9. / 40.};
  rkMatrix[3] = {3. / 10., -9. / 10., 6. / 5.};
  rkMatrix[4] = {-11. / 54., 5. / 2., -70. / 27., 35. / 27.};
  rkMatrix[5] = {1631. / 55296., 175. / 512., 575. / 13824., 44275. / 110592.,
                 253. / 4096.};
  std::vector<real> weights1(N);
  weights1[0] = 37. / 378.;
  weights1[1] = 0.;
  weights1[2] = 250. / 621.;
  weights1[3] = 125. / 594.;
  weights1[4] = 0.;
  weights1[5] = 512. / 1771.;
  std::vector<real> weights2(N);
  weights2[0] = 2825. / 27648.;
  weights2[1] = 0;
  weights2[2] = 18575. / 48384.;
  weights2[3] = 13525. / 55296.;
  weights2[4] = 277. / 14336.;
  weights2[5] = 1. / 4.;
  return ButcherTableau(nodes, rkMatrix, weights1, weights2, order1, order2);
}();

const ButcherTableau ButcherTableau::DormandPrince = []() {
  int N = 7;
  int order1 = 6;
  int order2 = 5;
  std::vector<real> nodes(N);
  nodes[0] = 0.;
  nodes[1] = 1. / 5.;
  nodes[2] = 3. / 10.;
  nodes[3] = 4. / 5.;
  nodes[4] = 8. / 9.;
  nodes[5] = 1.;
  nodes[6] = 1.;
  std::vector<std::vector<real>> rkMatrix(N);
  rkMatrix[0] = {};
  rkMatrix[1] = {1. / 5.};
  rkMatrix[2] = {3. / 40., 9. / 40.};
  rkMatrix[3] = {44. / 45., -56. / 15., 32. / 9.};
  rkMatrix[4] = {19372. / 6561., -25360. / 2187., 64448. / 6561., -212. / 729.};
  rkMatrix[5] = {9017. / 3168., -355. / 33., 46732. / 5247., 49. / 176.,
                 -5103. / 18656.};
  rkMatrix[6] = {35. / 384.,     0.,       500. / 1113., 125. / 192.,
                 -2187. / 6784., 11. / 84.};
  std::vector<real> weights1(N);
  weights1[0] = 35. / 384.;
  weights1[1] = 0.;
  weights1[2] = 500. / 1113.;
  weights1[3] = 125. / 192.;
  weights1[4] = -2187. / 6784.;
  weights1[5] = 11. / 84.;
  weights1[6] = 0.;
  std::vector<real> weights2(N);
  weights2[0] = 5179. / 57600.;
  weights2[1] = 0.;
  weights2[2] = 7571. / 16695.;
  weights2[3] = 393. / 640.;
  weights2[4] = -92097. / 339200.;
  weights2[5] = 187. / 2100.;
  weights2[6] = 1. / 40.;
  return ButcherTableau(nodes, rkMatrix, weights1, weights2, order1, order2);
}();
