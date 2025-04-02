#include <iostream>

#include "examples.hpp"

int main() {
  int example_number = 0;

  std::cout << "************ Mumax C++ Examples ************\n"
            << "1. Standard Problem 4.\n"
            << "2. Spinwave Dispersion.\n"
            << "*******************************************\n"
            << "Print example number to run it: ";

  std::cin >> example_number;

  switch (example_number) {
    case 1:
      standard_problem4();
      break;
    case 2:
      spinwave_dispersion();
      break;
    default:
      std::cout << "Wrong number, try again.\n";
      break;
  }
}
