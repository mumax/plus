#pragma once

#include <chrono>  // NOLINT [build/c++11]
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

class Timer {
 public:
  Timer() = default;
  void start(std::string clockName) {
    if (clocks.find(clockName) == clocks.end()) clocks[clockName] = Clock();
    clocks[clockName].start();
  }

  void stop(std::string clockName) {
    if (clocks.find(clockName) == clocks.end())
      throw std::runtime_error(
          "Can not stop clock, because clock does not exists");
    clocks[clockName].stop();
  }

  void printTimings() {
    for (auto namedClock : clocks) {
      std::string name = namedClock.first;
      Clock clock = namedClock.second;
      std::cout << name << "\t";
      std::cout << clock.invocations << "\t";
      std::cout << clock.total.count() << "\t";
      std::cout << clock.total.count() / clock.invocations << "\t";
      std::cout << std::endl;
    }
  }

 private:
  struct Clock {
    Clock() : invocations(0), total(0) {}
    std::chrono::microseconds total;
    std::chrono::high_resolution_clock::time_point started;
    int invocations;
    void start() { started = std::chrono::high_resolution_clock::now(); }
    void stop() {
      auto finish = std::chrono::high_resolution_clock::now();
      total += std::chrono::duration_cast<std::chrono::microseconds>(finish -
                                                                     started);
      invocations++;
    }
  };

  std::map<std::string, Clock> clocks;
};

extern Timer timer;
