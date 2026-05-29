#include "lsystem.h"

std::string expand(const l_system &sys, int iterations) {
  std::string current = sys.axiom;
  for (int i = 0; i < iterations; ++i) {
    std::string next;
    next.reserve(current.size() * 2);
    for (char symbol : current) {
      auto rule = sys.rules.find(symbol);
      if (rule != sys.rules.end()) {
        next += rule->second;
      } else {
        next += symbol;
      }
    }
    current = std::move(next);
  }
  return current;
}
