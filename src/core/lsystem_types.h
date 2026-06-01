#pragma once

#include <string>
#include <unordered_map>

// An L-system grammar G = (V, omega, P)
struct l_system {
  std::string axiom;
  std::unordered_map<char, std::string> rules;
};
