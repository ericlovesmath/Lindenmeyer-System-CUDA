#include "cpu/expand.h"

namespace {

std::string rewrite(const l_system &sys, const std::string &s) {
  std::string out;
  out.reserve(s.size() * 2);
  for (char symbol : s) {
    auto rule = sys.rules.find(symbol);
    if (rule != sys.rules.end()) {
      out += rule->second;
    } else {
      out += symbol;
    }
  }
  return out;
}

} // namespace

std::string expand(const l_system &sys, int iterations) {
  std::string s = sys.axiom;
  for (int i = 0; i < iterations; ++i) {
    s = rewrite(sys, s);
  }
  return s;
}
