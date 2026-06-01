#include "cpu/examples.h"

// Koch snowflake
const example koch{
    "koch", {"F++F++F", {{'F', "F-F++F-F"}}}, 8, {1.0, 60.0, 0.0}, {0, 0, 0}};

// Fractal plant
const example plant{"plant",
                    {"-X", {{'X', "F+[[X]-X]-F[-FX]+X"}, {'F', "FF"}}},
                    6,
                    {1.0, 25.0, 90.0},
                    {30, 130, 40}};

// Heighway dragon
const example dragon{"dragon",
                     {"FX", {{'X', "X+YF+"}, {'Y', "-FX-Y"}}},
                     12,
                     {1.0, 90.0, 0.0},
                     {160, 30, 30}};

// Hilbert space-filling curve
const example hilbert{"hilbert",
                      {"X", {{'X', "+YF-XFX-FY+"}, {'Y', "-XF+YFY+FX-"}}},
                      9,
                      {1.0, 90.0, 0.0},
                      {30, 60, 160}};

// Sierpinski arrowhead curve
const example sierpinski{"sierpinski",
                         {"XF", {{'X', "YF+XF+Y"}, {'Y', "XF-YF-X"}}},
                         11,
                         {1.0, 60.0, 0.0},
                         {0, 0, 0}};
