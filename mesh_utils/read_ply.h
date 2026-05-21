#pragma once

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ── PLY loader ────────────────────────────────────────────────────────────────
struct Mesh
{
    std::vector<float> verts;
    std::vector<uint32_t> faces;
};

Mesh load_ply(const std::string &path);