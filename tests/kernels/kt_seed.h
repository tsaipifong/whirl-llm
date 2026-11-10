// kt: one independent random stream per test family (v0252-ktrng, PLAN B26-53 (c)). HIP-free, header only.
// Before: Ctx held ONE std::mt19937_64{12345} that every family drew from in run order, so adding a check (or a family,
// or running with --only) shifted the inputs of every later check (found 11-05 by v0251-ktfull). Now the driver calls
// Ctx::beginFamily(name) before each family: the stream restarts from familySeed(name), a pure function of the name
// (FNV-1a 64 of the bytes, then a splitmix64 finaliser; llama.cpp test-backend-ops tensor_seed hashes the tensor name
// with FNV-1a the same way). Families that call other families (moemx -> router / gemm checks) share their caller's stream.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <random>
#include <string>

namespace kt {

inline std::uint64_t familySeed(const std::string& name) {
    std::uint64_t h = 1469598103934665603ULL;  // FNV-1a 64 offset basis
    for (const unsigned char ch : name) {
        h ^= ch;
        h *= 1099511628211ULL;
    }
    h += 0x9e3779b97f4a7c15ULL;  // splitmix64 finaliser: spreads the low-entropy FNV bits
    h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
    h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
    return h ^ (h >> 31);
}

struct FamilyRng {
    std::mt19937_64 rng{12345};  // the pre-v0252-ktrng default, used only before the first begin()
    void begin(const std::string& family) { rng.seed(familySeed(family)); }
};

}  // namespace kt
