// The verify widths (tokens per launch) the ggemv bench times (whirl-kernel-test --bench ggemv, feat026-ggemv /
// v0251-a6bench). ONE table, derived from the runtime: ops::chooseGemvTile takes a gemvk kernel for
// 2 <= n <= kernels::kGemvkMaxNt (src/ops/matmul.cpp), the loader looks the entries up for the same range
// (src/kernels_host/kernels_abi.cpp) and kernels/gfx1151/gemv_gk.hip instantiates exactly those widths. So the bench
// can never time fewer widths than WHIRL_GEMV_K can select (the A6 first round timed nt 2 / 4 only while Swift's MTP
// verify ran 5-6 rows). --bench-nt picks a SUBSET of this table (to split a GPU round); a width outside it is a usage
// error, never a second list. Pinned by whirl-model-tests (testGgemvBenchWidths: table, MTP widths, parse, route).
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "whirl/kernels_abi.h"

#include <string>
#include <string_view>
#include <vector>

namespace kt {

inline std::vector<int> ggemvBenchNts() {
    return {2, 4};  // RED (test-first state): the A6 first round's literal list
}

// "--bench-nt 5,6": the listed widths, in table order, each of which must be in ggemvBenchNts(). Returns false (and
// err) for an empty list, a non-number, a duplicate or a width outside the table.
inline bool parseBenchNt(std::string_view list, std::vector<int>& out, std::string& err) {
    const std::vector<int> table = ggemvBenchNts();
    std::vector<bool> want(static_cast<std::size_t>(whirl::kernels::kGemvkMaxNt) + 1, false);
    std::size_t p = 0;
    int n_items = 0;
    while (p <= list.size()) {
        std::size_t e = list.find(',', p);
        if (e == std::string_view::npos) e = list.size();
        const std::string item(list.substr(p, e - p));
        p = e + 1;
        if (item.empty()) {
            if (e == list.size()) break;
            err = "empty item in --bench-nt '" + std::string(list) + "'";
            return false;
        }
        int nt = 0;
        for (const char ch : item) {
            if (ch < '0' || ch > '9' || nt > 1000) {
                err = "--bench-nt item '" + item + "' is not a width";
                return false;
            }
            nt = nt * 10 + (ch - '0');
        }
        bool in_table = false;
        for (const int t : table) in_table = in_table || t == nt;
        if (!in_table) {
            err = "--bench-nt width " + item + " is not a gemvk width (the runtime has 2.." + std::to_string(whirl::kernels::kGemvkMaxNt) + ")";
            return false;
        }
        if (want[static_cast<std::size_t>(nt)]) {
            err = "--bench-nt width " + item + " listed twice";
            return false;
        }
        want[static_cast<std::size_t>(nt)] = true;
        ++n_items;
    }
    if (n_items == 0) {
        err = "--bench-nt needs at least one width";
        return false;
    }
    out.clear();
    for (const int t : table)
        if (want[static_cast<std::size_t>(t)]) out.push_back(t);
    return true;
}

inline std::string joinNts(const std::vector<int>& v) {
    std::string s;
    for (const int t : v) s += (s.empty() ? "" : ",") + std::to_string(t);
    return s;
}

}  // namespace kt
