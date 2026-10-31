// mat(): a MatShape of the kernel-choice tests, moved verbatim from tests/ops/ops_tests.cpp (arch025-n8b1).
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "whirl/gguf.h"
#include "whirl/ops/matmul.h"

#include <cstdint>

namespace {

namespace ops = whirl::ops;
using whirl::gguf::GgmlType;

inline ops::MatShape mat(GgmlType t, std::uint32_t rows, std::uint32_t cols, bool head = false, bool ref = false) {
    ops::MatShape m;
    m.ty = t;
    m.nrows = rows;
    m.ncols = cols;
    m.head = head;
    m.has_ref = ref;
    return m;
}

}  // namespace
