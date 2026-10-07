// Kernel table for the model: the table itself is whirl/kernels_abi.h
// (KernelTable::load); this maps the model's KV-format flags onto it.
// SPDX-License-Identifier: Apache-2.0

#include "whirl/model.h"

namespace whirl::qwen35 {

Kernels loadKernels(const hip::Module& m, bool q8, bool rot, bool kf16, bool q4) {
    kernels::KvFormat kv = kernels::KvFormat::f16;
    if (q4)
        kv = kernels::KvFormat::q4;
    else if (q8 && kf16)
        kv = kernels::KvFormat::q8v;
    else if (q8)
        kv = rot ? kernels::KvFormat::q8h : kernels::KvFormat::q8;
    return Kernels::load(m, kv);
}

}  // namespace whirl::qwen35
