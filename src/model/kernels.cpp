// Kernel table for the model: the table itself is whirl/kernels_abi.h
// (KernelTable::load) for the model's KV format.
// SPDX-License-Identifier: Apache-2.0

#include "whirl/model.h"

namespace whirl::qwen35 {

Kernels loadKernels(const hip::Module& m, KvFormat kv) { return Kernels::load(m, kv); }

}  // namespace whirl::qwen35
