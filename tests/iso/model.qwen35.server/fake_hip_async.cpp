// Link double of the asynchronous upload of the device runtime (src/hip) that the vision encoder (vision::Vision::encode,
// src/vision/vision.cpp) references, for the isolation program of model.qwen35.server only (ARCH-1 8.5, arch025-n8b3). Kept out
// of tests/iso/fake_hip.cpp on purpose: that double defines only what the forward / model core reference, so a forward source
// that starts to call it is still LNK2019 in their programs. Like fake_hip.cpp, a call that would reach the GPU throws (no
// case here encodes an image). downloadAsync moved to tests/iso/fake_hip.cpp (v0252-hostwait: the forward references it
// since Context::readCtlAsync; defining it in both files would be LNK2005 in this program).
// SPDX-License-Identifier: Apache-2.0
#include "whirl/hip.h"

#include <stdexcept>

namespace whirl::hip {

void uploadAsync(DevPtr, const void*, std::size_t, Stream) { throw std::logic_error("iso fake hip: uploadAsync reached the device runtime"); }

}  // namespace whirl::hip
