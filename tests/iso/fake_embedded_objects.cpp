// Link double of the build-generated code-object table (src/hip/embedded_objects.cpp.in, bin2c arrays of the HIP kernels) for the
// isolation programs of hip.runtime / hip.mem (ARCH-1 8.5, arch025-iso4). hip_runtime.cpp's module loader walks hip::embeddedObjects()
// only when a test loads kernels onto a device; the isolated programs never do, so the table is empty.
// SPDX-License-Identifier: Apache-2.0
#include "whirl/hip.h"

#include <span>

namespace whirl::hip {

std::span<const EmbeddedObject> embeddedObjects() { return {}; }

}  // namespace whirl::hip
