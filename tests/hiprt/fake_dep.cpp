// whirl_fake_dep.dll: the import of the WHIRL_FAKE_WITH_DEP fake runtime
// (tests/hiprt/fake_hiprt.cpp), standing in for rocm_kpack.dll.
// SPDX-License-Identifier: Apache-2.0
//
// Written for WHIRL.

extern "C" __declspec(dllexport) int whirl_fake_dep_fn(void) { return 3; }
