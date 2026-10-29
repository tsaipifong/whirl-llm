// Link double of the build-generated vision code objects (cmake/vision.cmake: the hipcc-compiled kernels/vision/vision.hip per architecture, embedded by
// whirl-bin2c as whirl_vision_<arch>[] / whirl_vision_<arch>_size) for the isolation program of vision (ARCH-1 8.5, arch025-iso4). The isolated build has no
// device compiler: the table logic of src/vision/kernel_image.cpp (look an image up by hip::archName, one distinct object per family) is what the test
// runs, and the double gives each family a distinct 80-byte ELF64 header (64-byte header + 16 bytes) whose byte 0x30 (the low byte of e_flags, EF_AMDGPU_MACH) is the machine value
// of that family (0x4e gfx1201, 0x4a gfx1151). The real objects are pinned by whirl-vision-tests in the product build.
// SPDX-License-Identifier: Apache-2.0

// row 0: e_ident (ELF, 64-bit, little endian); row 1: e_type ET_DYN, e_machine 0xe0 (AMDGPU); row 2: zeros; row 3: e_flags (byte 0x30 = mach), e_ehsize 0x40
#define ISO_ELF_HEADER(mach)                                                                     \
    {0x7f, 'E', 'L', 'F', 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, /* 0x00 */                        \
     3, 0, 0xe0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       /* 0x10 */                        \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           /* 0x20 */                        \
     mach, 0, 0, 0, 0x40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,    /* 0x30 */                       \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}           /* 0x40: the table test wants more than the 64-byte header */

extern const unsigned char whirl_vision_gfx1201[] = ISO_ELF_HEADER(0x4e);
extern const unsigned long long whirl_vision_gfx1201_size = sizeof whirl_vision_gfx1201;
extern const unsigned char whirl_vision_gfx1151[] = ISO_ELF_HEADER(0x4a);
extern const unsigned long long whirl_vision_gfx1151_size = sizeof whirl_vision_gfx1151;
