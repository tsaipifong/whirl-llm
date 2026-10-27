# WHIRL - device module (src/device, ARCH-1 D-2..D-5)
# SPDX-License-Identifier: Apache-2.0
#
# Included from cmake/server.cmake (before tier.cmake, whose tests link it).
#   whirl_device_null : device::NullBackend, host memory, no HIP (unit tests of the engine / tiers)
#   whirl_device      : device::Backend over HIP (whirl.exe, whirl-server.exe)
#   whirl-device-tests: the module's own unit tests (CPU)

add_library(whirl_device_null STATIC ${CMAKE_SOURCE_DIR}/src/device/null_backend.cpp)
target_include_directories(whirl_device_null PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(whirl_device_null PUBLIC whirl_platform)  # platform::env::isSet (WHIRL_LEAKCHECK_QUIET)

if(WHIRL_WITH_HIP)
  add_library(whirl_device STATIC ${CMAKE_SOURCE_DIR}/src/device/backend_hip.cpp)
  target_include_directories(whirl_device PUBLIC ${CMAKE_SOURCE_DIR}/src ${CMAKE_SOURCE_DIR}/include)
  target_link_libraries(whirl_device PUBLIC whirl)
endif()

add_executable(whirl-device-tests ${CMAKE_SOURCE_DIR}/tests/device/device_tests.cpp)
target_link_libraries(whirl-device-tests PRIVATE whirl_device_null)
