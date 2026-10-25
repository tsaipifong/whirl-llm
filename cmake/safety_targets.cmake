# WHIRL - safe025 test-only fault driver (after cmake/server.cmake: needs whirl_device_null). Option default OFF.
# SPDX-License-Identifier: Apache-2.0
option(WHIRL_SAFETY_FAULTS "Build whirl-safety-faults, the injected-fault driver for the safety mechanisms (test only)" OFF)
if(WHIRL_SAFETY_FAULTS)
  add_executable(whirl-safety-faults ${CMAKE_SOURCE_DIR}/tests/safety/safety_faults.cpp)
  target_link_libraries(whirl-safety-faults PRIVATE whirl_device_null)
endif()
