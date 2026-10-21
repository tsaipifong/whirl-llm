# REVIEW-GOV-2 test targets (deferred to the end of the tree's top-level CMakeLists.txt)
# SPDX-License-Identifier: Apache-2.0
add_executable(review-gov2-unit ${REVIEW_GOV2_DIR}/gov_unit_tests.cpp)
target_link_libraries(review-gov2-unit PRIVATE whirl)

add_executable(review-gov2-restore
  ${REVIEW_GOV2_DIR}/gov_restore_tests.cpp
  ${REVIEW_GOV2_DIR}/mock/mock_backend.cpp
  ${REVIEW_GOV2_DIR}/mock/http_client.cpp
  ${REVIEW_GOV2_DIR}/vision_stub.cpp
  ${CMAKE_SOURCE_DIR}/src/model/spec.cpp)
target_include_directories(review-gov2-restore PRIVATE ${REVIEW_GOV2_DIR}/mock)
target_link_libraries(review-gov2-restore PRIVATE whirl_server_core)
