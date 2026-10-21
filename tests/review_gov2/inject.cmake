# REVIEW-GOV-2: adds the reviewer's test targets to any WHIRL tree without editing it.
# cmake -S <tree> -B <build> -G Ninja -DWHIRL_WITH_HIP=OFF -DCMAKE_PROJECT_INCLUDE=<this file>
# SPDX-License-Identifier: Apache-2.0
set(REVIEW_GOV2_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL "")
cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}" CALL include "${REVIEW_GOV2_DIR}/targets.cmake")
