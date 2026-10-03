# Test-only chat-template oracle built from an EXTERNAL llama.cpp checkout
# (WHIRL_LLAMACPP_SRC). No llama.cpp code is copied into WHIRL; the binary
# is only used by `whirl-parity template` (tests/parity) and is never shipped.
# SPDX-License-Identifier: Apache-2.0

set(LL "${WHIRL_LLAMACPP_SRC}")
if(NOT EXISTS "${LL}/common/jinja/runtime.cpp")
  message(FATAL_ERROR "WHIRL_LLAMACPP_SRC does not look like a llama.cpp tree: ${LL}")
endif()

add_executable(whirl-template-oracle
  oracle.cpp
  "${LL}/common/jinja/caps.cpp"
  "${LL}/common/jinja/lexer.cpp"
  "${LL}/common/jinja/parser.cpp"
  "${LL}/common/jinja/runtime.cpp"
  "${LL}/common/jinja/string.cpp"
  "${LL}/common/jinja/value.cpp"
  "${LL}/common/json.cpp"
  "${LL}/common/unicode.cpp")
target_include_directories(whirl-template-oracle PRIVATE
  "${LL}/common" "${LL}/vendor" "${LL}/ggml/include")
if(MSVC)
  target_compile_options(whirl-template-oracle PRIVATE /W0 /bigobj)
endif()
