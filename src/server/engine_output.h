// Server engine, a slot's output (arch025-engine S5, module server.slot_output): emit (one token into the slot's batch;
// the end-of-turn / end-of-text token or max_tokens ends the request), flushOut (the batch through the slot's response
// writer; a matched stop string ends the request) and writeError (an HTTP error answer to a job). Free functions
// (progress/arch025-engine.DESIGN.md v3 2.3, layer L1): before S5 they were members of Engine (engine.h:186-187 and
// :242 at 6d004b6). The only Engine state they use is the chat tokens, which emit takes as its first parameter under
// the Engine member's name ct_ (core.shared, DESIGN v3 3.1). The bodies: engine_output.cpp. Includes are spelled from
// src/ (server/...), as in engine_drafter.h.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "server/engine_api.h"   // Job
#include "server/engine_slot.h"  // Slot
#include "whirl/chat_tokens.h"   // chat::ChatTokens

#include <cstdint>
#include <string>

namespace whirl::server {

// request lifecycle (were Engine members)
bool emit(const chat::ChatTokens& ct_, Slot& sl, std::uint32_t t);
void flushOut(Slot& sl);
void writeError(Job& job, int status, const std::string& msg);

}  // namespace whirl::server
