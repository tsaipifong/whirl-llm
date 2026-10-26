// Baseline dumps (arch025-S0a): what a load decided, printed only when asked, so the refactoring steps of 0.2.5 can
// show they decide the same (arch025.PLAN.md S0a / S0b-2). With the variables unset nothing is printed and nothing
// changes (I4: no parsed log line changes); the names are in the SSD fingerprint's skip list (I6).
//   WHIRL_DUMP_KV=1    every numerics::chooseKv call of a load or a server sizing estimate: one "D kv:" line to stderr
//                      with the call site, its inputs and its result (site loader.uma_est / loader.load /
//                      server.est_dgpu (server_main ~:801) / server.est_uma (~:911))
//   WHIRL_DUMP_EXEC=1  after load + tune (qwen35::loadOrTune): the Model's numerics / execution flags, the plan and the
//                      KV choice, one "D exec: key=value" line each, to stderr
//   --dump-only        (whirl CLI commands and whirl-server) exit 0 right after load + tune, after the dumps; for the
//                      S0b-2 dump matrix (one load per cell, no generation)
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "whirl/numerics.h"

namespace whirl::dump {

// WHIRL_<name> set, not empty and not "0"; read on every call through platform::env::get (a dump is not on a hot
// path). Was envOn (arch025 PL1b renamed it: the env read is the platform layer's, this is a dump switch)
bool switchOn(const char* name);

// "D kv: site=<site> mode=<m> items=<hex> custom=<0|1> moe=<0|1> caps=q8h:<b>,q8v:<b>,q4:<b>,fast_q8h:<b>
//  forced=<name|none> -> kv=<name> debug=<0|1> why=<why>"
std::string kvLine(std::string_view site, const numerics::Request& r, bool moe, const numerics::KvCaps& caps,
                   std::optional<KvFormat> forced, const numerics::KvChoice& c);
// prints kvLine to stderr when WHIRL_DUMP_KV is on; returns c (so a call site can wrap its chooseKv call)
const numerics::KvChoice& kv(std::string_view site, const numerics::Request& r, bool moe, const numerics::KvCaps& caps,
                             std::optional<KvFormat> forced, const numerics::KvChoice& c);

// --dump-only (set by the CLI / server argument parsers)
void setDumpOnly(bool on);
bool dumpOnly();

}  // namespace whirl::dump

namespace whirl::qwen35 {
class Model;
// WHIRL_DUMP_EXEC: the "D exec:" lines (src/model/exec_dump.cpp)
std::string execDump(const Model& m);
// end of loadOrTune: the exec dump when WHIRL_DUMP_EXEC is on; exit(0) when --dump-only
void afterLoadTune(const Model& m);
}  // namespace whirl::qwen35
