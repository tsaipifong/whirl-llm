# SPDX-License-Identifier: Apache-2.0
"""Self-test of tools/gates/engine_split_check.py (arch025-engine S0; review3 A5).

Every mutant names the report line (a regex) of the rule that must catch it; CAUGHT only when the checker fails AND
that line is printed (a failure for another reason is "CAUGHT BY ANOTHER RULE" = a self-test failure). Baselines
must pass. Mutants come from design v2 (M1-M5), review2 (N1-N8), design v3 (X1-X14), review3 (Y1-Y14), the alias
write of DESIGN 1.5 (AW1), and S0's own "fix the cut the obvious way" variants (*b: the deeper rule must still
catch it) and cut-pin mutants (P1-P3, review3 C2).

  python -I tools/gates/engine_split_selftest.py [--work W] [--base R ...] [--jobs N]
  --base: source trees with src/ (and include/); default = this repo. Mutants run on the FIRST base; every base
  runs as a baseline (B0, B1, ...). X13 (the leg-1 tree) runs on the second base when one is given.
Exit 0 = "engine split selftest: N/N as expected".
"""
import json, os, re, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PY = sys.executable
CUT = os.path.join(HERE, "engine_cut.json")
INL = os.path.join(HERE, "engine_split_inline_c1ca7ce.tsv")
PIN = os.path.join(REPO, "tests", "pins", "engine_cut_arch.sha256")
REVIEWS = os.path.join(REPO, "tests", "pins", "engine_cut_reviews.tsv")


def arg(name, default=None):
    a = sys.argv[1:]
    return a[a.index(name) + 1] if name in a else default


bases = [sys.argv[i + 1] for i, x in enumerate(sys.argv) if x == "--base"] or [REPO]
WORK = arg("--work", os.path.join(REPO, "build", "engine_split_selftest"))
JOBS = int(arg("--jobs", "8"))

EMIT = "bool Engine::emit(Slot& sl, std::uint32_t t) {\n"
GOVT = "void Engine::govTick(const wddmwatch::Sample& s) {\n"
GOVT_DECL = "    void govTick(const wddmwatch::Sample& s);   // after each sample: decide (and log the decisions)\n"
EMIT_DECL = "    bool emit(Slot& sl, std::uint32_t t);\n"
DROPC = "void Engine::dropCache(Slot& sl) {\n"
DROP_DECL = "    void dropCache(Slot& sl);\n"
RUNL = "void Engine::runLoop() {\n"
HOLD = "    bool gov_hold_ = false;"
EH, EC, EP, EG = "server/engine.h", "server/engine.cpp", "server/engine_pages.cpp", "server/engine_gov.cpp"
V = r"  <-- VIOLATION"


def viol(a, b, kind, why):
    return re.escape(f"{a} -> {b} [{kind} ") + r"\d+\]" + re.escape(V) + r".*\b" + re.escape(why) + r"\b"


def ow(fd):   # write to another module's data, not in outside_writers
    return r"^   not in the expected list '\w+': .*\b" + re.escape(fd) + r"\b"


def sw(fd):   # write to a core.shared member by an unlisted writer
    return r"^== writes to core\.shared not in shared_writers '\w+' \(\d+\): .*\b" + re.escape(fd) + r"\b"


def unas(e):
    return r"^unassigned entries: [^;]*(?<![\w:])" + re.escape(e) + r"(?![\w:])"


def fnd(t):
    return r"^   " + re.escape(t)


ENG = "server.engine"
SO, SE, SC, GV, SS, JQ = "server.slot_output", "server.slot_end", "server.slot_cache", "server.governor", "server.slot_start", "server.job_queue"
# (id, what, edits [(rel, old|None|"APPEND", new)], cut_edit(dict)->None or None, extra graph args, expected regex, base index)
MUT = [
    ("M1", "inline body in engine.h (kvTokBytes, slot_cache) reads governor data",
     [(EH, "        for (const KvArr& a : kv_arrays_) b += a.row;\n", "        for (const KvArr& a : kv_arrays_) b += a.row;\n        if (gov_hold_) b += 0;\n")],
     None, [], viol(SC, GV, "data", "kvTokBytes:gov_hold_")),
    ("M2", "constructor body writes governor data", [(EC, "    fl_.epoch = Clock::now();\n", "    fl_.epoch = Clock::now();\n    gov_pending_ = true;\n")],
     None, [], ow("Engine:gov_pending_")),
    ("M3", "a definition in a file not named engine_*", [("server/vram_watch.cpp", None, '#include "engine.h"\nnamespace whirl::server {\nvoid Engine::vramNote2() { gov_warn_.clear(); }\n}\n')],
     None, [], unas("vramNote2")),
    ("M4", "emit -> finishJob (slot_output calls up)", [(EC, EMIT, EMIT + "    if (t == 0xffffffffu) finishJob(sl);\n")], None, [], viol(SO, SE, "call", "emit->finishJob")),
    ("M5", "dropCache -> startJob", [(EP, DROPC, DROPC + "    if (sl.job && sl.pos == 0xffffffffu) (void)startJob(sl, *sl.job);\n")], None, [], viol(SC, SS, "call", "dropCache->startJob")),
    ("N1", "this->gov_hold_ read in emit", [(EC, EMIT, EMIT + "    if (this->gov_hold_) t = 0;\n")], None, [], viol(SO, GV, "data", "emit:gov_hold_")),
    ("N2", "this->finishJob(sl) in emit", [(EC, EMIT, EMIT + "    if (t == 1u) this->finishJob(sl);\n")], None, [], viol(SO, SE, "call", "emit->finishJob")),
    ("N3", "Engine::finishJob(sl) in emit", [(EC, EMIT, EMIT + "    if (t == 1u) Engine::finishJob(sl);\n")], None, [], viol(SO, SE, "call", "emit->finishJob")),
    ("N4", "emit writes spec_on_ (spec_cycle's)", [(EC, EMIT, EMIT + "    spec_on_ = false;\n")], None, [], ow("emit:spec_on_")),
    ("N5", "emit writes fl_ (prefill's)", [(EC, EMIT, EMIT + "    fl_.epoch = Clock::now();\n")], None, [], ow("emit:fl_")),
    ("N6", "macro alias of a member in an engine file", [(EC, EMIT, "#define WHX_HOLD gov_hold_\n" + EMIT + "    if (WHX_HOLD) t = 0;\n")], None, [], fnd("#define in an engine file server/engine.cpp")),
    ("N7", "emit becomes SlotOutput::emit and calls finishJob", [(EC, EMIT, "bool SlotOutput::emit(Slot& sl, std::uint32_t t) {\n    if (t == 1u) finishJob(sl);\n"), (EH, EMIT_DECL, "")],
     None, [], unas("SlotOutput::emit")),
    ("N7b", "N7 with the cut 'fixed': SlotOutput::emit added to slot_output (alias -> emit): the upward call still fails",
     [(EC, EMIT, "bool SlotOutput::emit(Slot& sl, std::uint32_t t) {\n    if (t == 1u) finishJob(sl);\n"), (EH, EMIT_DECL, "")],
     lambda c: c["aliases"].update({"SlotOutput::emit": "emit"}), ["--no-pin"], viol(SO, SE, "call", "emit->finishJob")),
    ("N8", "govTick becomes GovernorGlue::tick and calls startJob", [(EG, GOVT, "void GovernorGlue::tick(const wddmwatch::Sample& s) {\n    (void)startJob(*slots_.data(), *slots_[0].job);\n"), (EH, GOVT_DECL, "")],
     None, [], viol(GV, SS, "call", "govTick->startJob")),
    ("X1", "SlotPages::dropCache reads gst_.gov_hold_", [(EP, DROPC, "void SlotPages::dropCache(Slot& sl) {\n    if (gst_.gov_hold_) return;\n"), (EH, DROP_DECL, "")], None, [], viol(SC, GV, "data", "dropCache:gov_hold_")),
    ("X2", "GovernorGlue::tick calls ss_.startJob", [(EG, GOVT, "void GovernorGlue::tick(const wddmwatch::Sample& s) {\n    (void)ss_.startJob(*slots_.data(), *slots_[0].job);\n"), (EH, GOVT_DECL, "")], None, [], viol(GV, SS, "call", "govTick->startJob")),
    ("X3", "SlotPages::dropCache writes jq_.stop_", [(EP, DROPC, "void SlotPages::dropCache(Slot& sl) {\n    jq_.stop_ = true;\n"), (EH, DROP_DECL, "")], None, [], ow("dropCache:stop_")),
    ("X4", "body moves to an undeclared class Helper::emit", [(EC, EMIT, "bool Helper::emit(Slot& sl, std::uint32_t t) {\n"), (EH, EMIT_DECL, "")], None, [], unas("Helper::emit")),
    ("X5", "dropCache writes bst_ (core.shared)", [(EP, DROPC, DROPC + "    bst_ = BatchStats{};\n")], None, [], sw("dropCache:bst_")),
    ("X6", "new Engine inline function reads queue_", [(EH, "    ServerModel& m_;\n", "    std::size_t queued() { return queue_.size(); }\n    ServerModel& m_;\n")], None, [], unas("queued")),
    ("X7", "log.h #define naming a member", [("server/log.h", "APPEND", "\n#define WHX_STOP_FLAG stop_\n")], None, [], fnd("#define naming tracked names server/log.h")),
    ("X8", "control: dropCache reads gov_hold_", [(EP, DROPC, DROPC + "    if (gov_hold_) return;\n")], None, [], viol(SC, GV, "data", "dropCache:gov_hold_")),
    ("X9", "new unlisted slot_output.cpp includes engine.h, free function reads gst_.gov_hold_",
     [("server/slot_output.cpp", None, '#include "engine.h"\nnamespace whirl::server {\nbool flushOutFree(Slot& sl, const GovState& gst_) { return gst_.gov_hold_; }\n}\n')],
     None, [], fnd("file includes an engine header but is neither a module file nor a listed user: server/slot_output.cpp")),
    ("X9b", "X9 with the file listed in slot_output's files (as leg 2 will): the free function is unassigned",
     [("server/slot_output.cpp", None, '#include "engine.h"\nnamespace whirl::server {\nbool flushOutFree(Slot& sl, const GovState& gst_) { return gst_.gov_hold_; }\n}\n')],
     lambda c: c["modules"][SO].setdefault("files", []).append("server/slot_output.cpp"), ["--no-pin"], unas("flushOutFree")),
    ("X9c", "X9b with the function also added to slot_output: the upward data use still fails",
     [("server/slot_output.cpp", None, '#include "engine.h"\nnamespace whirl::server {\nbool flushOutFree(Slot& sl, const GovState& gst_) { return gst_.gov_hold_; }\n}\n')],
     lambda c: (c["modules"][SO].setdefault("files", []).append("server/slot_output.cpp"), c["modules"][SO]["funcs"].append("flushOutFree")),
     ["--no-pin"], viol(SO, GV, "data", "flushOutFree:gov_hold_")),
    ("X10", "emit reads vis_ (not in slot_output's refs)", [(EC, EMIT, EMIT + "    if (vis_.encoded() == 7u) t = 0;\n")], None, [], r"^== reference lists \(R4\): uses outside the list: .*server\.slot_output uses vis_"),
    ("X11", "cut puts stats in two core categories", [], lambda c: c["core_categories"]["accessor"].append("stats"), ["--no-pin"], r"^== core categories overlap \(R6\): stats in "),
    ("X12", "--core on today's tree (core still schedules) must fail", [], None, ["--core"], r"^== --core \(leg-end shape of server\.engine, R5 / R6\): [1-9]\d* problems"),
    ("X13", "leg-1 tree (second base): emit -> finishJob", [(EC, EMIT, EMIT + "    if (t == 0xffffffffu) finishJob(sl);\n")], None, [], viol(SO, SE, "call", "emit->finishJob"), 1),
    ("X14", "friend in class Engine", [(EH, "    ServerModel& m_;\n", "    friend struct Peek;\n    ServerModel& m_;\n")], None, [], fnd("friend in tracked class Engine")),
    # --- review3 (Y1-Y14) ---
    ("Y1", "runLoop gov_json_.append (mutator method on governor data)", [(EC, RUNL, RUNL + '    gov_json_.append("x");\n')], None, [], ow("runLoop:gov_json_")),
    ("Y1c", "control: runLoop gov_json_ += x", [(EC, RUNL, RUNL + '    gov_json_ += "x";\n')], None, [], ow("runLoop:gov_json_")),
    ("Y2", "dropCache ps_.spes().clear() (chained mutator on core.shared)", [(EP, DROPC, DROPC + "    ps_.spes().clear();\n")], None, [], sw("dropCache:ps_")),
    ("Y3", "dropCache binds auto& q = stop_ and writes it", [(EP, DROPC, DROPC + "    { auto& q = stop_; q = true; }\n")], None, [], ow("dropCache:stop_")),
    ("Y4", "Engine& me = *this; me.finishJob", [(EP, DROPC, DROPC + "    { Engine& me = *this; if (sl.pos == 0xffffffffu) me.finishJob(sl); }\n")], None, [], viol(SC, SE, "call", "dropCache->finishJob")),
    ("Y5", "nested struct Peek with inline body reads gov_hold_; dropCache calls it",
     [(EH, HOLD, HOLD + "\n    struct Peek { static bool h(const Engine& e) { return e.gov_hold_; } };"), (EP, DROPC, DROPC + "    if (Peek::h(*this)) return;\n")],
     None, [], unas("type:Peek")),
    ("Y5b", "Y5 with type:Peek assigned to slot_cache: its read of governor data still fails",
     [(EH, HOLD, HOLD + "\n    struct Peek { static bool h(const Engine& e) { return e.gov_hold_; } };"), (EP, DROPC, DROPC + "    if (Peek::h(*this)) return;\n")],
     lambda c: c["modules"][SC]["funcs"].append("type:Peek"), ["--no-pin"], viol(SC, GV, "data", "type:Peek:gov_hold_")),
    ("Y5u", "a type used by dropCache but assigned nowhere (struct Peek declared, not defined in the cut)",
     [(EH, HOLD, HOLD + "\n    struct Peek;"), (EP, DROPC, DROPC + "    Peek* pk = nullptr; (void)pk;\n")], None, [], r"^S0: used but unassigned types: .*type:Peek"),
    ("Y6", "member template forPages (slot_cache) reads governor data",
     [(EH, HOLD, HOLD + "\n    template <class F> void forPages(F f);"), (EP, DROPC, "template <class F> void Engine::forPages(F f) {\n    if (gov_hold_) f();\n}\n" + DROPC)],
     lambda c: c["modules"][SC]["funcs"].append("forPages"), ["--no-pin"], viol(SC, GV, "data", "forPages:gov_hold_")),
    ("Y7", "lambda [this] in dropCache calls finishJob", [(EP, DROPC, DROPC + "    auto k = [this](Slot& s) { finishJob(s); };\n    if (sl.pos == 0xffffffffu) k(sl);\n")], None, [], viol(SC, SE, "call", "dropCache->finishJob")),
    ("Y8", "member function pointer &Engine::finishJob in dropCache", [(EP, DROPC, DROPC + "    auto fp = &Engine::finishJob;\n    if (sl.pos == 0xffffffffu) (this->*fp)(sl);\n")], None, [], viol(SC, SE, "call", "dropCache->finishJob")),
    ("Y9", "Engine* up_ (cut gives up_ to slot_cache); dropCache -> up_->finishJob",
     [(EH, HOLD, HOLD + "\n    Engine* up_ = nullptr;"), (EP, DROPC, DROPC + "    if (up_ && sl.pos == 0xffffffffu) up_->finishJob(sl);\n")],
     lambda c: c["modules"][SC]["data"].append("up_"), ["--no-pin"], fnd("member of tracked class Engine holds Engine but is not a cut holder of it")),
    ("Y9b", "Y9 with up_ also listed as a holder of Engine: still a finding",
     [(EH, HOLD, HOLD + "\n    Engine* up_ = nullptr;"), (EP, DROPC, DROPC + "    if (up_ && sl.pos == 0xffffffffu) up_->finishJob(sl);\n")],
     lambda c: (c["modules"][SC]["data"].append("up_"), c["holders"].update({"up_": "Engine"})), ["--no-pin"], fnd("cut holder of Engine: up_")),
    ("Y10", "runLoop std::swap(gov_warn_, gov_last_)", [(EC, RUNL, RUNL + "    std::swap(gov_warn_, gov_last_);\n")], None, [], ow("runLoop:gov_warn_")),
    ("Y11", "runLoop std::fill(gov_json_.begin(), ...)", [(EC, RUNL, RUNL + "    std::fill(gov_json_.begin(), gov_json_.end(), ' ');\n")], None, [], ow("runLoop:gov_json_")),
    ("Y12", "runLoop queue_.front()->id = 0", [(EC, RUNL, RUNL + "    if (!queue_.empty()) queue_.front()->id = 0;\n")], None, [], ow("runLoop:queue_")),
    ("Y13", "dropCache poke(gov_pending_) through a new header's bool& parameter",
     [("server/slot_util.h", None, "#pragma once\nnamespace whirl::server {\ninline void poke(bool& b) { b = true; }\n}\n"), (EP, DROPC, DROPC + "    poke(gov_pending_);\n")], None, [], ow("dropCache:gov_pending_")),
    ("Y14", "runLoop dropAll(queue_) defined in a new .cpp", [("server/sched_util.cpp", None, "#include <deque>\nnamespace whirl::server {\nstruct Job;\nvoid dropAll(std::deque<Job*>& q) { q.clear(); }\n}\n"), (EC, RUNL, RUNL + "    dropAll(queue_);\n")],
     None, [], ow("runLoop:queue_")),
    # --- DESIGN 1.5 alias write ---
    ("AW1", "alias write (DESIGN 1.5): dropCache auto& q = queue_; q.push_back(nullptr)", [(EP, DROPC, DROPC + "    { auto& q = queue_; q.push_back(nullptr); }\n")], None, [], ow("dropCache:queue_")),
    ("AW2", "alias pointer write: dropCache auto* p = &n_active_; *p = 0", [(EP, DROPC, DROPC + "    { auto* p = &n_active_; *p = 0; }\n")], None, [], ow("dropCache:n_active_")),
    # --- review3 C2: the cut pin ---
    ("P1", "cut edit: slot_output moved up to layer 9 (would allow emit -> finishJob)", [], lambda c: c["modules"][SO].update({"layer": 9}), [], r"^engine cut pin: MISMATCH section modules\.server\.slot_output"),
    ("P2", "cut edit: outside_writers.now widened", [], lambda c: c["outside_writers"]["now"].append("runLoop:gov_json_"), [], r"^engine cut pin: MISMATCH section outside_writers"),
    ("P3", "cut edit: a holder added", [], lambda c: c["holders"].update({"zz_": "SlotPages"}), [], r"^engine cut pin: MISMATCH section holders"),
    # --- S0 review (progress/arch025-engine.S0.REVIEW.md; reviewer mutants scratch/a25_engine/s0review/s0r_mut.py), S0b ---
    # act-on 2: calls through receivers the census does not resolve must not be dropped
    ("U1", "(*this).finishJob(sl) in emit", [(EC, EMIT, EMIT + "    if (t == 1u) (*this).finishJob(sl);\n")], None, [], viol(SO, SE, "call", "emit->finishJob")),
    ("U2", "two-level auto alias: auto& me = *this; auto& me2 = me; me2.stats() in emit",
     [(EC, EMIT, EMIT + "    { auto& me = *this; auto& me2 = me; (void)me2.stats(); }\n")], None, [], viol(SO, ENG, "call", "emit->stats")),
    ("U3", "control: auto& me = *this; me.stats() in emit", [(EC, EMIT, EMIT + "    { auto& me = *this; (void)me.stats(); }\n")], None, [], viol(SO, ENG, "call", "emit->stats")),
    ("U4", "a new unresolved call (unknown receiver, name also an untracked method) is a finding, not only unresolved.txt",
     [(EC, EMIT, EMIT + "    if (t == 7u && sl.job) (void)sl.job->stats();\n")], None, [], fnd("unresolved call not in the cut's unresolved_ok: emit: ")),
    # act-on 1: write shapes the census missed (W1-W10), the accessor returning a reference (W11)
    ("W1", "auto&& q = stop_; q = true in dropCache", [(EP, DROPC, DROPC + "    { auto&& q = stop_; q = true; }\n")], None, [], ow("dropCache:stop_")),
    ("W2", "for (auto&& j : queue_) j = nullptr in dropCache", [(EP, DROPC, DROPC + "    for (auto&& j : queue_) j = nullptr;\n")], None, [], ow("dropCache:queue_")),
    ("W3", "structured binding auto& [a, b] = fl_; a = {} in dropCache", [(EP, DROPC, DROPC + "    { auto& [fa, fb] = fl_; fa = {}; (void)fb; }\n")], None, [], ow("dropCache:fl_")),
    ("W3b", "W3 from a higher layer (runLoop, may read prefill data): only the write rule can catch it",
     [(EC, RUNL, RUNL + "    { auto& [fa, fb] = fl_; fa = {}; (void)fb; }\n")], None, [], ow("runLoop:fl_")),
    ("W4", "parenthesized lvalue (stop_) = true in dropCache", [(EP, DROPC, DROPC + "    (stop_) = true;\n")], None, [], ow("dropCache:stop_")),
    ("W5", "static_cast<bool&>(stop_) = true in dropCache", [(EP, DROPC, DROPC + "    static_cast<bool&>(stop_) = true;\n")], None, [], ow("dropCache:stop_")),
    ("W6", "auto it = queue_.begin(); *it = nullptr in dropCache", [(EP, DROPC, DROPC + "    if (!queue_.empty()) { auto it = queue_.begin(); *it = nullptr; }\n")], None, [], ow("dropCache:queue_")),
    ("W7", "auto* p = gov_json_.data(); p[0] = 'x' in runLoop", [(EC, RUNL, RUNL + "    if (!gov_json_.empty()) { auto* p = gov_json_.data(); p[0] = 'x'; }\n")], None, [], ow("runLoop:gov_json_")),
    ("W8", "auto s = std::span(gov_json_); s[0] = 'x' in runLoop", [(EC, RUNL, RUNL + "    if (!gov_json_.empty()) { auto s = std::span(gov_json_); s[0] = 'x'; }\n")], None, [], ow("runLoop:gov_json_")),
    ("W9", "data member pointer: auto pm = &Engine::stop_; this->*pm = true in dropCache",
     [(EP, DROPC, DROPC + "    { auto pm = &Engine::stop_; this->*pm = true; }\n")], None, [], ow("dropCache:stop_")),
    ("W10", "(*queue_.begin()) = nullptr in dropCache", [(EP, DROPC, DROPC + "    if (!queue_.empty()) (*queue_.begin()) = nullptr;\n")], None, [], ow("dropCache:queue_")),
    ("W11", "bool& holdRef() { return gov_hold_; } in Engine (new_funcs -> governor, pin unchanged); runLoop holdRef() = true",
     [(EH, HOLD, HOLD + "\n    bool& holdRef() { return gov_hold_; }"), (EC, RUNL, RUNL + "    holdRef() = true;\n")],
     lambda c: c["new_funcs"].update({"holdRef": GV}), [], ow("runLoop:gov_hold_")),
    ("W11b", "W11 with the pin skipped (as after a reviewed cut change): the write through the accessor still fails",
     [(EH, HOLD, HOLD + "\n    bool& holdRef() { return gov_hold_; }"), (EC, RUNL, RUNL + "    holdRef() = true;\n")],
     lambda c: c["new_funcs"].update({"holdRef": GV}), ["--no-pin"], ow("runLoop:gov_hold_")),
    ("W11c", "the accessor's result bound to auto& and written: auto& h = holdRef(); h = true in runLoop",
     [(EH, HOLD, HOLD + "\n    bool& holdRef() { return gov_hold_; }"), (EC, RUNL, RUNL + "    { auto& h = holdRef(); h = true; }\n")],
     lambda c: c["new_funcs"].update({"holdRef": GV}), ["--no-pin"], ow("runLoop:gov_hold_")),
    # act-on 3: the cut pin covers new_funcs / new_data (rows may only be deleted), the tools, and needs a review row
    ("P4", "planned function govActive moved from governor to slot_output via new_funcs", [],
     lambda c: c["new_funcs"].update({"govActive": SO}), [], r"^engine cut pin: MISMATCH new_funcs govActive"),
    ("P4b", "a new planned function added to new_funcs", [], lambda c: c["new_funcs"].update({"zzPlanned": SO}), [], r"^engine cut pin: MISMATCH new_funcs zzPlanned"),
    ("P4c", "a new planned member added to new_data", [], lambda c: c["new_data"].update({"zz_": SO}), [], r"^engine cut pin: MISMATCH new_data zz_"),
    ("P5", "new function peekHold() (slot_output via new_funcs) reads governor data; emit calls it",
     [(EH, HOLD, HOLD + "\n    bool peekHold() const;"), (EC, EMIT, "bool Engine::peekHold() const { return gov_hold_; }\n" + EMIT + "    if (peekHold()) t = 0;\n")],
     lambda c: c["new_funcs"].update({"peekHold": SO}), ["--no-pin"], viol(SO, GV, "data", "peekHold:gov_hold_")),
    ("P6", "a checker tool edited (census copy): the tools hash in the pin differs",
     [("TOOL:engine_split_census.py", "APPEND", "\n# a harmless-looking edit\n")], None, [], r"^engine cut pin: MISMATCH tools engine_split_census\.py"),
    ("P7", "the pin rewritten with --pin-write for a cut edit (same commit), no review row: UNREVIEWED",
     [], lambda c: c["holders"].update({"zz_": "SlotPages"}), ["@PINWRITE"], r"^engine cut pin: UNREVIEWED"),
]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)


def one(item):
    mid, what, edits, cut_edit, gargs, expect = item[:6]
    bi = item[6] if len(item) > 6 else 0
    if bi >= len(bases):
        return mid, "SKIP", f"{mid} {what}\n   needs --base #{bi + 1} -> SKIPPED (not counted)"
    base = bases[bi]
    d = os.path.join(WORK, mid)
    if os.path.isdir(d): shutil.rmtree(d)
    os.makedirs(d)
    for sub in ("src", "include"):
        if os.path.isdir(os.path.join(base, sub)):
            shutil.copytree(os.path.join(base, sub), os.path.join(d, sub))
    tools = HERE
    if any(rel.startswith("TOOL:") for rel, _, _ in edits):   # S0b: a copy of the checker tools, one of them edited
        tools = os.path.join(d, "tools", "gates")
        os.makedirs(tools)
        for f in os.listdir(HERE):
            if f.startswith(("engine_split_", "engine_cut", "member_order")):
                shutil.copy2(os.path.join(HERE, f), tools)
    for rel, old, new in edits:
        if rel.startswith("TOOL:"):
            open(os.path.join(tools, rel[5:]), "a", encoding="utf-8").write(new); continue
        p = os.path.join(d, "src", rel)
        if old is None:
            open(p, "w", encoding="utf-8").write(new); continue
        s = open(p, encoding="utf-8").read()
        if old == "APPEND":
            open(p, "w", encoding="utf-8").write(s + new); continue
        if s.count(old) != 1:
            return mid, "ANCHOR", f"{mid} {what}\n   anchor missing / not unique in {rel} -> ANCHOR FAIL"
        open(p, "w", encoding="utf-8").write(s.replace(old, new))
    cutp = CUT
    if cut_edit:
        cj = json.load(open(CUT, encoding="utf-8")); cut_edit(cj)
        cutp = os.path.join(d, "engine_cut.json"); json.dump(cj, open(cutp, "w", encoding="utf-8"), indent=1)
    pin = PIN
    if "@PINWRITE" in gargs:   # S0b: the step rewrites the pin in the same commit (no review row added)
        gargs = [g for g in gargs if g != "@PINWRITE"]
        pin = os.path.join(d, "engine_cut_arch.sha256")
        run([PY, "-I", os.path.join(tools, "engine_split_check.py"), "--pin-write", "--cut", cutp, "--pin", pin])
    cmd = [PY, "-I", os.path.join(tools, "engine_split_check.py"), "--root", d, "--cut", cutp, "--work", os.path.join(d, "w"),
           "--pin", pin, "--reviews", REVIEWS, "--inline", INL] + gargs
    rc, out = run(cmd)
    rep = ""
    for f in ("graph_engine_cut.txt",):
        p = os.path.join(d, "w", "census", f)
        if os.path.exists(p): rep += open(p, encoding="utf-8").read()
    fp = os.path.join(d, "w", "census", "findings.txt")
    allt = out + "\n" + rep
    if os.path.exists(fp):
        allt += "\n" + "".join("   " + l for l in open(fp, encoding="utf-8"))
    hit = re.search(expect, allt, re.M) is not None
    if rc == 3: v = "CRASH"
    elif rc == 1 and hit: v = "CAUGHT"
    elif rc == 1: v = "CAUGHT BY ANOTHER RULE"
    else: v = "NOT CAUGHT"
    m = re.search(expect, allt, re.M)
    shown = m.group(0)[:160] if m else "(expected line not printed: /" + expect[:120] + "/)"
    return mid, v, f"{mid} {what}\n   rc={rc}; expected line: {shown} -> {v}"


def main():
    os.makedirs(WORK, exist_ok=True)
    res = []
    for i, b in enumerate(bases):   # baselines
        rc, out = run([PY, "-I", os.path.join(HERE, "engine_split_check.py"), "--root", b, "--work", os.path.join(WORK, f"B{i}"), "--pin", PIN])
        ok = rc == 0 and "engine split check: ok" in out
        res.append((f"B{i}", "PASS" if ok else "FAIL", f"B{i} baseline {b}\n   rc={rc}: {out.strip().splitlines()[-1] if out.strip() else ''} -> {'PASS' if ok else 'FAIL (false positive)'}"))
    with ThreadPoolExecutor(JOBS) as ex:
        res += list(ex.map(one, MUT))
    good = sum(1 for _, v, _ in res if v in ("PASS", "CAUGHT"))
    counted = sum(1 for _, v, _ in res if v != "SKIP")
    txt = "\n".join(r[2] for r in res)
    txt += f"\nengine split selftest: {good}/{counted} as expected" + ("" if good == counted else " - FAIL")
    open(os.path.join(WORK, "engine_split_selftest.txt"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)
    return 0 if good == counted else 1


if __name__ == "__main__":
    sys.exit(main())
