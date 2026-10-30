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
import hashlib, json, os, re, shutil, subprocess, sys
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
LEDGER = os.path.join(WORK, "ledger", "engine_cut_reviews.tsv")   # S0c: synthetic ledger (setup_ledger)
# S0d: every mutant's census shares one AST cache (keyed by the content of each TU's files; a mutant re-parses only the
# TUs whose files it edits)
os.environ.setdefault("WHIRL_ENGINE_AST_CACHE", os.path.join(REPO, "build", "engine_split_ast_cache"))
BASESNAP = os.path.join(WORK, "ledger", "base")                     # S0c: base snapshot for --base-dir

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


def _dc(code, m="stop_"):   # S0d (S0c review runner shapes): a statement in dropCache that must be a write of m
    return [(EP, DROPC, DROPC + "    " + code + "\n")], None, [], ow("dropCache:" + m)


def _rl(code, m="gov_json_"):
    return [(EC, RUNL, RUNL + "    " + code + "\n")], None, [], ow("runLoop:" + m)


def _acc(decl, use, mod="server.governor", name="holdRef", m="gov_hold_"):
    return ([(EH, HOLD, HOLD + "\n    " + decl), (EC, RUNL, RUNL + "    " + use + "\n")],
            lambda c: c["new_funcs"].update({name: mod}), ["--no-pin"], ow("runLoop:" + m))


_T1_WHY = "writable handle begin() used (, gov_json_.)"   # S0e T1b: the text rule's reason for the T1 row


def _okc(code):   # a read control in runLoop: the check must pass
    return [(EC, RUNL, RUNL + "    " + code + "\n")], None, [], r"(?#OK)^engine split check: ok"


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
    # --- S0b review (progress/arch025-engine.S0b.REVIEW.md; reviewer runner scratch/a25_engine/s0b_review/runner/r_mut.py) ---
    # act-on 1: write shapes the S0b census still missed (N1-N8, N10, N12, N13, N15); N9 / N14 / C1-C5 are controls
    ("N1", "deref binding: auto& j = *queue_.begin(); j = nullptr (dropCache)", [(EP, DROPC, DROPC + "    if (!queue_.empty()) { auto& j = *queue_.begin(); j = nullptr; }\n")], None, [], ow("dropCache:queue_")),
    ("N2", "brace-init reference: auto& q{stop_}; q = true (dropCache)", [(EP, DROPC, DROPC + "    { auto& q{stop_}; q = true; }\n")], None, [], ow("dropCache:stop_")),
    ("N3", "C-style reference cast: ((bool&)stop_) = true (dropCache)", [(EP, DROPC, DROPC + "    ((bool&)stop_) = true;\n")], None, [], ow("dropCache:stop_")),
    ("N4", "ternary binding: auto& q = sl.job ? stop_ : stop_; q = true (dropCache)", [(EP, DROPC, DROPC + "    { auto& q = sl.job ? stop_ : stop_; q = true; }\n")], None, [], ow("dropCache:stop_")),
    ("N5", "deref as call argument: std::swap(*queue_.begin(), *queue_.rbegin()) (dropCache)", [(EP, DROPC, DROPC + "    if (queue_.size() > 1) std::swap(*queue_.begin(), *queue_.rbegin());\n")], None, [], ow("dropCache:queue_")),
    ("N6", "pointer arithmetic deref: *(gov_json_.data() + 1) = 'x' (runLoop)", [(EC, RUNL, RUNL + "    if (gov_json_.size() > 1) *(gov_json_.data() + 1) = 'x';\n")], None, [], ow("runLoop:gov_json_")),
    ("N7", "handle + offset as argument: std::memset(gov_json_.data() + 1, 0, 1) (runLoop)", [(EC, RUNL, RUNL + "    if (gov_json_.size() > 1) std::memset(gov_json_.data() + 1, 0, 1);\n")], None, [], ow("runLoop:gov_json_")),
    ("N8", "fill_n(gov_json_.begin() + 1, 1, 'x') (runLoop)", [(EC, RUNL, RUNL + "    if (gov_json_.size() > 1) std::fill_n(gov_json_.begin() + 1, 1, 'x');\n")], None, [], ow("runLoop:gov_json_")),
    ("N9", "control: C++20 range-for init-statement with const: for (const auto n = queue_.size(); auto& j : queue_) j = nullptr", [(EP, DROPC, DROPC + "    for (const auto n = queue_.size(); auto& j : queue_) { (void)n; j = nullptr; }\n")], None, [], ow("dropCache:queue_")),
    ("N10", "trailing return type accessor: auto holdRef() -> bool& { return gov_hold_; }; runLoop holdRef() = true",
     [(EH, HOLD, HOLD + "\n    auto holdRef() -> bool& { return gov_hold_; }"), (EC, RUNL, RUNL + "    holdRef() = true;\n")], lambda c: c["new_funcs"].update({"holdRef": GV}), ["--no-pin"], ow("runLoop:gov_hold_")),
    ("N11", "decltype(auto) holdRef() { return (gov_hold_); }; runLoop holdRef() = true (the write rule, not 'unassigned decltype')",
     [(EH, HOLD, HOLD + "\n    decltype(auto) holdRef() { return (gov_hold_); }"), (EC, RUNL, RUNL + "    holdRef() = true;\n")], lambda c: c["new_funcs"].update({"holdRef": GV}), ["--no-pin"], ow("runLoop:gov_hold_")),
    ("N12", "accessor through a local reference: bool& holdRef() { auto& r = gov_hold_; return r; }; runLoop holdRef() = true",
     [(EH, HOLD, HOLD + "\n    bool& holdRef() { auto& r = gov_hold_; return r; }"), (EC, RUNL, RUNL + "    holdRef() = true;\n")], lambda c: c["new_funcs"].update({"holdRef": GV}), ["--no-pin"], ow("runLoop:gov_hold_")),
    ("N13", "accessor returning a writable view: std::span<char> jsonView() { return gov_json_; }; runLoop jsonView()[0] = 'x'",
     [(EH, HOLD, HOLD + "\n    std::span<char> jsonView() { return gov_json_; }"), (EC, RUNL, RUNL + "    if (!jsonView().empty()) jsonView()[0] = 'x';\n")], lambda c: c["new_funcs"].update({"jsonView": GV}), ["--no-pin"], ow("runLoop:gov_json_")),
    ("N14", "control: accessor whose return type has parentheses: std::function<wddmwatch::Sample()>& probeRef(); runLoop probeRef() = nullptr (vram_probe_ is core.shared)",
     [(EH, HOLD, HOLD + "\n    std::function<wddmwatch::Sample()>& probeRef() { return vram_probe_; }"), (EC, RUNL, RUNL + "    probeRef() = nullptr;\n")], lambda c: c["new_funcs"].update({"probeRef": "server.vram_watch"}), ["--no-pin"], sw("runLoop:vram_probe_")),
    ("N15", "accessor returning an iterator: auto qBegin() { return queue_.begin(); }; runLoop *qBegin() = nullptr",
     [(EH, HOLD, HOLD + "\n    auto qBegin() { return queue_.begin(); }"), (EC, RUNL, RUNL + "    if (!queue_.empty()) *qBegin() = nullptr;\n")], lambda c: c["new_funcs"].update({"qBegin": GV}), ["--no-pin"], ow("runLoop:queue_")),
    ("C1", "control: init capture by reference [&q = stop_]{ q = true; }() (dropCache)", [(EP, DROPC, DROPC + "    [&q = stop_] { q = true; }();\n")], None, [], ow("dropCache:stop_")),
    ("C2", "control: std::exchange(stop_, true) (dropCache)", [(EP, DROPC, DROPC + "    (void)std::exchange(stop_, true);\n")], None, [], ow("dropCache:stop_")),
    ("C3", "control: *queue_.begin() = nullptr (dropCache)", [(EP, DROPC, DROPC + "    if (!queue_.empty()) *queue_.begin() = nullptr;\n")], None, [], ow("dropCache:queue_")),
    ("C4", "control: auto p = this; p->finishJob(sl) in emit (unknown pointer alias)", [(EC, EMIT, EMIT + "    if (t == 1u) { auto p = this; p->finishJob(sl); }\n")], None, [], viol(SO, SE, "call", "emit->finishJob")),
    ("C5", "control: std::invoke(&Engine::finishJob, this, sl) in emit", [(EC, EMIT, EMIT + "    if (t == 1u) std::invoke(&Engine::finishJob, this, sl);\n")], None, [], viol(SO, SE, "call", "emit->finishJob")),
    # S0c's own probes of the inverted rule (shapes no earlier list enumerated) and read controls that must stay reads
    ("Z1", "std::ref(stop_).get() = true (dropCache)", [(EP, DROPC, DROPC + "    std::ref(stop_).get() = true;\n")], None, [], ow("dropCache:stop_")),
    ("Z2", "bool* p = std::addressof(stop_); *p = true (dropCache)", [(EP, DROPC, DROPC + "    { bool* p = std::addressof(stop_); *p = true; }\n")], None, [], ow("dropCache:stop_")),
    ("Z3", "for (auto it = queue_.rbegin(); it != queue_.rend(); ++it) *it = nullptr (dropCache)", [(EP, DROPC, DROPC + "    for (auto it = queue_.rbegin(); it != queue_.rend(); ++it) *it = nullptr;\n")], None, [], ow("dropCache:queue_")),
    ("Z4", "if (auto& q = stop_; !q) q = true (if-init binding, dropCache)", [(EP, DROPC, DROPC + "    if (auto& q = stop_; !q) q = true;\n")], None, [], ow("dropCache:stop_")),
    ("Z5", "std::tie(stop_) = std::make_tuple(true) (dropCache)", [(EP, DROPC, DROPC + "    std::tie(stop_) = std::make_tuple(true);\n")], None, [], ow("dropCache:stop_")),
    ("Z6", "control: reads in runLoop (compare a handle, size arithmetic, ==, !, subscript read) stay reads",
     [(EC, RUNL, RUNL + "    if (!queue_.empty() && queue_.front() != nullptr && gov_json_.size() + 1 > 2 && gov_json_[0] == 'x' && !stop_) {}\n")],
     None, [], r"(?#OK)^engine split check: ok"),
    # act-on 2: the ledger against the base of the range (base = the selftest's snapshot of today's cut / pin / ledger)
    ("P8", "reviewer's bypass (s0b_review/bypass): cut edit + --pin-write + ledger row '<sha> x x' in the same change",
     [], lambda c: c["outside_writers"]["now"].append("runLoop:gov_json_"), ["@PINWRITE", "@ROW:xx"], r"^engine cut pin: REVIEW ROW INVALID .*needs 4 non-empty columns"),
    ("P8b", "a row whose review report does not name the pin sha256", [], lambda c: c["outside_writers"]["now"].append("runLoop:gov_json_"),
     ["@PINWRITE", "@ROW:nosha"], r"^engine cut pin: REVIEW ROW INVALID .*does not name the pin sha256"),
    ("P8c", "a row whose report's reviewer is the author (self-review)", [], lambda c: c["outside_writers"]["now"].append("runLoop:gov_json_"),
     ["@PINWRITE", "@ROW:self"], r"^engine cut pin: REVIEW ROW INVALID .*is the author \(no self-review\)"),
    ("P8d", "a row whose review column says pending (the S0b row shape)", [], lambda c: c["outside_writers"]["now"].append("runLoop:gov_json_"),
     ["@PINWRITE", "@ROW:pending"], r"^engine cut pin: REVIEW ROW INVALID .*says pending"),
    ("P8e", "control: the same cut edit with a valid row (report names the sha, reviewer != author) passes", [],
     lambda c: c["outside_writers"]["now"].append("runLoop:gov_json_"), ["@PINWRITE", "@ROW:valid"], r"(?#OK)^engine split check: ok"),
    ("P9", "a new planned row with --pin-write and a valid row: not in the base cut", [], lambda c: c["new_funcs"].update({"zzPlanned": SO}),
     ["@PINWRITE", "@ROW:valid"], r"^engine cut pin: NEW PLANNED ROW new_funcs zzPlanned"),
    ("P9b", "a planned row moved to another module with --pin-write and a valid row", [], lambda c: c["new_funcs"].update({"govActive": SO}),
     ["@PINWRITE", "@ROW:valid"], r"^engine cut pin: PLANNED ROW MOVED new_funcs govActive"),
    ("P10", "an old ledger row edited (its review column rewritten)", [], None, ["@LEDGEREDIT"], r"^engine cut pin: LEDGER EDITED"),
    ("P11", "e8_judge.py edited (S0c: in the hashed tool list)", [("TOOL:e8_judge.py", "APPEND", "\n# a harmless-looking edit\n")], None, [], r"^engine cut pin: MISMATCH tools e8_judge\.py"),
    ("P11b", "revtext.py edited (S0c: in the hashed tool list)", [("TOOL:revtext.py", "APPEND", "\n# a harmless-looking edit\n")], None, [], r"^engine cut pin: MISMATCH tools revtext\.py"),
    # --- S0c review (progress/arch025-engine.S0c.REVIEW.md act-on 1, consider 2; reviewer runner
    # scratch/a25_engine/s0c_review/runner/r_mut.py): write shapes only declared types decide (S0d: the AST census) ---
    ("R1", "alias-typedef reference: using BR = bool&; BR q = stop_; q = true", *_dc("{ using BR = bool&; BR q = stop_; q = true; }")),
    ("R2", "std::reference_wrapper<bool> q{stop_}; q.get() = true", *_dc("{ std::reference_wrapper<bool> q{stop_}; q.get() = true; }")),
    ("R3", "std::reference_wrapper<bool> q = stop_; q.get() = true", *_dc("{ std::reference_wrapper<bool> q = stop_; q.get() = true; }")),
    ("R4", "conditional lvalue: (sl.job ? stop_ : stop_) = true", *_dc("(sl.job ? stop_ : stop_) = true;")),
    ("R5", "comma lvalue: (void(0), stop_) = true", *_dc("(void(0), stop_) = true;")),
    ("R6", "immediately-invoked lambda: [](bool& b) { b = true; }(stop_)", *_dc("[](bool& b) { b = true; }(stop_);")),
    ("R7", "parenthesized callee: auto set = [](bool& b) { b = true; }; (set)(stop_)", *_dc("{ auto set = [](bool& b) { b = true; }; (set)(stop_); }")),
    ("R8", "local lambda returning a reference: auto r = [this]() -> bool& { return stop_; }; r() = true", *_dc("{ auto r = [this]() -> bool& { return stop_; }; r() = true; }")),
    ("R9", "decltype((stop_)) q = stop_; q = true", *_dc("{ decltype((stop_)) q = stop_; q = true; }")),
    ("R10", "aggregate with a reference member: struct H { bool& r; }; H h{stop_}; h.r = true", *_dc("{ struct H { bool& r; }; H h{stop_}; h.r = true; }")),
    ("R11", "move-from through a cast: auto q = static_cast<std::deque<Job*>&&>(queue_)", *_dc("{ auto q = static_cast<std::deque<Job*>&&>(queue_); (void)q; }", "queue_")),
    ("R12", "span view copy-init: std::span<char> v = gov_json_; v[0] = 'x'", *_rl("if (!gov_json_.empty()) { std::span<char> v = gov_json_; v[0] = 'x'; }")),
    ("R13", "span view brace-init: std::span<char> v{gov_json_}; v[0] = 'x'", *_rl("if (!gov_json_.empty()) { std::span<char> v{gov_json_}; v[0] = 'x'; }")),
    ("R14", "control: pointer typedef, address taken: using P = bool*; P p = &stop_; *p = true", *_dc("{ using P = bool*; P p = &stop_; *p = true; }")),
    ("R15", "range-for with a reference typedef: using JR = Job*&; for (JR j : queue_) j = nullptr", *_dc("{ using JR = Job*&; for (JR j : queue_) j = nullptr; }", "queue_")),
    ("R16", "accessor returning reference_wrapper: holdRef().get() = true",
     *_acc("std::reference_wrapper<bool> holdRef() { return gov_hold_; }", "holdRef().get() = true;")),
    ("R17", "accessor returning std::add_pointer_t<bool>: *holdPtr() = true", *_acc("std::add_pointer_t<bool> holdPtr() { return &gov_hold_; }", "*holdPtr() = true;", name="holdPtr")),
    ("R18", "control: vram_probe_ = nullptr in runLoop (core.shared write)", [(EC, RUNL, RUNL + "    vram_probe_ = nullptr;\n")], None, [], sw("runLoop:vram_probe_")),
    ("R19", "std::function member: non-const target<T>() handed out", [(EC, RUNL, RUNL + "    (void)vram_probe_.target<int>();\n")], None, [], sw("runLoop:vram_probe_")),
    ("R20", "std::function member: vram_probe_.swap(f)", [(EC, RUNL, RUNL + "    { std::function<wddmwatch::Sample()> f; vram_probe_.swap(f); }\n")], None, [], sw("runLoop:vram_probe_")),
    # read controls (S0c review consider 2): a write finding here is a FALSE POSITIVE
    ("F1", "read: std::find(queue_.begin(), queue_.end(), nullptr) != queue_.end() (the review's F1 with whirl::MutexLock; as written it does not compile, F1x)",
     *_okc("{ whirl::MutexLock lk(q_mutex_); (void)(std::find(queue_.begin(), queue_.end(), nullptr) != queue_.end()); }")),
    ("F1x", "the review's F1 as written (std::lock_guard<std::mutex> on a whirl::Mutex) does not compile: the AST census fails closed",
     [(EC, RUNL, RUNL + "    { std::lock_guard<std::mutex> lk(q_mutex_); (void)(std::find(queue_.begin(), queue_.end(), nullptr) != queue_.end()); }\n")],
     None, [], fnd("clang errors in src/server/engine.cpp")),
    ("F2", "read: std::format(\"{}\", gov_last_) into a local", *_okc("{ const std::string s = std::format(\"{}\", gov_last_); (void)s; }")),
    ("F3", "read: std::size(gov_last_)", *_okc("{ const auto n = std::size(gov_last_); (void)n; }")),
    ("F4", "read: throw std::runtime_error(gov_last_) under a false condition", *_okc("if (gov_last_ == \"never\") throw std::runtime_error(gov_last_);")),
    ("F5", "read: std::any_of over slots_.begin()/end()", *_okc("(void)std::any_of(slots_.begin(), slots_.end(), [](const Slot& s) { return s.job != nullptr; });")),
    ("F6", "read: const auto& r = gov_last_; copy", *_okc("{ const auto& r = gov_last_; std::string c = r; (void)c; }")),
    ("F7", "read: std::string_view v{gov_last_}", *_okc("{ std::string_view v{gov_last_}; (void)v; }")),
    ("F8", "read: lambda init-capture copy [h = gov_hold_]", *_okc("{ auto f = [h = gov_hold_] { return h; }; (void)f(); }")),
    ("F9", "read: by-value auto accessor const auto h = holdVal()",
     [(EH, HOLD, HOLD + "\n    auto holdVal() { return gov_hold_; }"), (EC, RUNL, RUNL + "    { const auto h = holdVal(); (void)h; }\n")],
     lambda c: c["new_funcs"].update({"holdVal": GV}), ["--no-pin"], r"(?#OK)^engine split check: ok"),
    # S0d's own probes
    ("D1", "S0d: a new std::function member (governor data) called in runLoop, not in callable_reads: a write",
     [(EH, HOLD, HOLD + "\n    std::function<void()> on_gov_;"), (EC, RUNL, RUNL + "    if (on_gov_) on_gov_();\n")],
     lambda c: c["modules"][GV]["data"].append("on_gov_"), ["--no-pin"], ow("runLoop:on_gov_")),
    ("D2", "S0d control: vram_probe_ (in callable_reads) called in govTick stays a read",
     [(EG, GOVT, GOVT + "    if (vram_probe_) (void)vram_probe_();\n")], None, [], r"(?#OK)^engine split check: ok"),
    ("D3", "S0d: an accessor's pointer kept in a variable and written: auto p = holdPtr(); *p = true",
     *_acc("bool* holdPtr() { return &gov_hold_; }", "{ auto p = holdPtr(); *p = true; }", name="holdPtr")),
    ("D4", "S0d: a holder in a member initializer (the S4 dc_{bst_, m_, opt_} shape): std::reference_wrapper<bool> hd_{stop_}",
     [(EH, HOLD, HOLD + "\n    std::reference_wrapper<bool> hd_{stop_};")], lambda c: c["modules"][GV]["data"].append("hd_"), ["--no-pin"], ow("Engine:stop_")),
    # --- S0d review (progress/arch025-engine.S0d.REVIEW.md act-on 1; reviewer runner scratch/a25_engine/s0d_review/runner/r_mut.py):
    # handles returned by std algorithms are followed (the S0d tools ended the walk at the algorithm: rc 0) ---
    ("Q1c", "control: queue_.front() = nullptr", *_dc("if (!queue_.empty()) queue_.front() = nullptr;", "queue_")),
    ("Q1", "std::find result written: *it = nullptr", *_dc("{ auto it = std::find(queue_.begin(), queue_.end(), nullptr); if (it != queue_.end()) *it = nullptr; }", "queue_")),
    ("Q2", "std::find_if(...)->job = nullptr on slots_ (core.shared)",
     [(EP, DROPC, DROPC + "    if (!slots_.empty()) std::find_if(slots_.begin(), slots_.end(), [](const Slot& s) { return s.job != nullptr; })->job = nullptr;\n")],
     None, [], r"^== writes to core\.shared not in shared_writers '\w+' \(\d+\): .*\bdropCache:slots_\b|^   not in the expected list '\w+': .*\bdropCache:slots_\b"),
    ("Q2c", "control: slots_[0].job = nullptr (core.shared)", [(EP, DROPC, DROPC + "    if (!slots_.empty()) slots_[0].job = nullptr;\n")],
     None, [], r"^== writes to core\.shared not in shared_writers '\w+' \(\d+\): .*\bdropCache:slots_\b|^   not in the expected list '\w+': .*\bdropCache:slots_\b"),
    ("Q3", "*std::max_element(crlf_norm_) = 0", *_dc("if (!crlf_norm_.empty()) *std::max_element(crlf_norm_.begin(), crlf_norm_.end()) = 0;", "crlf_norm_")),
    ("Q4", "*std::next(queue_.begin()) = nullptr", *_dc("if (queue_.size() > 1) *std::next(queue_.begin()) = nullptr;", "queue_")),
    ("Q5", "*std::lower_bound(crlf_norm_) = 1", *_dc("if (!crlf_norm_.empty()) *std::lower_bound(crlf_norm_.begin(), crlf_norm_.end(), 0u) = 1;", "crlf_norm_")),
    ("Q6", "std::exchange(*std::find(...), nullptr)", *_dc("{ auto it = std::find(queue_.begin(), queue_.end(), nullptr); if (it != queue_.end()) (void)std::exchange(*it, nullptr); }", "queue_")),
    # act-on 2: a text-write / AST-read row is a finding unless the cut pins it (and a pinned row must still occur)
    # (S0e, coordinator 08:5x: a read the AST proves from types / the call target is no finding; a read that rests on a
    # measured name list - here std::count, a value algorithm - needs the pin)
    ("T1", "S0e: text write / AST read through a name list (std::count over gov_json_.begin()/end()), not pinned: a finding",
     [(EC, RUNL, RUNL + "    (void)std::count(gov_json_.begin(), gov_json_.end(), 'x');\n")], None, [],
     fnd("text census write the AST calls a read, not pinned in the cut's text_write_ok: runLoop:gov_json_")),
    ("T1b", "S0e control: the same row pinned in text_write_ok with the text rule's exact reason passes",
     [(EC, RUNL, RUNL + "    (void)std::count(gov_json_.begin(), gov_json_.end(), 'x');\n")],
     lambda c: c["text_write_ok"].update({"runLoop:gov_json_": {"text_why": _T1_WHY, "reason": "selftest"}}), ["--no-pin"],
     r"(?#OK)^engine split check: ok"),
    ("T2", "S0e: a text_write_ok pin whose text reason differs does not cover the row",
     [(EC, RUNL, RUNL + "    (void)std::count(gov_json_.begin(), gov_json_.end(), 'x');\n")],
     lambda c: c["text_write_ok"].update({"runLoop:gov_json_": {"text_why": "other", "reason": "selftest"}}), ["--no-pin"],
     fnd("text census write the AST calls a read, not pinned in the cut's text_write_ok: runLoop:gov_json_")),
    ("T3", "S0e: a stale text_write_ok entry is a finding", [], lambda c: c["text_write_ok"].update({"runLoop:stop_": {"text_why": "x", "reason": "x"}}),
     ["--no-pin"], fnd("cut text_write_ok entry no longer occurs (remove it): runLoop:stop_")),
    ("D4b", "S0d: the same binding in a declarator after a nested struct body (struct Hd { bool& r; } hd_{stop_}; the text census has no "
     "entry for it): the AST write outside every entry is a finding",
     [(EH, HOLD, HOLD + "\n    struct Hd { bool& r; } hd_{stop_};")], lambda c: c["modules"][GV]["data"].append("hd_"), ["--no-pin"],
     fnd("AST write outside every census entry: server/engine.h:")),
]
# S0c: tools a TOOL: edit may name -> path under tools/ (a copy of tools/gates, bench, impact is made for the mutant)
TOOL_PATHS = {"e8_judge.py": "bench/e8_judge.py", "abjudge.py": "bench/abjudge.py", "revtext.py": "impact/revtext.py"}


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
            if f.startswith(("engine_split_", "engine_cut", "member_order", "mt_equiv_")):
                shutil.copy2(os.path.join(HERE, f), tools)
        for rp in TOOL_PATHS.values():   # S0c: bench / impact tools are hashed too
            src = os.path.join(os.path.dirname(HERE), rp)
            if os.path.exists(src):
                os.makedirs(os.path.join(d, "tools", os.path.dirname(rp)), exist_ok=True)
                shutil.copy2(src, os.path.join(d, "tools", rp))
    for rel, old, new in edits:
        if rel.startswith("TOOL:"):
            tp = os.path.join(d, "tools", TOOL_PATHS[rel[5:]]) if rel[5:] in TOOL_PATHS else os.path.join(tools, rel[5:])
            open(tp, "a", encoding="utf-8").write(new); continue
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
    ledger = LEDGER
    if "@PINWRITE" in gargs:   # S0b: the step rewrites the pin in the same commit (no review row added)
        pin = os.path.join(d, "engine_cut_arch.sha256")
        run([PY, "-I", os.path.join(tools, "engine_split_check.py"), "--pin-write", "--cut", cutp, "--pin", pin])
    for g in gargs:   # S0c: a ledger row added in the same change (@ROW:kind) or an old row edited (@LEDGEREDIT)
        if g.startswith("@ROW:") or g == "@LEDGEREDIT":
            ledger = os.path.join(d, "engine_cut_reviews.tsv")
            txt = open(LEDGER, encoding="utf-8").read()
            if g == "@LEDGEREDIT":
                rows = txt.split("\n")
                k = next(i for i, l in enumerate(rows) if l.strip() and not l.startswith("#"))
                q = rows[k].split("\t"); q[1] = "progress/arch025-engine.S0b.REVIEW.md (rewritten afterwards)"; rows[k] = "\t".join(q)
                txt = "\n".join(rows)
            else:
                kind = g[5:]
                sha = hashlib.sha256(open(pin, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
                rp = os.path.join(d, "REVIEW.md")
                body = {"nosha": "reviewer: rev-agent\nlooked at the cut\n", "self": f"pin sha256: {sha}\nreviewer: step-agent\n"}
                open(rp, "w", encoding="utf-8").write(body.get(kind, f"pin sha256: {sha}\nreviewer: rev-agent\n"))
                row = {"xx": f"{sha}\tx\tx", "pending": f"{sha}\t{rp} (review pending)\tcut edit\tstep-agent"}.get(
                    kind, f"{sha}\t{rp} (rev-agent)\tcut edit\tstep-agent")
                txt = txt.rstrip("\n") + "\n" + row + "\n"
            open(ledger, "w", encoding="utf-8").write(txt)
    gargs = [g for g in gargs if not g.startswith("@")]
    cmd = [PY, "-I", os.path.join(tools, "engine_split_check.py"), "--root", d, "--cut", cutp, "--work", os.path.join(d, "w"),
           "--pin", pin, "--reviews", ledger, "--base-dir", BASESNAP, "--inline", INL] + gargs
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
    if expect.startswith("(?#OK)"):   # S0c: a control that must pass
        v = "PASS" if rc == 0 and hit else "FALSE POSITIVE"
    elif rc == 3: v = "CRASH"
    elif rc == 1 and hit: v = "CAUGHT"
    elif rc == 1: v = "CAUGHT BY ANOTHER RULE"
    else: v = "NOT CAUGHT"
    m = re.search(expect, allt, re.M)
    shown = m.group(0)[:160] if m else "(expected line not printed: /" + expect[:120] + "/)"
    return mid, v, f"{mid} {what}\n   rc={rc}; expected line: {shown} -> {v}"


def setup_ledger():
    """S0c: the self-test does not depend on today's review state: a synthetic report + ledger row for today's pin, and a
    base snapshot (cut, pin, that ledger) = "nothing changed in this range". The real gate run (no --base-dir) judges
    the real ledger against the real merge-base."""
    os.makedirs(os.path.join(WORK, "ledger", "base"), exist_ok=True)
    sha = hashlib.sha256(open(PIN, "rb").read().replace(b"\r\n", b"\n")).hexdigest()
    rp = os.path.join(WORK, "ledger", "REVIEW.md")
    open(rp, "w", encoding="utf-8").write(f"pin sha256: {sha}\nreviewer: selftest-reviewer\n")
    txt = open(REVIEWS, encoding="utf-8").read().replace("\r\n", "\n").rstrip("\n") + f"\n{sha}\t{rp} (selftest)\tselftest baseline\tselftest-author\n"
    open(LEDGER, "w", encoding="utf-8", newline="\n").write(txt)
    shutil.copy2(LEDGER, os.path.join(BASESNAP, "engine_cut_reviews.tsv"))
    shutil.copy2(PIN, os.path.join(BASESNAP, "engine_cut_arch.sha256"))
    shutil.copy2(CUT, os.path.join(BASESNAP, "engine_cut.json"))
    return sha


def git_cases(sha):
    """S0c: the git-mode commit-order rule of pin_check (GitHub 'approval of the most recent push': the review row must
    land in a later commit than the pin change), with a synthetic base state (no git writes)"""
    sys.path.insert(0, HERE)
    try:
        import engine_split_check as esc
    except Exception as ex:
        return [(f"G{i}", "CRASH", f"G{i} import engine_split_check: {ex} -> CRASH") for i in range(1, 7)]
    d = os.path.join(WORK, "G"); os.makedirs(d, exist_ok=True)
    pin = os.path.join(d, "pin.sha256")
    open(pin, "w", encoding="utf-8", newline="\n").write(open(PIN, encoding="utf-8").read().replace("\r\n", "\n") + "# changed in the range\n")
    psha = esc.sha_text(pin)
    rp = os.path.join(d, "REVIEW.md"); open(rp, "w", encoding="utf-8").write(f"pin sha256: {psha}\nreviewer: rev-agent\n")
    led = os.path.join(d, "reviews.tsv")
    row = f"{psha}\t{rp} (rev-agent)\tcut edit\tstep-agent"
    base_led = open(LEDGER, encoding="utf-8").read()
    open(led, "w", encoding="utf-8", newline="\n").write(base_led.rstrip("\n") + "\n" + row + "\n")
    ptxt = open(pin, encoding="utf-8").read()
    bpin = open(PIN, encoding="utf-8").read().replace("\r\n", "\n")
    mk = lambda **g: {"ref": "deadbeef", "cut": open(CUT, encoding="utf-8").read(), "pin": bpin, "reviews": base_led,
                      "git": dict({"head_pin": ptxt, "pin_commit": "c0ffee00", "pin_commit_reviews": base_led}, **g)}
    out = []
    for gid, what, base, want in (
            ("G1", "git: the pin change is uncommitted, its row too -> SELF-REVIEWED", mk(head_pin=bpin), r"REVIEW ROW INVALID .*SELF-REVIEWED: the pin change is not committed"),
            ("G2", "git: the row is in the same commit as the pin change -> SELF-REVIEWED", mk(pin_commit_reviews=base_led + row + "\n"), r"REVIEW ROW INVALID .*SELF-REVIEWED: the row was added in the same commit"),
            ("G3", "control: git, the row lands after the pin commit -> ok", mk(), None),
            ("G4", "no base (no merge-base): today's 3-column rows are not valid reviews -> REVIEW ROW INVALID", None, r"REVIEW ROW INVALID"),
            ("G5", "empty range (merge-base == HEAD on the integration branch), only a '<sha> x x' row -> REVIEW ROW INVALID", "empty",
             r"REVIEW ROW INVALID .*needs 4 non-empty columns"),
            ("G6", "S0d (S0c review consider 3): a base without the cut file (an integration branch without arch025-engine) fails closed",
             dict(mk(), cut=None), r"BASE CUT MISSING")):
        try:
            if gid == "G5":
                led5 = os.path.join(d, "reviews_xx.tsv")
                open(led5, "w", encoding="utf-8").write(f"{sha}\tx\tx\n")
                lines = esc.pin_check(CUT, PIN, led5, {"ref": "head", "cut": open(CUT, encoding="utf-8").read(), "pin": bpin,
                                                       "reviews": open(led5, encoding="utf-8").read(), "git": None})
            elif gid == "G4":   # the S0b ledger row shape (3 columns, no author)
                led4 = os.path.join(d, "reviews_s0b_shape.tsv")
                open(led4, "w", encoding="utf-8").write(f"{sha}\tprogress/arch025-engine.S0b.REVIEW.md (s0b-review)\tS0b row shape\n")
                lines = esc.pin_check(CUT, PIN, led4, None)
            else:
                lines = esc.pin_check(CUT, pin, led, base)
            txt = "\n".join(lines)
            ok = (not lines) if want is None else re.search(want, txt) is not None
            v = ("PASS" if ok else "FALSE POSITIVE") if want is None else ("CAUGHT" if ok else "NOT CAUGHT")
        except Exception as ex:
            txt, v = f"{type(ex).__name__}: {ex}", "CRASH"
        out.append((gid, v, f"{gid} {what}\n   {txt[:200] or '(no pin lines)'} -> {v}"))
    return out


def main():
    os.makedirs(WORK, exist_ok=True)
    os.makedirs(BASESNAP, exist_ok=True)
    sha = setup_ledger()
    res = []
    for i, b in enumerate(bases):   # baselines
        rc, out = run([PY, "-I", os.path.join(HERE, "engine_split_check.py"), "--root", b, "--work", os.path.join(WORK, f"B{i}"), "--pin", PIN,
                       "--reviews", LEDGER, "--base-dir", BASESNAP])
        ok = rc == 0 and "engine split check: ok" in out
        res.append((f"B{i}", "PASS" if ok else "FAIL", f"B{i} baseline {b}\n   rc={rc}: {out.strip().splitlines()[-1] if out.strip() else ''} -> {'PASS' if ok else 'FAIL (false positive)'}"))
    with ThreadPoolExecutor(JOBS) as ex:
        res += list(ex.map(one, MUT))
    res += git_cases(sha)
    # S0e: the AST rules' unit self-test (S0d review probes on a synthetic TU: act-on 1, 3, 4, consider 1, 2, 5)
    rc, out = run([PY, "-I", os.path.join(HERE, "engine_split_ast.py"), "--selftest"])
    last = out.strip().splitlines()[-1] if out.strip() else "(no output)"
    v = "PASS" if rc == 0 and re.search(r"^engine split ast selftest: (\d+)/\1 as expected", out, re.M) else "FAIL"
    bad = [l.strip() for l in out.splitlines() if l.startswith(("  MISS", "  FP"))]
    res.append(("A1", v, f"A1 engine_split_ast.py --selftest (synthetic TU)\n   rc={rc}: {last}" + "".join("\n   " + b for b in bad[:12]) + f" -> {v}"))
    good = sum(1 for _, v, _ in res if v in ("PASS", "CAUGHT"))
    counted = sum(1 for _, v, _ in res if v != "SKIP")
    txt = "\n".join(r[2] for r in res)
    txt += f"\nengine split selftest: {good}/{counted} as expected" + ("" if good == counted else " - FAIL")
    open(os.path.join(WORK, "engine_split_selftest.txt"), "w", encoding="utf-8").write(txt + "\n")
    print(txt)
    return 0 if good == counted else 1


if __name__ == "__main__":
    sys.exit(main())
