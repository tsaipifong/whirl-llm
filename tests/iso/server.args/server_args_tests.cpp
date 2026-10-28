// whirl-iso-server-args-tests: src/server/server_args.cpp built with only its declared dependencies (numerics::resolve
// of src/common/numerics.cpp, its dump::kv in src/common/dump.cpp, platform env) (ARCH-1 8.5 isolation, arch025-isoall).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "server/server_args.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace whirl::server;

namespace {

bool visOk(std::string_view m) { return m == "auto" || m == "gpu" || m == "cpu"; }

void testDefaults() {
    const Options o = parseServerArgs({"m.gguf"}, visOk);
    ISO_CHECK(o.path == "m.gguf" && o.host == "127.0.0.1" && o.port == 8080 && o.parallel == 4 && !o.parallel_given);
    ISO_CHECK(!o.mode_spec.has_value());
    ISO_THROWS(parseServerArgs({}, visOk), ArgError);  // missing MODEL.gguf
}

void testFlags() {
    const Options o = parseServerArgs({"m.gguf", "--port", "9001", "--parallel", "2"}, visOk);
    ISO_CHECK(o.port == 9001 && o.parallel == 2 && o.parallel_given);
    ISO_THROWS(parseServerArgs({"m.gguf", "--port", "x"}, visOk), ArgError);
    ISO_THROWS(parseServerArgs({"m.gguf", "--no-such-option"}, visOk), ArgError);
}

void testResolve() {
    Options o;
    const ResolvedConfig r = resolveServerConfig(o, std::nullopt);
    ISO_CHECK(r.opt.parallel == 4);
    Options bad;
    bad.parallel = 17;
    ISO_THROWS(resolveServerConfig(bad, std::nullopt), ArgError);
    Options d;
    ISO_THROWS(checkKvSsdDirEnv(d, std::optional<std::string>("")), ArgError);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("server.args", {{"defaults", testDefaults}, {"flags", testFlags}, {"resolve", testResolve}}, argc, argv);
}
