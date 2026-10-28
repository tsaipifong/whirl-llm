// whirl-iso-server-args-tests: src/server/server_args.cpp built with only its declared dependencies (numerics::resolve
// of src/common/numerics.cpp, its dump::kv in src/common/dump.cpp, platform env) (ARCH-1 8.5 isolation, arch025-isoall).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "server/server_args.h"

#include <cctype>
#include <cstdio>
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

// arch025-lane1 leg 8: the model name /v1/models and the startup lines show (5728296 src/server/server_main.cpp:953-958)
std::string old_name(const std::string& base, const std::optional<std::string>& alias) {
    std::string name = alias ? *alias : base;
    if (!alias && name.size() > 5) {
        std::string tail = name.substr(name.size() - 5);
        for (char& c : tail) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (tail == ".gguf") name = name.substr(0, name.size() - 5);
    }
    return name;
}

// :1543-1544 the SSD tier mode: --kv-ssd-mode, else WHIRL_KV_SSD_MODE, else "shutdown"; anything but always / off = shutdown
std::string old_ssd_mode(const std::optional<std::string>& cli, const std::optional<std::string>& env) {
    std::string ssd_mode = cli.value_or(env.value_or("shutdown"));
    if (ssd_mode != "always" && ssd_mode != "off") ssd_mode = "shutdown";
    return ssd_mode;
}

void testKvSsdMode() {
    int n = 0;
    const std::optional<std::string> vs[] = {std::nullopt, std::string("always"), std::string("off"), std::string("shutdown"), std::string(""),
                                             std::string("Always"), std::string("x")};
    for (const auto& c : vs)
        for (const auto& e : vs) {
            ISO_CHECK(kvSsdMode(c, e) == old_ssd_mode(c, e));
            ++n;
        }
    std::printf("kv ssd mode: %d cases == 5728296\n", n);
}

void testServedModelName() {
    int n = 0;
    for (const char* b : {"", "a", ".gguf", "x.gguf", "Swift-1.5-27B-MXFP4.GGUF", "m.GgUf", "model.gguf.bak", "model.bin", "ab.ggu",
                          "\xe6\xa8\xa1\xe5\x9e\x8b.gguf", "a.gguf.gguf", "     .gguf"})
        for (const std::optional<std::string>& al : {std::optional<std::string>(), std::optional<std::string>(""),
                                                     std::optional<std::string>("alias.gguf"), std::optional<std::string>("x")}) {
            ISO_CHECK(servedModelName(b, al) == old_name(b, al));
            ++n;
        }
    ISO_CHECK(servedModelName("Swift.gguf", std::nullopt) == "Swift" && servedModelName("Swift.gguf", std::string("s")) == "s");
    std::printf("served model name: %d cases == 5728296\n", n);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("server.args", {{"defaults", testDefaults}, {"flags", testFlags}, {"resolve", testResolve}, {"served_model_name", testServedModelName}, {"kv_ssd_mode", testKvSsdMode}}, argc, argv);
}
