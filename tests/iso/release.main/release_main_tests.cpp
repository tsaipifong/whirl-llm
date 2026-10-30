// whirl-iso-release-main-tests: src/release/release.cpp (the front-end part of the module release: --version text, the error
// explanations and exit codes, the model-file / port pre-checks, the environment help) built with its declared dependencies only
// and run (ARCH-1 8.5 isolation, arch025-n8b2, PLAN N8 batch 2; step cpu_iso_release_main). Host-only cases: nothing here asks the
// GPU runtime for its device list (the NoTargetGpu / driver-problem texts do; they are the GPU steps' business).
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"

#include "model/qwen35/model_error.h"
#include "release/release.h"
#include "whirl/gguf.h"
#include "whirl/hip.h"
#include "whirl/version.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <new>
#include <stdexcept>
#include <string>

namespace {

namespace app = whirl::app;
using whirl::qwen35::ModelError;

struct Explained {
    int code = -1;
    std::string msg;
};

Explained ex(const std::exception& e) {
    Explained r;
    r.msg = app::explainText(e, &r.code);
    return r;
}

bool has(const std::string& s, const std::string& part) { return s.find(part) != std::string::npos; }

void testVersionText() {
    ISO_CHECK(std::string(app::version()) == WHIRL_VERSION_STRING);
    const std::string v = app::versionText("whirl-server");
    ISO_CHECK(v.rfind(std::string("whirl-server ") + WHIRL_VERSION_STRING + " (WHIRL", 0) == 0);
    ISO_CHECK(has(v, std::string("GPU kernels for: ") + WHIRL_GPU_ARCHS_STRING + "\n") && has(v, "License: Apache-2.0\n"));
    ISO_CHECK(has(app::exitCodeHelp(), "6 server port in use") && has(app::exitCodeHelp(), "0 ok, 1 other error, 2 bad command line"));
}

void testExplainUserError() {
    const Explained r = ex(app::UserError(app::exit_port, "port taken"));
    ISO_CHECK(r.code == app::exit_port && r.msg == "port taken\n");
    ISO_CHECK(app::explain(app::UserError(app::exit_usage, "bad flag"), "whirl-iso") == app::exit_usage);  // prints one stderr line
    ISO_CHECK(app::explainText(app::UserError(app::exit_usage, "x"), nullptr) == "x\n");                 // no exit code wanted
}

void testExplainModelErrors() {
    Explained r = ex(ModelError("UnsupportedArch", "llama"));
    ISO_CHECK(r.code == app::exit_model && has(r.msg, "architecture is not supported (UnsupportedArch: llama)") && has(r.msg, "qwen35moe"));
    for (const char* c : {"UnsupportedTensorType", "UnsupportedMoeShape", "UnsupportedTensorShape", "UnsupportedConfig", "MissingTensor",
                          "Truncated", "OpenFailed"}) {
        r = ex(ModelError(c, "d"));
        ISO_CHECK(r.code == app::exit_model && has(r.msg, "this GGUF file cannot be loaded ("));
    }
    r = ex(ModelError("VramLimit", "need 40 GiB"));
    ISO_CHECK(r.code == app::exit_vram && has(r.msg, "does not fit in the GPU's memory (VramLimit: need 40 GiB)"));
    r = ex(ModelError("GpuHidden", "HIP_VISIBLE_DEVICES=7 hides every GPU"));
    ISO_CHECK(r.code == app::exit_gpu && r.msg == "no AMD GPU is visible to WHIRL: HIP_VISIBLE_DEVICES=7 hides every GPU.\n");
    r = ex(ModelError("DeviceChoice", "pick one: 0, 1"));
    ISO_CHECK(r.code == app::exit_usage && r.msg == "pick one: 0, 1\n");
    r = ex(ModelError("GpuBusy"));
    ISO_CHECK(r.code == app::exit_gpu && has(r.msg, "WHIRL_GPU_SHARE=1"));
    r = ex(ModelError("UmaTooSmall", "raise the carve-out\n"));
    ISO_CHECK(r.code == app::exit_vram && r.msg == "raise the carve-out\n");  // the detail as it is (it ends in its own newline)
    r = ex(ModelError("KvDoesNotFit", "ctx 1M"));
    ISO_CHECK(r.code == app::exit_vram && r.msg == "ctx 1M\n");
    r = ex(ModelError("UnsupportedKvFormat", "q2"));
    ISO_CHECK(r.code == app::exit_usage && r.msg == "UnsupportedKvFormat: q2\n");
    r = ex(ModelError("ContextTooLong", "9000 > 8192"));
    ISO_CHECK(r.code == app::exit_usage && has(r.msg, "do not fit in the context (ContextTooLong: 9000 > 8192)") && has(r.msg, "Raise --ctx"));
    r = ex(ModelError("SomethingNew", "z"));
    ISO_CHECK(r.code == app::exit_error && r.msg == "SomethingNew: z\n");
}

void testExplainHipErrors() {
    // only the texts that need no device list (codes 35 / 100 / 3 / ... print the driver's GPU list: GPU steps)
    Explained r = ex(whirl::hip::Error("VramLimit", 2, "simulated"));
    ISO_CHECK(r.code == app::exit_vram && has(r.msg, "simulated VRAM limit") && has(r.msg, "Unset WHIRL_VRAM_LIMIT_MB"));
    r = ex(whirl::hip::Error("hipMalloc", 1, "m"));
    ISO_CHECK(r.code == app::exit_vram && has(r.msg, "out of GPU memory ("));
    r = ex(whirl::hip::Error("hipLaunchKernel", 2, "oom"));
    ISO_CHECK(r.code == app::exit_vram && has(r.msg, "out of GPU memory ("));
    r = ex(whirl::hip::Error("hipHostMalloc", 1, "p"));
    ISO_CHECK(r.code == app::exit_vram && has(r.msg, "cannot allocate pinned host memory (") && has(r.msg, "--kv-ram-mb 2048"));
    r = ex(whirl::hip::Error("hipMemcpy", 1, "other"));
    ISO_CHECK(r.code == app::exit_error && has(r.msg, "other"));
}

void testExplainOther() {
    Explained r = ex(whirl::gguf::Error("gguf: bad magic"));
    ISO_CHECK(r.code == app::exit_model && has(r.msg, "this file is not a GGUF model ("));
    r = ex(whirl::gguf::Error("gguf: short read"));
    ISO_CHECK(r.code == app::exit_model && has(r.msg, "damaged or incomplete ("));
    for (const char* w : {"cannot open m.gguf", "cannot map m.gguf", "cannot stat m.gguf"}) {
        r = ex(std::runtime_error(w));
        ISO_CHECK(r.code == app::exit_model && has(r.msg, "Check that the file exists and is readable"));
    }
    r = ex(std::bad_alloc());
    ISO_CHECK(r.code == app::exit_vram && has(r.msg, "out of memory ("));
    r = ex(std::runtime_error("cannot listen on 127.0.0.1:8080 (10048)"));
    ISO_CHECK(r.code == app::exit_port && has(r.msg, "The port is already in use"));
    r = ex(std::runtime_error("cannot listen on 127.0.0.1:8080 (10049)"));
    ISO_CHECK(r.code == app::exit_error && r.msg == "cannot listen on 127.0.0.1:8080 (10049)\n");
    for (const char* w : {"unknown option --x", "missing value for --ctx", "--model: missing file"}) {
        r = ex(std::runtime_error(w));
        ISO_CHECK(r.code == app::exit_usage && r.msg == std::string(w) + "\n");
    }
    r = ex(std::runtime_error("plain"));
    ISO_CHECK(r.code == app::exit_error && r.msg == "plain\n");
}

int userErrorCode(const std::function<void()>& f, std::string* what) {
    try {
        f();
    } catch (const app::UserError& e) {
        if (what) *what = e.what();
        return e.exitCode();
    }
    return -1;
}

void testRequireModelFile() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "whirl_iso_release_main";
    std::error_code ec;
    fs::create_directories(dir, ec);
    const fs::path file = dir / "m.gguf";
    { std::ofstream(file) << "x"; }
    std::string w;
    ISO_CHECK(userErrorCode([&] { app::requireModelFile((dir / "none.gguf").string()); }, &w) == app::exit_model &&
              has(w, "model file not found: ") && has(w, "put it in quotes"));
    ISO_CHECK(userErrorCode([&] { app::requireModelFile(dir.string(), "projector"); }, &w) == app::exit_model &&
              has(w, "projector is a folder, not a file: "));
    ISO_CHECK(userErrorCode([&] { app::requireModelFile(file.string()); }, nullptr) == -1);
    fs::remove_all(dir, ec);
}

void testRequirePortFree() {
    ISO_CHECK(userErrorCode([] { app::requirePortFree("not an address", 0); }, nullptr) == -1);  // port 0: no check at all
    std::string w;
    ISO_CHECK(userErrorCode([&] { app::requirePortFree("not an address", 18089); }, &w) == app::exit_usage &&
              has(w, "invalid --host not an address"));
}

void testEnvHelp() {
    ISO_CHECK(app::envHelp(0).empty());  // nothing in scope: no header either
    const std::string all = app::envHelp(app::sc_all), serve = app::envHelp(app::sc_serve), vis = app::envHelp(app::sc_vis);
    ISO_CHECK(all.rfind("\nenvironment variables:\n  ", 0) == 0 && has(all, "    WHIRL_TIER_VERIFY=1"));
    ISO_CHECK(has(serve, "WHIRL_TIER_VERIFY=1") && !has(vis, "WHIRL_TIER_VERIFY=1"));
    ISO_CHECK(serve.size() < all.size() && !vis.empty());
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("release.main",
                    {{"version_text", testVersionText},
                     {"explain_user_error", testExplainUserError},
                     {"explain_model_errors", testExplainModelErrors},
                     {"explain_hip_errors", testExplainHipErrors},
                     {"explain_other", testExplainOther},
                     {"require_model_file", testRequireModelFile},
                     {"require_port_free", testRequirePortFree},
                     {"env_help", testEnvHelp}},
                    argc, argv);
}
