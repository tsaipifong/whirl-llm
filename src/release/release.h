// Release front-end shared by whirl.exe and whirl-server.exe: version,
// console setup, the GPU runtime pre-flight check, plain-language error
// messages with distinct exit codes, and the environment-variable reference
// printed by the --help texts.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>

namespace whirl::app {

// Process exit codes (documented in `whirl --help` and docs/cli.md).
enum Exit : int {
    exit_ok = 0,
    exit_error = 1,   // anything not listed below
    exit_usage = 2,   // bad command line
    exit_gpu = 3,     // no supported AMD GPU, driver missing / too old, GPU busy
    exit_model = 4,   // model file missing, not a GGUF, unsupported architecture / tensor type
    exit_vram = 5,    // out of GPU memory (or pinned host memory)
    exit_port = 6,    // server: the address / port is already in use
};

// An error whose message is already written for the user.
class UserError : public std::runtime_error {
public:
    UserError(int exit_code, const std::string& message) : std::runtime_error(message), code_(exit_code) {}
    int exitCode() const { return code_; }

private:
    int code_;
};

const char* version();                                   // "0.1.1"
std::string versionText(std::string_view program);       // multi-line text for --version

// UTF-8 console output while the process runs (the previous code page is put
// back at exit). Redirected output is not affected (it is UTF-8 bytes either way).
void initConsole();

// Checks that the AMD GPU runtime of the graphics driver (amdhip64_7.dll) can be
// loaded before any HIP call. Returns exit_ok, or prints a plain-language message
// and returns exit_gpu. Also installs the handler for a driver that lacks a HIP
// function WHIRL uses (an older driver): message + exit_gpu instead of a crash.
int gpuPreflight();

// Throws UserError(exit_model) when the model file does not exist / is a directory.
void requireModelFile(const std::string& path, std::string_view what = "model file");

// Throws UserError(exit_port) when host:port cannot be bound right now (checked
// before the model is loaded so the user does not wait for nothing).
void requirePortFree(const std::string& host, std::uint16_t port);

// Plain-language message for an exception escaping a command: printed to stderr;
// returns the exit code. `program` prefixes the first line ("whirl", "whirl-server").
int explain(const std::exception& e, std::string_view program);
// Same text without printing (for the server log).
std::string explainText(const std::exception& e, int* exit_code);

// Environment variables, for the --help texts. `scope` is a bit mask of the
// commands that read a variable.
enum Scope : unsigned {
    sc_chat = 1,
    sc_bench = 2,
    sc_selftest = 4,
    sc_seqtest = 8,
    sc_serve = 16,
    sc_vis = 32,
    sc_all = 63,
};
// "  WHIRL_NAME=VALUES   description\n" lines of every variable read by `scope`,
// grouped under headings.
std::string envHelp(unsigned scope);
// The exit-code table for help texts.
std::string exitCodeHelp();

}  // namespace whirl::app
