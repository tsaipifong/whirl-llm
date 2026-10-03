// Entry point of the server executable (whirl-server, or `whirl serve`).
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

namespace whirl::server {

// Parses the command line (UTF-8; argv[0] is the program or the `serve`
// subcommand, options start at argv[1]), loads the model and
// serves until the process ends. Returns the process exit code.
// `program` names the command in help and error messages.
int serveMain(int argc, char** argv, const char* program = "whirl-server");

// The complete --help text (options, environment variables, exit codes).
std::string serveHelp(const char* program);

}  // namespace whirl::server
