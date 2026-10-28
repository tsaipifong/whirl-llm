// qwen35::ModelError, the exception every model-layer part throws (code() names the condition). Moved verbatim from
// include/whirl/model.h (arch025-lane2 N9 S2c) so the model's lower modules (model.kv, model.scratch, model.weights)
// throw it without including the Model class; model.h includes this header where the class used to be.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdexcept>
#include <string>
#include <utility>

namespace whirl::qwen35 {

class ModelError : public std::runtime_error {
public:
    ModelError(std::string code, const std::string& detail = {})
        : std::runtime_error(detail.empty() ? code : code + ": " + detail), code_(std::move(code)) {}
    const std::string& code() const { return code_; }

private:
    std::string code_;
};

}  // namespace whirl::qwen35
