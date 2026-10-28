// Model::writeTune / readTune: the tune cache text (qwen35::tuneText / applyTuneText, src/model/qwen35/tune_table.cpp)
// over every layer's tuneMats. Moved verbatim from tune_table.cpp (arch025-lane2), so the tune table module no longer
// includes whirl/model.h; this file is the model core's.
// SPDX-License-Identifier: Apache-2.0

#include "model/qwen35/tune_table.h"

#include "whirl/model.h"

#include <string>
#include <string_view>
#include <vector>

namespace whirl::qwen35 {

std::string Model::writeTune() const {
    std::vector<Mat*> mats;
    for (const Layer& L : layers) {
        Layer& LL = const_cast<Layer&>(L);
        for (Mat* m : const_cast<Model*>(this)->tuneMats(LL)) mats.push_back(m);
    }
    return tuneText(mats);
}

bool Model::readTune(std::string_view text, std::size_t n_buckets) {
    std::vector<Mat*> mats;
    for (Layer& L : layers)
        for (Mat* m : tuneMats(L)) mats.push_back(m);
    return applyTuneText(text, n_buckets, mats);
}

}  // namespace whirl::qwen35
