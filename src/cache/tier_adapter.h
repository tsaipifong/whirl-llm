// cache::Tier over the host RAM / SSD tier (tier::Tier).
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cache/prefix_cache.h"

namespace whirl::tier {
class Tier;
}

namespace whirl::cache {

std::unique_ptr<Tier> makeTierAdapter(tier::Tier& t);

}  // namespace whirl::cache
