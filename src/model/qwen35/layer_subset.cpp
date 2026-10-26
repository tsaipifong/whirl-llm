// Model::LayerSubset (include/whirl/model.h): Model::run over a subset of the trunk layers, the stage-bench seam
// (tests/bench/stage_bench_main.cpp; arch025-M3 C7: moved here verbatim from there so the bench no longer writes the
// Model's members). Test-only: the product never constructs one.
// SPDX-License-Identifier: Apache-2.0

#include "whirl/model.h"

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace whirl::qwen35 {

namespace {

using u32 = std::uint32_t;

}  // namespace

Model::LayerSubset::LayerSubset(Model& m, const std::vector<u32>& sel) : m_(m) {
    layers0_ = m.layers;
    kc0_ = m.kcache, vc0_ = m.vcache, ks0_ = m.kscale, vs0_ = m.vscale;
    conv0_ = m.conv_state, ssm0_ = m.ssm_state;
    const std::size_t L = m.cfg.n_layer;
    std::vector<Layer> nl;
    std::vector<hip::DevPtr> kc(L, 0), vc(L, 0), ks(L, 0), vs(L, 0);
    conv_.assign(L, 0);
    ssm_.assign(L, 0);
    for (std::size_t j = 0; j < sel.size(); ++j) {
        const u32 s = sel[j];
        nl.push_back(layers0_.at(s));
        kc[j] = kc0_[s], vc[j] = vc0_[s], ks[j] = ks0_[s], vs[j] = vs0_[s];
        conv_[j] = conv0_[s], ssm_[j] = ssm0_[s];
    }
    m.layers = std::move(nl);
    m.kcache = kc, m.vcache = vc, m.kscale = ks, m.vscale = vs;
    m.conv_state = std::span<hip::DevPtr>(conv_);
    m.ssm_state = std::span<hip::DevPtr>(ssm_);
    m.decode_graph.reset();
}

Model::LayerSubset::~LayerSubset() {
    m_.layers = std::move(layers0_);
    m_.kcache = kc0_, m_.vcache = vc0_, m_.kscale = ks0_, m_.vscale = vs0_;
    m_.conv_state = conv0_, m_.ssm_state = ssm0_;
    m_.decode_graph.reset();
}

}  // namespace whirl::qwen35
