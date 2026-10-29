// whirl-iso-model-arch-open-tests: the header-only module model.arch_open (src/model/arch_open.h, the architecture module of an opened GGUF for whirl.exe
// and whirl-server) built with its declared dependencies (model.arch_registry, the GGUF reader) and run (ARCH-1 8.5 isolation, arch025-iso4). No test
// existed. Every call passes its own table (the product table, builtinArchs in src/model/qwen35/module.cpp, is not a dependency): the checks pin that a
// known name gives its row and the row's config, and an unknown one throws ModelError UnsupportedArch with the table's text.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "model/arch_open.h"

#include <array>
#include <cstdint>
#include <string>
#include <tuple>
#include <string_view>
#include <vector>

namespace {

using namespace whirl;

std::vector<std::uint8_t> archGguf(std::string_view arch) {
    std::vector<std::uint8_t> b;
    const auto put = [&](auto v) {
        const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
        b.insert(b.end(), p, p + sizeof v);
    };
    const auto str = [&](std::string_view x) {
        put(static_cast<std::uint64_t>(x.size()));
        b.insert(b.end(), x.begin(), x.end());
    };
    put(std::uint32_t(0x46554747));
    put(std::uint32_t(3));
    put(std::uint64_t(0));  // tensors
    if (arch.empty()) {
        put(std::uint64_t(0));  // no key at all
    } else {
        put(std::uint64_t(1));
        str("general.architecture");
        put(static_cast<std::uint32_t>(gguf::ValueType::string));
        str(arch);
    }
    while (b.size() % 32) b.push_back(0);
    return b;
}

qwen35::Config cfgA(const gguf::File&) {
    qwen35::Config c;
    c.n_layer = 7;
    return c;
}
qwen35::Config cfgB(const gguf::File&) {
    qwen35::Config c;
    c.n_layer = 11;
    return c;
}

const std::array<model::ArchName, 2> kNamesA{{{"famA", false}, {"famAmoe", true}}};
const std::array<model::ArchName, 1> kNamesB{{{"famB", false}}};
const std::array<model::ArchModule, 2> kTable{{{"A", kNamesA, cfgA}, {"B", kNamesB, cfgB}}};

void testKnownName() {
    const model::ArchTable t(kTable);
    for (const auto& [name, family, layers] : {std::tuple<const char*, const char*, int>{"famA", "A", 7}, {"famAmoe", "A", 7}, {"famB", "B", 11}}) {
        const auto bytes = archGguf(name);
        const gguf::File f = gguf::File::parse(bytes);
        const model::OpenedArch o = model::openArch(f, t);
        ISO_CHECK(o.am != nullptr && o.am->family == family && o.cfg.n_layer == static_cast<std::uint32_t>(layers));
    }
}

void testUnsupported() {
    const model::ArchTable t(kTable);
    for (const char* name : {"llama", ""}) {  // an unknown name, and a file without general.architecture
        const auto bytes = archGguf(name);
        const gguf::File f = gguf::File::parse(bytes);
        bool thrown = false;
        try {
            (void)model::openArch(f, t);
        } catch (const qwen35::ModelError& e) {
            thrown = e.code() == "UnsupportedArch" &&
                     std::string(e.what()).find("(WHIRL supports famA, famAmoe and famB only)") != std::string::npos &&
                     std::string(e.what()).find(std::string("general.architecture = '") + name + "'") != std::string::npos;
        }
        ISO_CHECK(thrown);
    }
}

}  // namespace

int main(int argc, char** argv) { return iso::run("model.arch_open", {{"known_name", testKnownName}, {"unsupported", testUnsupported}}, argc, argv); }
