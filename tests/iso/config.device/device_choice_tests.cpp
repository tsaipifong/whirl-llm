// whirl-iso-config-device-tests: the header-only module config.device (src/config/device_choice.h, the GPU choice of both front ends) built with its
// declared dependencies (config.settings: run_settings.cpp, platform env, the gpu_select types) and the link double of the model layer's
// restrictDevice / pickDevice (tests/iso/config.device/fake_device_pick.cpp) (ARCH-1 8.5 isolation, arch025-iso4). No test of chooseDevice
// existed. The checks pin the order (restrict, the note, then pick), the shown index, the restriction note and the WHIRL_HIP_DEVICE override.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "config/device_choice.h"
#include "iso/config.device/fake_device_pick.h"

#include <cstdlib>
#include <string>
#include <vector>

namespace {

namespace st = whirl::settings;

void setHipDevice(const char* v) { _putenv_s("WHIRL_HIP_DEVICE", v ? v : ""); }  // "" removes it

whirl::gpusel::GpuRecord rec(int index) {
    whirl::gpusel::GpuRecord r;
    r.index = index;
    return r;
}

void testRestrictThenPick() {
    setHipDevice(nullptr);
    iso_fake::DevicePick& f = iso_fake::devicePick();
    f = {};
    f.restriction.active = true;  // the runtime now sees only the chosen GPU: device 0, listed as index 7
    // arch025-iso4-fix (REVIEW 5d): a record's index differs from its position, so shown must be all[selected[0]].index
    // (device_choice.h:40), not the position selected[0] (= 1) and not the picked device (= 0)
    f.restriction.all = {rec(5), rec(7)};
    f.restriction.selected = {1};
    f.restriction.note = "device: using the 8060S only";
    f.picked = 0;
    std::vector<std::string> notes;
    const st::DeviceChoice c = st::chooseDevice("8060s", {}, [&](const std::string& n) {
        notes.push_back(n);
        f.calls.push_back("note:" + n);  // one order for the note and the two model calls (device_choice.h:6-7)
    });
    ISO_CHECK(c.dev == 0 && c.shown == 7);  // shown = the list index before the restriction
    ISO_CHECK(c.note == "device: using the 8060S only" && notes.size() == 1 && notes[0] == c.note);
    // the server's order: restrict, the restriction's note, then pick
    ISO_CHECK(f.calls.size() == 3 && f.calls[0] == "restrict:8060s" && f.calls[1] == "note:device: using the 8060S only" &&
              f.calls[2] == "pick:8060s");
}

void testNoRestriction() {
    setHipDevice(nullptr);
    iso_fake::DevicePick& f = iso_fake::devicePick();
    f = {};
    f.restriction.active = false;  // one GPU, or the user set a visibility variable
    f.picked = 3;
    int notes = 0;
    const st::DeviceChoice c = st::chooseDevice("", {}, [&](const std::string&) { ++notes; });
    ISO_CHECK(c.dev == 3 && c.shown == 3 && c.note.empty() && notes == 0);  // an empty note is not logged
    const st::DeviceChoice d = st::chooseDevice("", {}, {});  // no note callback: still fine
    ISO_CHECK(d.dev == 3);
}

void testHipDeviceOverride() {
    iso_fake::DevicePick& f = iso_fake::devicePick();
    f = {};
    f.picked = 9;
    setHipDevice("2");
    std::vector<std::string> warns;
    st::DeviceChoice c = st::chooseDevice("r9700", [&](const std::string& w) { warns.push_back(w); }, {});
    ISO_CHECK(c.dev == 2 && c.shown == 2 && f.calls.empty() && warns.empty());  // the debug override skips both model calls
    setHipDevice("abc");  // a bad value warns and uses 0
    c = st::chooseDevice("r9700", [&](const std::string& w) { warns.push_back(w); }, {});
    ISO_CHECK(c.dev == 0 && c.shown == 0 && warns.size() == 1 && f.calls.empty());
    setHipDevice(nullptr);
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("config.device",
                    {{"restrict_then_pick", testRestrictThenPick}, {"no_restriction", testNoRestriction}, {"hip_device_override", testHipDeviceOverride}},
                    argc, argv);
}
