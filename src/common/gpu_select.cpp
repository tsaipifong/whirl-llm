// GPU selection (see include/whirl/gpu_select.h).
// SPDX-License-Identifier: Apache-2.0
//
// STUB for the failing-test commit (s8-719 root): declared seam only; the behaviour before this fix
// had no list parsing, no restriction and no identity check.

#include "whirl/gpu_select.h"

namespace whirl::gpusel {

std::vector<std::string> parseDeviceList(std::string_view spec) {
    if (spec.empty()) return {};
    return {std::string(spec)};
}
std::vector<int> selectGpus(const std::vector<GpuRecord>&, std::string_view, std::uint64_t) { return {}; }
int autoPick(const std::vector<GpuRecord>&, std::uint64_t) { return -1; }
std::string listText(const std::vector<GpuRecord>&) { return {}; }
std::string visibleDevicesValue(const std::vector<int>&) { return {}; }
std::string userVisibilityVar(const std::function<const char*(const char*)>&) { return {}; }
bool sameGpu(const GpuRecord&, const GpuRecord&) { return true; }
std::string formatRecord(const GpuRecord&) { return {}; }
std::vector<GpuRecord> parseRecords(std::string_view) { return {}; }
std::string restrictionNote(const std::vector<GpuRecord>&, const std::vector<int>&) { return {}; }

}  // namespace whirl::gpusel

namespace whirl::gpusel {
std::string pciText(const GpuRecord&) { return {}; }
std::string lockName(const GpuRecord& r) { return "whirl-gpu-" + std::to_string(r.index); }  // before: by index
std::string listingText(const std::vector<GpuRecord>&, int) { return {}; }
std::string userVarNote(const std::string&, const std::string&) { return {}; }
}  // namespace whirl::gpusel
namespace whirl::gpusel { std::string pciLongText(const GpuRecord&) { return {}; } }
namespace whirl::gpusel {
std::string pciId(const std::vector<GpuRecord>&, const GpuRecord&) { return {}; }
std::string pciIdLong(const std::vector<GpuRecord>&, const GpuRecord&) { return {}; }
}  // namespace whirl::gpusel
namespace whirl::gpusel {
std::uint64_t modelNeedMib(std::uint64_t, std::uint64_t) { return 0; }
std::string autoReason(const std::vector<GpuRecord>&, int, std::uint64_t) { return {}; }
}  // namespace whirl::gpusel
