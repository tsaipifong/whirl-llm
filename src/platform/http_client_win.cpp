// Platform layer, Windows: a minimal HTTP client (WinHTTP) for the launcher's chat window. See http_client.h.
// SPDX-License-Identifier: Apache-2.0

#include "platform/http_client.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <winhttp.h>

namespace whirl::platform::http_client {

namespace {

std::wstring widen(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<std::size_t>(n > 0 ? n : 0), L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

struct Handle {
    HINTERNET h = nullptr;
    explicit Handle(HINTERNET x) : h(x) {}
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    ~Handle() {
        if (h) WinHttpCloseHandle(h);
    }
};

std::string err(const char* what) { return std::string(what) + " failed (Windows error " + std::to_string(GetLastError()) + ")"; }

Result request(const wchar_t* verb, const std::string& host, std::uint16_t port, const std::string& path, const std::string* body,
               unsigned timeout_ms, const std::function<bool(const std::string&)>* on_data) {
    Result r;
    // no proxy: the server is on this computer
    Handle ses(WinHttpOpen(L"whirl-launcher", WINHTTP_ACCESS_TYPE_NO_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0));
    if (!ses.h) {
        r.why = err("WinHttpOpen");
        return r;
    }
    const int t = static_cast<int>(timeout_ms);
    WinHttpSetTimeouts(ses.h, t, t, t, t);
    Handle con(WinHttpConnect(ses.h, widen(host).c_str(), port, 0));
    if (!con.h) {
        r.why = err("WinHttpConnect");
        return r;
    }
    Handle req(WinHttpOpenRequest(con.h, verb, widen(path).c_str(), nullptr, WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, 0));
    if (!req.h) {
        r.why = err("WinHttpOpenRequest");
        return r;
    }
    const wchar_t* hdr = body ? L"Content-Type: application/json\r\n" : WINHTTP_NO_ADDITIONAL_HEADERS;
    const DWORD hdr_len = body ? static_cast<DWORD>(-1L) : 0;
    void* data = body ? const_cast<char*>(body->data()) : WINHTTP_NO_REQUEST_DATA;
    const DWORD len = body ? static_cast<DWORD>(body->size()) : 0;
    if (!WinHttpSendRequest(req.h, hdr, hdr_len, data, len, len, 0)) {
        r.why = err("connecting to the server");
        return r;
    }
    if (!WinHttpReceiveResponse(req.h, nullptr)) {
        r.why = err("waiting for the answer");
        return r;
    }
    DWORD status = 0, sz = sizeof(status);
    WinHttpQueryHeaders(req.h, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER, WINHTTP_HEADER_NAME_BY_INDEX, &status, &sz,
                        WINHTTP_NO_HEADER_INDEX);
    r.ok = true;
    r.status = static_cast<int>(status);
    for (;;) {
        DWORD avail = 0;
        if (!WinHttpQueryDataAvailable(req.h, &avail)) {
            r.why = err("reading the answer");
            break;
        }
        if (avail == 0) break;
        std::string chunk(avail, '\0');
        DWORD got = 0;
        if (!WinHttpReadData(req.h, chunk.data(), avail, &got)) {
            r.why = err("reading the answer");
            break;
        }
        chunk.resize(got);
        if (on_data && *on_data) {
            if (!(*on_data)(chunk)) break;
        } else {
            r.body += chunk;
        }
    }
    return r;
}

}  // namespace

Result get(const std::string& host, std::uint16_t port, const std::string& path, unsigned timeout_ms) {
    return request(L"GET", host, port, path, nullptr, timeout_ms, nullptr);
}

Result postJson(const std::string& host, std::uint16_t port, const std::string& path, const std::string& body, unsigned timeout_ms,
                const std::function<bool(const std::string&)>& on_data) {
    return request(L"POST", host, port, path, &body, timeout_ms, &on_data);
}

}  // namespace whirl::platform::http_client
