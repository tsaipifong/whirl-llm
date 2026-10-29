// whirl-iso-platform-net-tests: the platform socket layer (src/platform/net_win.cpp, net.h, net_fake.h) built with nothing but its own
// headers and ws2_32 (ARCH-1 8.5 isolation, arch025-iso4). No unit test file existed (the server tests drive it through the HTTP server).
// Every socket here is on 127.0.0.1 and closed before the case ends. The checks pin the error-code constants, a loopback round trip through
// the wrappers, the socket options and the FakeNet recording / injection that the server tests rely on.
// SPDX-License-Identifier: Apache-2.0
#include "iso/iso_test.h"
#include "platform/net.h"
#include "platform/net_fake.h"

#include <cstring>
#include <string>

namespace {

namespace net = whirl::platform::net;

// a listening loopback socket on an ephemeral port
struct Listener {
    net::Socket s = net::kInvalid;
    std::uint16_t port = 0;
    bool ok = false;
    Listener() {
        s = net::openTcp4();
        if (s == net::kInvalid) return;
        if (net::bindLoopback4(s, 0) != 0 || net::listen(s, 4) != 0 || net::localPort(s, &port) != 0 || port == 0) return;
        ok = true;
    }
    ~Listener() {
        if (s != net::kInvalid) net::close(s);
    }
};

net::Socket connectTo(std::uint16_t port) {
    const net::Addr* a = net::resolve("127.0.0.1", std::to_string(port).c_str(), net::Lookup::client_v4);
    if (!a) return net::kInvalid;
    net::Socket c = net::openTcp(a);
    if (c != net::kInvalid && net::connect(c, a) != 0) {
        net::close(c);
        c = net::kInvalid;
    }
    net::freeAddr(a);
    return c;
}

void testConstants() {
    ISO_CHECK(net::kInvalid == ~net::Socket(0) && net::kError == -1);
    ISO_CHECK(net::kErrAccess == 10013 && net::kErrWouldBlock == 10035 && net::kErrAddrInUse == 10048 && net::kErrAddrNotAvail == 10049);
    ISO_CHECK(net::kErrConnReset == 10054 && net::kErrNoBufs == 10055 && net::kErrTimedOut == 10060 && net::kErrConnRefused == 10061);
    ISO_CHECK(net::kMaxBacklog == 0x7fffffff && net::kPeek == 2);
    ISO_CHECK(static_cast<int>(net::Shut::send) == 1 && static_cast<int>(net::Shut::both) == 2);
}

void testLoopbackRoundTrip() {
    ISO_CHECK(net::startup());
    {
        Listener l;
        ISO_CHECK(l.ok);
        if (l.ok) {
            net::Socket c = connectTo(l.port);
            ISO_CHECK(c != net::kInvalid);
            net::Socket a = net::accept(l.s);
            ISO_CHECK(a != net::kInvalid);
            if (c != net::kInvalid && a != net::kInvalid) {
                ISO_CHECK(net::setRecvTimeout(a, 2000) == 0 && net::setSendTimeout(a, 2000) == 0);
                ISO_CHECK(net::setSendBuf(a, 65536) == 0 && net::setRecvBuf(a, 65536) == 0);
                ISO_CHECK(net::pollReadable(a) == 0);  // nothing sent yet
                const char msg[] = "GET / HTTP/1.1\r\n\r\n";
                ISO_CHECK(net::send(c, msg, static_cast<int>(sizeof msg - 1)) == static_cast<int>(sizeof msg - 1));
                char buf[64] = {};
                ISO_CHECK(net::recv(a, buf, 3, net::kPeek) == 3 && std::memcmp(buf, "GET", 3) == 0);  // peek leaves the bytes
                ISO_CHECK(net::pollReadable(a) > 0);
                ISO_CHECK(net::recv(a, buf, static_cast<int>(sizeof buf)) == static_cast<int>(sizeof msg - 1));
                ISO_CHECK(net::socketError(a) == 0);
                ISO_CHECK(net::shutdown(c, net::Shut::send) == 0);
                ISO_CHECK(net::recv(a, buf, static_cast<int>(sizeof buf)) == 0);  // orderly close seen as 0
            }
            if (a != net::kInvalid) net::close(a);
            if (c != net::kInvalid) net::close(c);
        }
    }
    net::cleanup();
}

void testNonBlockingAndErrors() {
    ISO_CHECK(net::startup());
    {
        Listener l;
        ISO_CHECK(l.ok);
        if (l.ok) {
            ISO_CHECK(net::setNonBlocking(l.s, true) == 0);
            ISO_CHECK(net::accept(l.s) == net::kInvalid);  // nothing waiting
            ISO_CHECK(net::lastError() == net::kErrWouldBlock);
            net::setLastError(0);
            ISO_CHECK(net::lastError() == 0);
        }
        // a second exclusive bind of the same loopback port is refused
        net::Socket dup = net::openTcp4();
        if (dup != net::kInvalid && l.ok) {
            ISO_CHECK(net::setExclusiveAddrUse(dup) == 0);
            ISO_CHECK(net::bindLoopback4(dup, l.port) == net::kError);
            ISO_CHECK(net::lastError() == net::kErrAddrInUse || net::lastError() == net::kErrAccess);
        }
        if (dup != net::kInvalid) net::close(dup);
    }
    net::cleanup();
}

void testProbeListen() {
    // a free port: done, err 0. The probe binds the number it is given, so take a port a listener just released.
    std::uint16_t port = 0;
    ISO_CHECK(net::startup());
    {
        Listener l;
        port = l.port;
    }
    net::cleanup();
    ISO_CHECK(port != 0);
    const net::ProbeResult free_r = net::probeListen("127.0.0.1", std::to_string(port).c_str());
    ISO_CHECK(free_r.stage == net::ProbeResult::Stage::done && free_r.err == 0);
    const net::ProbeResult bad = net::probeListen("not a numeric host", "80");
    ISO_CHECK(bad.stage == net::ProbeResult::Stage::bad_host);
}

void testFakeNet() {
    namespace fake = whirl::platform::net::fake;
    ISO_CHECK(net::startup());
    fake::begin();
    {
        const net::Addr* a = net::resolve("127.0.0.1", "0", net::Lookup::passive_numeric);  // a passive socket is tracked
        ISO_CHECK(a != nullptr);
        net::Socket s = a ? net::open(a) : net::kInvalid;  // open(): the socket of a passive lookup is the tracked one
        ISO_CHECK(s != net::kInvalid);
        if (s != net::kInvalid) {
            ISO_CHECK(net::setExclusiveAddrUse(s) == 0);
            ISO_CHECK(net::bind(s, a) == 0);
            ISO_CHECK(net::listen(s, 1) == 0);
            fake::failAccept(1, net::kErrNoBufs);
            ISO_CHECK(fake::injectedLeft() == 1);
            ISO_CHECK(net::accept(s) == net::kInvalid && net::lastError() == net::kErrNoBufs);  // injected, no Winsock call
            ISO_CHECK(fake::injectedLeft() == 0);
            net::close(s);
        }
        if (a) net::freeAddr(a);
    }
    const std::vector<fake::Call> calls = fake::end();
    ISO_CHECK(!calls.empty());
    bool saw_listen = false, saw_opt = false;
    for (const fake::Call& c : calls) {
        if (c.tag == "L1" && c.text.rfind("listen(", 0) == 0) saw_listen = true;
        if (c.tag == "L1" && c.text.rfind("setsockopt(", 0) == 0) saw_opt = true;
    }
    ISO_CHECK(saw_listen && saw_opt);
    ISO_CHECK(fake::end().empty());  // off again: nothing is recorded
    net::cleanup();
}

}  // namespace

int main(int argc, char** argv) {
    return iso::run("platform.net",
                    {{"constants", testConstants},
                     {"loopback_round_trip", testLoopbackRoundTrip},
                     {"nonblocking_errors", testNonBlockingAndErrors},
                     {"probe_listen", testProbeListen},
                     {"fake_net", testFakeNet}},
                    argc, argv);
}
