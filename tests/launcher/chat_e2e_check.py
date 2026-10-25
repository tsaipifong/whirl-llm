# SPDX-License-Identifier: Apache-2.0
# feat026 launcher: end-to-end check of the tray's chat window (`whirl.exe --whirl-launcher-chat HOST:PORT`) against a
# mock OpenAI server on 127.0.0.1 (no GPU, no model). Checks: the request body (messages, stream), the streamed answer
# printed, the history sent with the next message, /new, /exit, an HTTP error, UTF-8 (Chinese) both ways.
#   python -I tests\launcher\chat_e2e_check.py <path to whirl.exe>
# Exit 0 = pass.
import http.server
import json
import subprocess
import sys
import threading

ANSWERS = ["Hello", " 你好!"]  # streamed in two chunks
requests = []


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n).decode("utf-8"))
        requests.append(body)
        last = body["messages"][-1]["content"]
        if last == "fail please":
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"mock failure"}}')
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
        for a in ANSWERS:
            ev = json.dumps({"choices": [{"delta": {"content": a}}]}, ensure_ascii=False)
            self.wfile.write(("data: " + ev + "\n\n").encode("utf-8"))
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


def main():
    exe = sys.argv[1]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    q1 = "寫一個 Python 函式"  # 寫一個 Python 函式
    stdin = "\n".join([q1, "second question", "fail please", "/new", "third", "/exit", "never sent"]) + "\n"
    p = subprocess.run([exe, "--whirl-launcher-chat", "127.0.0.1:%d" % port], input=stdin.encode("utf-8"),
                       capture_output=True, timeout=60)
    srv.shutdown()
    out = p.stdout.decode("utf-8", "replace")
    bad = []

    def check(cond, what):
        print(("ok   " if cond else "FAIL ") + what)
        if not cond:
            bad.append(what)

    full = "".join(ANSWERS)
    check(p.returncode == 0, "exit code 0 (got %d)" % p.returncode)
    check(len(requests) == 4, "4 requests (got %d; 'never sent' after /exit must not go out)" % len(requests))
    if len(requests) == 4:
        r1, r2, r3, r4 = requests
        check(r1.get("stream") is True, "stream: true")
        check(r1["messages"] == [{"role": "user", "content": q1}], "first request = the Chinese question only")
        check([m["role"] for m in r2["messages"]] == ["user", "assistant", "user"], "second request carries the history")
        check(r2["messages"][1]["content"] == full, "history holds the streamed answer")
        check(r3["messages"][-1]["content"] == "fail please", "third request sent")
        check([m["content"] for m in r4["messages"]] == ["third"], "/new cleared the history (failed turn not kept)")
    check(out.count(full) == 3, "answer printed 3 times (got %d)" % out.count(full))
    check("HTTP 500" in out, "the HTTP error is shown")
    check("(new conversation)" in out, "/new acknowledged")
    if bad:
        print("--- stdout ---\n" + out + "\n--- stderr ---\n" + p.stderr.decode("utf-8", "replace"))
        print("chat_e2e_check: %d check(s) FAILED" % len(bad))
        return 1
    print("chat_e2e_check: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
