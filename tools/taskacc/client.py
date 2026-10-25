"""Minimal OpenAI-style chat client for whirl-server (stdlib only; feat027-taskacc)."""
import json
import time
import urllib.error
import urllib.request


class Client:
    def __init__(self, url, timeout=900.0):
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))

    def models(self):
        return [m.get("id") for m in self._get("/v1/models").get("data", [])]

    def props(self):
        try:
            return self._get("/props")
        except Exception as e:  # llama.cpp-style endpoint; optional
            return {"error": str(e)}

    def health(self):
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=5) as r:
                return r.status == 200
        except Exception:
            return False

    def chat(self, model, prompt, temperature=0.0, max_tokens=2048, seed=None, think=False, effort=None, retries=2):
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
                "stream": False, "chat_template_kwargs": {"enable_thinking": bool(think)}}
        if not think:
            body["enable_thinking"] = False
        if temperature != "card":
            body["temperature"] = float(temperature)
        if seed is not None:
            body["seed"] = int(seed)
        if effort and think:
            body["chat_template_kwargs"]["reasoning_effort"] = effort
            body["reasoning_effort"] = effort
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        last = None
        for attempt in range(retries + 1):
            t0 = time.time()
            try:
                req = urllib.request.Request(self.url + "/v1/chat/completions", data=data,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    j = json.loads(r.read().decode("utf-8"))
                ch = (j.get("choices") or [{}])[0]
                msg = ch.get("message") or {}
                return {"ok": True, "content": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
                        "finish": ch.get("finish_reason"), "usage": j.get("usage") or {},
                        "seconds": round(time.time() - t0, 3), "attempts": attempt + 1}
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, ValueError) as e:
                last = f"{type(e).__name__}: {e}"
                if isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500:
                    break
                time.sleep(2 + 3 * attempt)
        return {"ok": False, "error": last, "content": None, "reasoning": "", "finish": None, "usage": {},
                "seconds": None, "attempts": retries + 1}
