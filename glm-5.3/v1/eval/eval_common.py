#!/usr/bin/env python3
"""
eval_common.py — shared client + config for the GLM-5.3 apples-to-apples
intelligence + tool-calling harness (EXL3 TensorFold vs NVFP4 vLLM).

MIRRORED CONFIG (apples-to-apples, "as in the NVFP4 yaml"):
  The live NVFP4 serve is recipes/8x-spark-cluster/glm53-nvfp4-b12x-dcp2.yaml
  (confirmed 2026-10-16: /v1/models -> id "GLM-5.3-NVFP4",
   root /root/models/models11/glm53-nvfp4-vision, max_model_len 450000,
   system_fingerprint vllm-0.30.1rc1.dev723+g6bbad6acd...-tp8). Its serve flags:
     --reasoning-parser glm45                          (thinking ON, separate field)
     --tool-call-parser glm47 --enable-auto-tool-choice
     --seed 42
     --default-chat-template-kwargs {"reasoning_effort":"high","clear_thinking":false}
     TP8 / DCP2 / fp8 KV / MTP nst3 / max_model_len 450000
  OURS: TensorFold EXL3 4.04bpw h8 TP8 serve on .111, model name "glm53-w8",
  thinking ON by default (enable_thinking; reasoning_content separate).

  Per-request sampling pinned IDENTICALLY on BOTH endpoints:
     temperature=0.0, top_p=1.0, seed=42
  Documented choice: the yaml pins --seed 42 but no temperature; we pin
  temperature 0 (greedy) per request on both endpoints for deterministic
  scoring runs. Thinking mode stays ON for both (each serve's own default;
  NVFP4 at its configured reasoning_effort=high).

  REASONING FIELD HANDLING: this vLLM build emits reasoning in the `reasoning`
  field (NOT `reasoning_content`); our TensorFold serve uses
  `reasoning_content`. Both are read. Scoring uses ONLY `content`; residual
  inline <think|thinking> blocks are stripped before scoring.

  ENDPOINTS (presets):
     ours  : http://127.0.0.1:8080/v1   model glm53-w8   (loopback on .111 —
             the harness must run ON .111 for this endpoint; no API key)
     nvfp4 : http://192.168.178.11:5001/v1  model GLM-5.3-NVFP4, API key
             required (HTTP only — never ssh to .11-.18)

stdlib only (urllib); runs on python3.12.
"""

import json
import os
import re
import time
import urllib.error
import urllib.request

ENDPOINTS = {
    "ours": {
        "name": "ours (TensorFold EXL3 TP8)",
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "glm53-w8",
        "api_key": None,
    },
    "nvfp4": {
        "name": "nvfp4 (vLLM NVFP4 TP8 DCP2)",
        "base_url": "http://192.168.178.11:5001/v1",
        "model": "GLM-5.3-NVFP4",
        "api_key": os.environ.get("BENCH_API_KEY", "EMPTY"),
    },
}

# Pinned per-request sampling — identical on both endpoints (see header).
SAMPLING = {"temperature": 0.0, "top_p": 1.0, "seed": 42}


class EndpointDown(Exception):
    """Connection-level failure (refused/timeout) — endpoint not reachable."""


def _post(url, payload, api_key, timeout):
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def _sse_payloads(resp):
    for raw in resp:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def _finish(result, t0, ttft, streamed, tokens_source):
    wall = time.time() - t0
    result["wall"] = wall
    result["ttft"] = ttft if ttft is not None else wall
    result["streamed"] = streamed
    result["tokens_source"] = tokens_source
    u = result.get("usage") or {}
    ct = u.get("completion_tokens")
    if ct is None:
        ct = result.pop("_approx_tokens", None)
    result["completion_tokens"] = ct
    rtd = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
    result["reasoning_tokens"] = rtd
    return result


def _stream_chat(url, payload, api_key, timeout):
    t0 = time.time()
    ttft = None
    content, reasoning = [], []
    tool_calls = {}
    usage = None
    finish = None
    approx = 0
    got_any = False
    with _post(url, payload, api_key, timeout) as resp:
        for obj in _sse_payloads(resp):
            got_any = True
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                d = ch.get("delta") or {}
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                c = d.get("content")
                r = d.get("reasoning_content") or d.get("reasoning")
                tcs = d.get("tool_calls")
                if (c or r or tcs) and ttft is None:
                    ttft = time.time() - t0
                if c:
                    content.append(c)
                    approx += 1
                if r:
                    reasoning.append(r)
                    approx += 1
                for tc in tcs or []:
                    idx = tc.get("index", 0)
                    slot = tool_calls.setdefault(
                        idx, {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
    if not got_any:
        raise ValueError("empty stream (no SSE data lines)")
    return _finish(
        {
            "content": "".join(content),
            "reasoning": "".join(reasoning),
            "tool_calls": [tool_calls[i] for i in sorted(tool_calls)],
            "finish_reason": finish,
            "usage": usage,
            "_approx_tokens": approx,
        },
        t0,
        ttft,
        True,
        "usage" if usage else "approx-chunks",
    )


def _plain_chat(url, payload, api_key, timeout):
    t0 = time.time()
    with _post(url, payload, api_key, timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    ch = (body.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    tcs = []
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        tcs.append(
            {
                "id": tc.get("id") or "",
                "name": fn.get("name") or "",
                "arguments": fn.get("arguments") or "",
            }
        )
    return _finish(
        {
            "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "",
            "tool_calls": tcs,
            "finish_reason": ch.get("finish_reason"),
            "usage": body.get("usage"),
        },
        t0,
        None,
        False,
        "usage",
    )


def chat(ep, messages, max_tokens=4000, tools=None, timeout=900, stream=True,
         sampling=None):
    """One chat completion. Returns dict with content/reasoning/tool_calls/
    usage/ttft/wall/completion_tokens. Raises EndpointDown on connection
    failure, RuntimeError (with server body) on persistent HTTP error."""
    sampling = dict(SAMPLING if sampling is None else sampling)
    payload = {
        "model": ep["model"],
        "messages": messages,
        "max_tokens": max_tokens,
        **sampling,
    }
    if tools:
        payload["tools"] = tools
    url = ep["base_url"].rstrip("/") + "/chat/completions"
    key = ep.get("api_key")
    errors = []
    if stream:
        for extra in (
            {"stream": True, "stream_options": {"include_usage": True}},
            {"stream": True},
        ):
            try:
                return _stream_chat(url, {**payload, **extra}, key, timeout)
            except urllib.error.HTTPError as e:
                try:
                    body = e.read().decode("utf-8", "replace")[:400]
                except Exception:
                    body = str(e)
                errors.append("stream %s: HTTP %s %s" % (list(extra), e.code, body))
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                raise EndpointDown(str(e))
            except ValueError as e:
                errors.append("stream %s: %s" % (list(extra), e))
    try:
        return _plain_chat(url, payload, key, timeout)
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", "replace")[:800]
        except Exception:
            body = str(e)
        raise RuntimeError("HTTP %s: %s | prior attempts: %s" % (e.code, body, errors))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise EndpointDown(str(e))


THINK_RE = re.compile(r"<(?:think|thinking|reasoning)>(.*?)(?:</(?:think|thinking|reasoning)>|\Z)", re.S | re.I)


def clean_content(text):
    """Scored text: strip residual inline think blocks, trim."""
    if not text:
        return ""
    if THINK_RE.search(text):
        text = THINK_RE.sub("", text)
    return text.strip()


def answer_text(content):
    """Text after the LAST 'Answer:' marker (models bold it, add units, etc.),
    else None. Never includes the reasoning field."""
    content = clean_content(content)
    ms = re.findall(r"[Aa]nswer\s*[:：]\s*(.+)", content)
    if not ms:
        return None
    t = ms[-1].strip().strip("*").strip()
    t = t.splitlines()[0].strip().strip("*").strip()
    return t[:80]
