#!/usr/bin/env python3
"""
tool_bench.py — tool-calling bench for the GLM-5.3 apples-to-apples
comparison (EXL3 TensorFold vs NVFP4 vLLM).

First PROBES whether the endpoint supports the OpenAI tools API at all
(one request with "tools": [...]). Verdict recorded honestly:
  - "supported"  -> tool_calls emitted
  - "ignored"    -> HTTP 200 but the model never emits tool_calls
  - "unsupported"-> HTTP 4xx/5xx on the tools request
If unsupported, the remaining scenarios are skipped for that endpoint
(the OTHER endpoint is still benched).

10 scenarios (weather/lookup/calc style). Per scenario, three checks:
  1. name_ok   — a tool_call with the expected function name
  2. args_ok   — arguments are valid JSON and contain the expected
                 key/values (expected subset of actual; numbers compared
                 numerically, strings case-insensitively)
  3. followup_ok — a second turn with the canned tool result appended
                 (assistant tool_calls + role:"tool") yields a content
                 answer containing the expected fact

Sampling pinned: temperature 0.0, top_p 1.0, seed 42 (see eval_common.py).
Thinking ON for both; tool_calls scored from the message (content ignored
for checks 1-2; the NVFP4 serve uses glm47 tool-call parser +
--enable-auto-tool-choice per glm53-nvfp4-b12x-dcp2.yaml).

Usage:
  python3 tool_bench.py --preset ours|nvfp4 [--smoke] [--max-tokens N]
                        [--out results.json]
"""

import argparse
import json

from eval_common import ENDPOINTS, EndpointDown, chat, clean_content

TOOL_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather conditions for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string",
                                    "description": "City name, e.g. Paris"}},
            "required": ["city"],
        },
    },
}

SCENARIOS = [
    {
        "id": "t01",
        "tools": [TOOL_WEATHER],
        "user": "What's the weather in Paris right now? Use the get_weather tool.",
        "expect_name": "get_weather",
        "expect_args": {"city": "Paris"},
        "tool_result": "Current weather in Paris: 18°C, light rain, wind 12 km/h.",
        "followup_must_contain": ["18"],
    },
    {
        "id": "t02",
        "tools": [TOOL_WEATHER],
        "user": "Check the current weather in Tokyo for me with the get_weather tool.",
        "expect_name": "get_weather",
        "expect_args": {"city": "Tokyo"},
        "tool_result": "Current weather in Tokyo: 27°C, sunny, no wind.",
        "followup_must_contain": ["27"],
    },
    {
        "id": "t03",
        "tools": [{
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate an arithmetic expression and return the result.",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        }],
        "user": "What is 27 * 43? Use the calculator tool with the expression '27*43' — do not compute it yourself.",
        "expect_name": "calculator",
        "expect_args": {"expression": "27*43"},
        "tool_result": "1161",
        "followup_must_contain": ["1161"],
    },
    {
        "id": "t04",
        "tools": [{
            "type": "function",
            "function": {
                "name": "lookup_capital",
                "description": "Look up the capital city of a country.",
                "parameters": {
                    "type": "object",
                    "properties": {"country": {"type": "string"}},
                    "required": ["country"],
                },
            },
        }],
        "user": "What is the capital of France? Use the lookup_capital tool.",
        "expect_name": "lookup_capital",
        "expect_args": {"country": "France"},
        "tool_result": "The capital of France is Paris.",
        "followup_must_contain": ["Paris"],
    },
    {
        "id": "t05",
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_stock_price",
                "description": "Get the latest stock price for a ticker symbol.",
                "parameters": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                },
            },
        }],
        "user": "What's the latest price of Apple stock? Use the get_stock_price tool with ticker AAPL.",
        "expect_name": "get_stock_price",
        "expect_args": {"ticker": "AAPL"},
        "tool_result": "AAPL (Apple Inc.): $227.50, up 0.8% today.",
        "followup_must_contain": ["227"],
    },
    {
        "id": "t06",
        "tools": [{
            "type": "function",
            "function": {
                "name": "convert_currency",
                "description": "Convert an amount between currencies.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number"},
                        "from_currency": {"type": "string"},
                        "to_currency": {"type": "string"},
                    },
                    "required": ["amount", "from_currency", "to_currency"],
                },
            },
        }],
        "user": "Convert 250 US dollars to euros using the convert_currency tool.",
        "expect_name": "convert_currency",
        "expect_args": {"amount": 250, "from_currency": "USD", "to_currency": "EUR"},
        "tool_result": "250 USD = 229.50 EUR (rate 0.918).",
        "followup_must_contain": ["229"],
    },
    {
        "id": "t07",
        "tools": [{
            "type": "function",
            "function": {
                "name": "search_flights",
                "description": "Search for flights between two airports on a date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string"},
                        "destination": {"type": "string"},
                        "date": {"type": "string", "description": "YYYY-MM-DD"},
                    },
                    "required": ["origin", "destination", "date"],
                },
            },
        }],
        "user": "Find flights from JFK to LAX on 2026-11-01 using the search_flights tool.",
        "expect_name": "search_flights",
        "expect_args": {"origin": "JFK", "destination": "LAX", "date": "2026-11-01"},
        "tool_result": "3 flights found JFK->LAX on 2026-11-01; cheapest $312, nonstop 5h40m.",
        "followup_must_contain": ["312"],
    },
    {
        "id": "t08",
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get the current local time in an IANA timezone.",
                "parameters": {
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}},
                    "required": ["timezone"],
                },
            },
        }],
        "user": "What time is it in Berlin right now? Use the get_current_time tool with timezone Europe/Berlin.",
        "expect_name": "get_current_time",
        "expect_args": {"timezone": "Europe/Berlin"},
        "tool_result": "The current time in Europe/Berlin is 14:35 (Tuesday).",
        "followup_must_contain": ["14:35"],
    },
    {
        "id": "t09",
        "tools": [{
            "type": "function",
            "function": {
                "name": "calculate_bmi",
                "description": "Compute BMI from weight in kg and height in meters.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "weight_kg": {"type": "number"},
                        "height_m": {"type": "number"},
                    },
                    "required": ["weight_kg", "height_m"],
                },
            },
        }],
        "user": "Compute the BMI of someone who weighs 70 kg and is 1.75 m tall, using the calculate_bmi tool.",
        "expect_name": "calculate_bmi",
        "expect_args": {"weight_kg": 70, "height_m": 1.75},
        "tool_result": "BMI: 22.86 — normal weight range.",
        "followup_must_contain": ["22.8"],
    },
    {
        "id": "t10",
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_movie_release_year",
                "description": "Look up the release year of a movie by title.",
                "parameters": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                },
            },
        }],
        "user": "When was the movie Inception released? Use the get_movie_release_year tool.",
        "expect_name": "get_movie_release_year",
        "expect_args": {"title": "Inception"},
        "tool_result": "Inception was released in 2010, directed by Christopher Nolan.",
        "followup_must_contain": ["2010"],
    },
]

SMOKE_IDS = {"t01", "t03"}


def _norm(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    return str(v).strip().lower()


def check_args(args_str, expected):
    try:
        actual = json.loads(args_str) if args_str and args_str.strip() else {}
    except json.JSONDecodeError as e:
        return False, "invalid JSON: %s" % e
    if not isinstance(actual, dict):
        return False, "arguments not an object"
    for k, v in expected.items():
        if k not in actual:
            return False, "missing key %r" % k
        if _norm(actual[k]) != _norm(v):
            return False, "key %r: got %r want %r" % (k, actual[k], v)
    return True, "args match"


def probe_tools(ep, max_tokens, timeout):
    """One tools request. Returns verdict string ('supported' / 'ignored' /
    'unsupported: ...') and the raw chat result or None."""
    s = SCENARIOS[0]
    try:
        r = chat(ep, [{"role": "user", "content": s["user"]}],
                 max_tokens=max_tokens, tools=s["tools"], timeout=timeout)
    except EndpointDown as e:
        return "endpoint-down: %s" % e, None
    except RuntimeError as e:
        return "unsupported: %s" % e, None
    if r["tool_calls"]:
        return "supported", r
    return "ignored (HTTP 200, no tool_calls emitted)", r


def run(ep, smoke=False, max_tokens=2500, timeout=900):
    out = {"endpoint": ep["name"], "model": ep["model"], "smoke": smoke,
           "sampling": {"temperature": 0.0, "top_p": 1.0, "seed": 42}}
    verdict, first = probe_tools(ep, max_tokens, timeout)
    out["tools_api_verdict"] = verdict
    print("  [%s] tools probe: %s" % (ep["name"].split()[0], verdict), flush=True)
    if verdict != "supported":
        out["scenarios"] = []
        out["summary"] = {"probe": verdict,
                          "skipped": "tools request did not succeed (%s)" % verdict}
        return out

    scen = [s for s in SCENARIOS if not smoke or s["id"] in SMOKE_IDS]
    results = []
    for s in scen:
        rec = {"id": s["id"], "name_ok": False, "args_ok": False,
               "followup_ok": False}
        try:
            r = first if s["id"] == SCENARIOS[0]["id"] and first else chat(
                ep, [{"role": "user", "content": s["user"]}],
                max_tokens=max_tokens, tools=s["tools"], timeout=timeout)
        except (EndpointDown, RuntimeError) as e:
            rec["error"] = str(e)
            results.append(rec)
            continue
        tcs = r["tool_calls"]
        tc = next((t for t in tcs if t.get("name") == s["expect_name"]), None)
        if tc is None and tcs:
            tc = tcs[0]
            rec["note"] = "called %r instead" % tcs[0].get("name")
        if tc:
            rec["name_ok"] = tc.get("name") == s["expect_name"]
            rec["called"] = tc.get("name")
            rec["arguments"] = tc.get("arguments", "")[:200]
            ok, why = check_args(tc.get("arguments", ""), s["expect_args"])
            rec["args_ok"] = ok
            rec["args_verdict"] = why
            if ok:
                # follow-up turn with the tool result
                msgs = [
                    {"role": "user", "content": s["user"]},
                    {"role": "assistant", "content": r["content"] or None,
                     "tool_calls": [
                         {"id": t["id"] or "call_0", "type": "function",
                          "function": {"name": t["name"],
                                       "arguments": t["arguments"]}}
                         for t in tcs]},
                    {"role": "tool", "tool_call_id": (tc["id"] or "call_0"),
                     "content": s["tool_result"]},
                ]
                msgs[1] = {k: v for k, v in msgs[1].items() if v is not None}
                try:
                    r2 = chat(ep, msgs, max_tokens=max_tokens,
                              tools=s["tools"], timeout=timeout)
                    text = clean_content(r2["content"]).lower()
                    rec["followup_ok"] = all(
                        m.lower() in text for m in s["followup_must_contain"])
                    rec["followup_content_head"] = text[:160]
                    if not rec["followup_ok"]:
                        rec["followup_verdict"] = (
                            "missing %s in %r" % (
                                s["followup_must_contain"], text[:100]))
                except (EndpointDown, RuntimeError) as e:
                    rec["followup_verdict"] = "follow-up request failed: %s" % e
        results.append(rec)
        print("  [%s] %s name=%s args=%s followup=%s" % (
            ep["name"].split()[0], s["id"],
            rec["name_ok"], rec["args_ok"], rec["followup_ok"]), flush=True)
    out["scenarios"] = results
    n = len(results)
    out["summary"] = {
        "probe": "supported",
        "scenarios": n,
        "name_ok": sum(r["name_ok"] for r in results),
        "args_ok": sum(r["args_ok"] for r in results),
        "followup_ok": sum(r["followup_ok"] for r in results),
        "full_pass": sum(r["name_ok"] and r["args_ok"] and r["followup_ok"]
                         for r in results),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(ENDPOINTS), default=None)
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--api-key")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=2500)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.preset and not (args.base_url and args.model):
        ap.error("need --preset or --base-url + --model")
    ep = dict(ENDPOINTS[args.preset]) if args.preset else {
        "name": "custom", "base_url": args.base_url, "model": args.model,
        "api_key": args.api_key}
    if args.base_url:
        ep["base_url"] = args.base_url
    if args.model:
        ep["model"] = args.model
    if args.api_key:
        ep["api_key"] = args.api_key
    res = run(ep, smoke=args.smoke, max_tokens=args.max_tokens,
              timeout=args.timeout)
    print(json.dumps(res["summary"], indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
