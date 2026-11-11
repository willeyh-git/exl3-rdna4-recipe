"""Measure prefill and decode speed at several prompt lengths through a running TabbyAPI.

Usage: python bench/moe_context_bench.py LABEL [--contexts 2 8 16] [--repeats 3]
Appends one JSON record per request to logs/moe_context_bench.jsonl. Set MODEL_DIR to the model being served.
"""
import argparse
from datetime import datetime, timezone
import json
import os
import random
import sys
import time
import urllib.request

from tokenizers import Tokenizer

from common import BASE_URL, TOKENIZER, append_jsonl, auth_headers, log_path

WORDS = "the river city market old north stone bridge winter garden light road quiet harbor field tower".split()
INSTRUCTION = (
    "\n\nWrite a detailed explanation of how to implement an LRU cache with get, put, and delete operations. "
    "Include type hints, docstrings, and tests."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="label to group these measurements in the output")
    parser.add_argument("--contexts", type=int, nargs="+", default=[2, 8, 16], metavar="K",
                        help="prompt size targets in thousands of tokens (default: 2 8 16)")
    parser.add_argument("--repeats", type=int, default=3, help="requests per context (default: 3)")
    parser.add_argument("--max-tokens", type=int, default=128, help="maximum generated tokens (default: 128)")
    parser.add_argument("--url", default=BASE_URL, help="OpenAI-compatible API base URL (default: %(default)s)")
    parser.add_argument("--api-key", help="API key (defaults to TABBY_API_KEY or the local TabbyAPI key)")
    parser.add_argument("--out", help="JSONL output path (default: logs/moe_context_bench.jsonl)")
    args = parser.parse_args()
    if args.repeats < 1 or args.max_tokens < 1 or any(k < 1 for k in args.contexts):
        parser.error("contexts, repeats, and max-tokens must be positive")
    return args


def encode_len(tokenizer, text):
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def build_prompt(tokenizer, target_tokens, seed):
    """Build fresh, low-semantic-load text close to the target, leaving room for template/output tokens."""
    budget = max(1, target_tokens - 512)
    instruction_tokens = encode_len(tokenizer, INSTRUCTION)
    sample = "Section 0: " + " ".join(WORDS[i % len(WORDS)] for i in range(20)) + "."
    per_section = max(1, encode_len(tokenizer, sample))
    count = max(1, (budget - instruction_tokens) // per_section)
    rnd = random.Random(seed)

    def section(index):
        return f"Section {index}: " + " ".join(rnd.choice(WORDS) for _ in range(20)) + "."

    sections = [section(i) for i in range(count)]
    prompt = "\n".join(sections) + INSTRUCTION
    prompt_tokens = encode_len(tokenizer, prompt)
    while prompt_tokens > budget and len(sections) > 1:
        sections.pop()
        prompt = "\n".join(sections) + INSTRUCTION
        prompt_tokens = encode_len(tokenizer, prompt)
    while prompt_tokens < budget:
        sections.append(section(len(sections)))
        prompt = "\n".join(sections) + INSTRUCTION
        updated_count = encode_len(tokenizer, prompt)
        if updated_count > budget:
            sections.pop()
            prompt = "\n".join(sections) + INSTRUCTION
            prompt_tokens = encode_len(tokenizer, prompt)
            break
        prompt_tokens = updated_count
    return prompt, prompt_tokens


def timed_request(base_url, headers, prompt, max_tokens, timeout):
    body = {
        "model": "x",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", json.dumps(body).encode(), headers)
    started = time.perf_counter()
    first_token_at = None
    output = ""
    finish_reason = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            event = json.loads(line[5:])
            if not event.get("choices"):
                continue
            choice = event["choices"][0]
            delta = choice.get("delta", {})
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if piece:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                output += piece
            finish_reason = choice.get("finish_reason") or finish_reason
    ended = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("server returned no generated text")
    return output, {
        "ttft_s": first_token_at - started,
        "decode_s": ended - first_token_at,
        "finish_reason": finish_reason,
    }


def main():
    args = parse_args()
    tokenizer_path = os.path.abspath(TOKENIZER)
    if not os.path.isfile(tokenizer_path):
        sys.exit(f"Tokenizer not found: {tokenizer_path}; set MODEL_DIR to the model directory being benchmarked")
    tokenizer = Tokenizer.from_file(tokenizer_path)
    headers = auth_headers(args.api_key)
    output_path = args.out or log_path("moe_context_bench.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    failed = False

    for context_k in args.contexts:
        for repeat in range(1, args.repeats + 1):
            target_tokens = context_k * 1024
            prompt, prompt_tokens = build_prompt(tokenizer, target_tokens, time.time_ns() + context_k * repeat)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "label": args.label,
                "context_k": context_k,
                "repeat": repeat,
                "target_prompt_tokens": target_tokens,
                "prompt_tokens": prompt_tokens,
                "max_tokens": args.max_tokens,
                "url": args.url.rstrip("/"),
            }
            try:
                text, timing = timed_request(args.url, headers, prompt, args.max_tokens, timeout=1800)
                generated_tokens = len(tokenizer.encode(text, add_special_tokens=False).ids)
                record.update(timing)
                record["ttft_s"] = round(timing["ttft_s"], 3)
                record["decode_s"] = round(timing["decode_s"], 3)
                record["generated_tokens"] = generated_tokens
                record["prefill_tps"] = (round(prompt_tokens / timing["ttft_s"])
                                          if timing["ttft_s"] > 0 else None)
                record["decode_tps"] = (round((generated_tokens - 1) / timing["decode_s"], 1)
                                         if generated_tokens > 1 and timing["decode_s"] > 0 else None)
            except Exception as error:
                record["error"] = str(error)[:400]
                failed = True
            append_jsonl(output_path, record)
            print(json.dumps(record), flush=True)

    print(f"Appended results to {output_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())