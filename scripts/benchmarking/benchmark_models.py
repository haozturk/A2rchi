#!/usr/bin/env python3
"""
Benchmark script: runs questions from a CSV against Archi using multiple models.
Collects question, response, response_time, model_name, and gpu_spec.

Usage:
    python benchmark_models.py \
        --base-url https://your-archi-instance:7869 \
        --cookie "session=<your_flask_session_cookie>" \
        --input user_questions_with_feedback.csv \
        --output benchmark_results.csv \
        --gpu-spec "NVIDIA A100 80GB"

To get the session cookie:
    1. Log in to the Archi UI via browser (SSO)
    2. Open DevTools -> Application -> Cookies
    3. Copy the 'session' cookie value
"""

import argparse
import csv
import json
import time
import sys
import urllib3

import requests

# Some CSV fields contain very large pasted logs; raise the field size limit
csv.field_size_limit(sys.maxsize)

# Suppress InsecureRequestWarning if using --no-verify
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

MODELS = [
    "openai/openai/gpt-oss-120b",
    "MiniMaxAI/MiniMax-M2.7",
    "google/Gemma4",
]

PROVIDER = "local"

STREAM_ENDPOINT = "/api/get_chat_response_stream"


def extract_questions(csv_path: str) -> list:
    """Read questions from the user_questions_with_feedback.csv."""
    questions = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", "").strip()
            if q:
                questions.append({
                    "conversation_id": row.get("conversation_id", ""),
                    "question": q,
                    "feedback": row.get("feedback", ""),
                })
    return questions


def send_question(session: requests.Session, base_url: str, question: str, model: str, verify_ssl: bool) -> dict:
    """
    Send a question to Archi's streaming endpoint and collect the full response.
    Returns dict with 'response', 'response_time', 'error'.
    """
    url = f"{base_url.rstrip('/')}{STREAM_ENDPOINT}"

    payload = {
        "last_message": [["User", question]],
        "conversation_id": None,
        "config_name": None,
        "client_sent_msg_ts": int(time.time() * 1000),
        "client_timeout": 300000,
        "client_id": "benchmark_script",
        "include_agent_steps": False,
        "include_tool_steps": False,
        "provider": PROVIDER,
        "model": model,
    }

    start = time.perf_counter()
    try:
        resp = session.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=600,
            verify=verify_ssl,
        )

        if resp.status_code == 401:
            return {"response": "", "response_time": 0, "error": "401 Unauthorized - session cookie expired or invalid"}

        if not resp.ok:
            return {"response": "", "response_time": 0, "error": f"HTTP {resp.status_code}: {resp.text[:500]}"}

        # Read NDJSON stream - collect chunks and final event
        chunks = []
        final_response = ""
        model_used = model

        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            etype = event.get("type", "")

            if etype == "chunk":
                content = event.get("content", "")
                if event.get("accumulated"):
                    final_response = content
                else:
                    chunks.append(content)

            elif etype == "final":
                final_response = event.get("response", final_response)
                model_used = event.get("model_used") or event.get("model") or model

            elif etype == "error":
                return {
                    "response": "",
                    "response_time": time.perf_counter() - start,
                    "error": event.get("message", "Unknown streaming error"),
                }

        elapsed = time.perf_counter() - start

        if not final_response and chunks:
            final_response = "".join(chunks)

        return {
            "response": final_response,
            "response_time": round(elapsed, 3),
            "model_used": model_used,
            "error": None,
        }

    except requests.exceptions.Timeout:
        return {"response": "", "response_time": time.perf_counter() - start, "error": "Request timed out"}
    except requests.exceptions.ConnectionError as e:
        return {"response": "", "response_time": 0, "error": f"Connection error: {e}"}


def main():
    parser = argparse.ArgumentParser(description="Benchmark Archi with multiple models")
    parser.add_argument("--base-url", required=True, help="Archi instance URL, e.g. https://host:7869")
    parser.add_argument("--cookie", required=True, help="Flask session cookie value (from browser after SSO login)")
    parser.add_argument("--input", required=True, help="Path to user_questions_with_feedback.csv")
    parser.add_argument("--output", required=True, help="Output CSV path for results")
    parser.add_argument("--gpu-spec", default="", help="GPU spec string to include in results (entered manually)")
    parser.add_argument("--models", nargs="*", default=None, help="Override model list (space-separated)")
    parser.add_argument("--no-verify", action="store_true", help="Disable SSL verification")
    parser.add_argument("--max-questions", type=int, default=None, help="Limit number of questions (for testing)")
    parser.add_argument("--resume", action="store_true", help="Resume from where a previous run left off (appends to output)")
    args = parser.parse_args()

    models = args.models if args.models else MODELS
    verify_ssl = not args.no_verify

    # Load questions
    questions = extract_questions(args.input)
    if args.max_questions:
        questions = questions[:args.max_questions]

    print(f"Loaded {len(questions)} questions")
    print(f"Models to benchmark: {models}")
    print(f"Total runs: {len(questions) * len(models)}")

    # Handle resume: figure out how many rows already exist
    already_done = 0
    if args.resume:
        try:
            with open(args.output, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                already_done = sum(1 for _ in reader)
            print(f"Resuming: {already_done} rows already completed")
        except FileNotFoundError:
            already_done = 0

    # Set up session with cookie
    sess = requests.Session()
    sess.cookies.set("session", args.cookie)

    fieldnames = ["question", "response", "response_time_seconds", "model_name", "gpu_spec", "error"]

    # Open in append mode if resuming, write mode otherwise
    write_mode = "a" if args.resume and already_done > 0 else "w"
    write_header = write_mode == "w"

    with open(args.output, write_mode, newline="", encoding="utf-8") as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()

        total = len(questions) * len(models)
        global_idx = 0

        for model in models:
            print(f"\n{'='*60}")
            print(f"Model: {model}")
            print(f"{'='*60}")

            for i, q_item in enumerate(questions):
                global_idx += 1

                # Skip already-completed rows when resuming
                if global_idx <= already_done:
                    continue

                question = q_item["question"]
                display_q = question[:80] + "..." if len(question) > 80 else question
                print(f"  [{global_idx}/{total}] Q: {display_q}")

                result = send_question(sess, args.base_url, question, model, verify_ssl)

                row = {
                    "question": question,
                    "response": result.get("response", ""),
                    "response_time_seconds": result.get("response_time", 0),
                    "model_name": result.get("model_used", model),
                    "gpu_spec": args.gpu_spec,
                    "error": result.get("error", ""),
                }
                writer.writerow(row)
                outf.flush()

                if result.get("error"):
                    print(f"         ERROR: {result['error']}")
                else:
                    print(f"         Done in {result['response_time']:.1f}s ({len(result['response'])} chars)")

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
