#!/usr/bin/env python3
"""Run relation extraction over input shards against one vLLM server.

Resumable by design: a 25-hour run will be interrupted, so every completed
pmcid is appended as it finishes and a restart skips what is already on disk.
Records whose `prompt` is empty (no_relations / no_text / oversized) are
carried through with an empty label list, so the output stays a complete image
of the input shard rather than a filtered subset.

    python run_extraction.py --shards full_input/input_000.jsonl \
        --out labels_000.jsonl --base-url http://127.0.0.1:8000/v1

Start the server first with serve.sh, which pins the model and the context
window the published labels were produced with.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from vllm_client import chat, parse_labels, parse_reasons


def main() -> int:
    args = parse_arguments()
    done = already_annotated(args.out)
    pending = [r for shard in args.shards for r in read_shard(shard)
               if record_key(r) not in done]
    print(f"[{args.tag}] {len(pending):,} publications pending "
          f"({len(done):,} already done)", flush=True)

    annotate_all(pending, args)
    return 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--request-timeout", type=int, default=1800)
    parser.add_argument("--tag", default="prod")
    return parser.parse_args()


def read_shard(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def record_key(record: dict) -> str:
    """Identity for resume. A chunked roster splits one publication across
    several records, so pmcid alone would make the first chunk suppress the
    rest."""
    chunk = record.get("chunk")
    return record["pmcid"] if chunk is None else f"{record['pmcid']}#{chunk}"


def already_annotated(out: Path) -> set[str]:
    """Tolerates a truncated final line from an interrupted run."""
    if not out.exists():
        return set()
    seen = set()
    with out.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                seen.add(record_key(json.loads(line)))
            except (json.JSONDecodeError, KeyError):
                continue
    return seen


def annotate_all(pending: list[dict], args) -> None:
    started = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as sink:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(annotate_one, record, args): record_key(record)
                       for record in pending}
            for position, future in enumerate(as_completed(futures), 1):
                sink.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                sink.flush()                       # a 25-hour run must not buffer
                if position % 100 == 0:
                    rate = position / (time.time() - started)
                    print(f"[{args.tag}] {position:,}/{len(pending):,} "
                          f"at {3600*rate:.0f}/hr, "
                          f"~{(len(pending)-position)/rate/3600:.1f}h left", flush=True)


def annotate_one(record: dict, args) -> dict:
    """One publication. Empty-prompt records pass straight through."""
    if not record["prompt"]:
        return {"pmcid": record["pmcid"], "chunk": record.get("chunk"),
                "status": record["status"], "labels": [], "skipped": True}

    reply = chat(args.base_url, args.model, record["prompt"], args.max_tokens,
                 args.temperature, args.request_timeout, think=True)
    labels = parse_labels(reply.get("text", ""))
    reasons = parse_reasons(reply.get("text", ""))
    asked = [relation["row_id"] for relation in record["relations"]]
    return {
        "pmcid": record["pmcid"],
        "chunk": record.get("chunk"),
        "status": record["status"],
        "labels": [{"row_id": rid, "label": labels.get(rid, ""),
                    "reason": reasons.get(rid, "")} for rid in asked],
        "missing": [rid for rid in asked if rid not in labels],
        "error": reply.get("error", ""),
        "finish": reply.get("finish", ""),
        "prompt_tokens": reply.get("ptok"),
        "completion_tokens": reply.get("ctok"),
        "latency_s": reply.get("latency"),
    }


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted; rerun to resume", file=sys.stderr)
        sys.exit(130)
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
