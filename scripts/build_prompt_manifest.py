#!/usr/bin/env python3
"""Build the NeuroSwap prompt manifest from one WildChat-1M parquet shard.

Keeps only the first user turn of each conversation, filters it, counts tokens
with the real Qwen2.5 GGUF tokenizer (via llama-server /tokenize), buckets by
token length and draws a seeded stratified sample.

Outputs:
  data/manifest.csv   prompt metadata, no prompt text (committed)
  data/prompts.jsonl  prompt_id + prompt text (gitignored, regenerated)
  docs/DATASET.md     dataset card, all numbers from this run
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parent.parent
HOME = Path.home()

KEEP_COLUMNS = ["conversation_hash", "language", "toxic", "redacted",
                "openai_moderation", "detoxify_moderation", "conversation"]
# Only these fields of a conversation turn are ever materialized. The per-turn
# hashed_ip / header / state / country are never selected.
TURN_FIELDS = ["role", "content", "turn_identifier"]
PII_FIELDS = {"hashed_ip", "header", "state", "country"}

MAX_TOKENS = 1500
# Half-open token ranges [lo, hi); the last bucket includes MAX_TOKENS.
BUCKETS = [("<64", 0, 64), ("64-256", 64, 256), ("256-1024", 256, 1024),
           ("1024-1500", 1024, MAX_TOKENS + 1)]
PER_BUCKET = 75
SEED = 42

# Whitespace control characters that are legitimate in prompts.
ALLOWED_CONTROL = {"\n", "\t", "\r"}
# Cc control, Cf format (zero-width, BOM, bidi), Cs surrogate, Co private use,
# Cn unassigned.
JUNK_CATEGORIES = {"Cc", "Cf", "Cs", "Co", "Cn"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def has_junk(text):
    for ch in text:
        if ch in ALLOWED_CONTROL:
            continue
        if ch == "�" or unicodedata.category(ch) in JUNK_CATEGORIES:
            return True
    return False


def has_moderation(entry):
    return (entry is not None and entry["flagged"] is not None
            and entry["categories"] is not None)


def moderation_flagged(entry):
    return bool(entry["flagged"]) or any(entry["categories"].values())


def load_rows(path):
    table = pq.read_table(path, columns=KEEP_COLUMNS)
    assert not PII_FIELDS & set(table.column_names)
    # Project the first turn down to role/content/turn_identifier and drop the
    # full conversation (which carries per-turn PII) right away.
    first = pc.list_element(table["conversation"], 0)
    first_turn = {f: pc.struct_field(first, f).to_pylist() for f in TURN_FIELDS}
    table = table.drop_columns(["conversation"])
    oai_first = pc.list_element(table["openai_moderation"], 0).to_pylist()
    cols = {c: table[c].to_pylist()
            for c in ["conversation_hash", "language", "toxic", "redacted"]}
    del table
    rows = []
    for i in range(len(oai_first)):
        rows.append({
            "conversation_hash": cols["conversation_hash"][i],
            "language": cols["language"][i],
            "toxic": cols["toxic"][i],
            "redacted": cols["redacted"][i],
            "oai_first": oai_first[i],
            "role": first_turn["role"][i],
            "content": first_turn["content"][i],
            "turn_identifier": first_turn["turn_identifier"][i],
        })
    return rows


def apply_filters(rows, log):
    filters = [
        ("first turn role == user", lambda r: r["role"] == "user"),
        ("language == English", lambda r: r["language"] == "English"),
        ("toxic == False", lambda r: r["toxic"] is False),
        ("redacted == False", lambda r: r["redacted"] is False),
        ("OpenAI moderation result present on first user turn",
         lambda r: has_moderation(r["oai_first"])),
        ("no OpenAI moderation flag/category on first user turn",
         lambda r: not moderation_flagged(r["oai_first"])),
        ("non-empty content",
         lambda r: r["content"] is not None and r["content"].strip() != ""),
        ("no non-printable junk", lambda r: not has_junk(r["content"])),
    ]
    counts = []
    for name, keep in filters:
        before = len(rows)
        rows = [r for r in rows if keep(r)]
        counts.append((name, before - len(rows)))
        log(f"  {name:<55} removed {before - len(rows):>6}  remaining {len(rows):>6}")
    # Exact duplicates: keep the first occurrence in shard order.
    before = len(rows)
    seen, unique = set(), []
    for r in rows:
        if r["content"] not in seen:
            seen.add(r["content"])
            unique.append(r)
    rows = unique
    name = "drop exact duplicate prompts"
    counts.append((name, before - len(rows)))
    log(f"  {name:<55} removed {before - len(rows):>6}  remaining {len(rows):>6}")
    return rows, counts


class TokenizerServer:
    def __init__(self, server_bin, model, port):
        self.url = f"http://127.0.0.1:{port}"
        self.cmd = [str(server_bin), "-m", str(model), "-c", "2048",
                    "--host", "127.0.0.1", "--port", str(port), "-t", "4"]
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        deadline = time.time() + 120
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited with {self.proc.returncode}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as resp:
                    if resp.status == 200:
                        return self
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.5)
        raise RuntimeError("llama-server did not become healthy")

    def __exit__(self, *exc):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

    def count(self, text):
        # add_special=False: count the prompt text itself, no BOS/chat template.
        body = json.dumps({"content": text, "add_special": False}).encode()
        req = urllib.request.Request(self.url + "/tokenize", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return len(json.load(resp)["tokens"])


def bucket_of(n):
    for name, lo, hi in BUCKETS:
        if lo <= n < hi:
            return name
    return None


def write_dataset_md(path, info):
    lines = [
        "# Prompt dataset",
        "",
        "Generated by `scripts/build_prompt_manifest.py`; every number below comes from that run.",
        "",
        "## Source (source = external)",
        "",
        "- Dataset: WildChat-1M (allenai/WildChat-1M), https://huggingface.co/datasets/allenai/WildChat-1M",
        "- License: ODC-BY (Open Data Commons Attribution License)",
        "- Paper: Zhao et al., \"WildChat: 1M ChatGPT Interaction Logs in the Wild\", ICLR 2024, arXiv:2405.01470",
        f"- Shard used: `{info['shard_name']}` ({info['shard_rows']} rows), kept outside the repo",
        f"- Shard sha256: `{info['shard_sha256']}`",
        "",
        "Only the first user turn of each conversation is used. `hashed_ip`, `header`,",
        "`state` and `country` (top-level and per-turn) are never read into any output.",
        "",
        "## Filters (applied in order)",
        "",
        "| filter | removed | remaining |",
        "|---|---|---|",
    ]
    remaining = info["shard_rows"]
    for name, removed in info["filter_counts"]:
        remaining -= removed
        lines.append(f"| {name} | {removed} | {remaining} |")
    lines += [
        "",
        "Notes: moderation uses the OpenAI moderation entry of the first user turn",
        "(`flagged` or any category true). Entries with no moderation result (null",
        "`flagged`/`categories`) are dropped, since they cannot be shown unflagged.",
        "Detoxify has scores only, no flags, so it is",
        "not used as a filter. Non-printable junk = Unicode categories Cc (except",
        "\\n \\t \\r), Cf, Cs, Co, Cn, or U+FFFD.",
        "",
        "## Token counting",
        "",
        f"- Tokenizer: `{info['model_name']}` via llama.cpp `llama-server` `/tokenize`, `add_special=false`",
        "- Counts are for the raw prompt text, without chat template tokens.",
        f"- Prompts tokenized: {info['n_tokenized']}",
        f"- Dropped for > {MAX_TOKENS} tokens: {info['n_too_long']}",
        f"- Eligible (<= {MAX_TOKENS} tokens): {info['n_eligible']}",
        "",
        "## Stratified sample",
        "",
        f"- Seed: {SEED}, up to {PER_BUCKET} prompts per bucket",
        "- Buckets are half-open token ranges [lo, hi); 1024-1500 includes 1500.",
        "",
        "| bucket | eligible | sampled | tokens min | median | max |",
        "|---|---|---|---|---|---|",
    ]
    for b in info["buckets"]:
        lines.append(f"| {b['name']} | {b['eligible']} | {b['sampled']} | "
                     f"{b['min']} | {b['median']} | {b['max']} |")
    s = info["sample_stats"]
    lines += [
        f"| **all** | {info['n_eligible']} | {s['n']} | {s['min']} | {s['median']} | {s['max']} |",
        "",
    ]
    short = [b["name"] for b in info["buckets"] if b["sampled"] < PER_BUCKET]
    if short:
        lines.append(f"Buckets with fewer than {PER_BUCKET} eligible prompts (all taken): {', '.join(short)}.")
    else:
        lines.append(f"Every bucket had at least {PER_BUCKET} eligible prompts.")
    lines += [
        "",
        "## Files",
        "",
        "- `data/manifest.csv`: prompt_id, conversation_hash, turn_identifier, token_count, bucket, source. No prompt text.",
        "- `data/prompts.jsonl`: prompt_id and prompt text. Gitignored; regenerate with the script.",
        "",
    ]
    path.write_text("\n".join(lines))


def stats(values):
    if not values:
        return {"n": 0, "min": "-", "median": "-", "max": "-"}
    med = float(np.median(values))
    return {"n": len(values), "min": min(values),
            "median": int(med) if med.is_integer() else med, "max": max(values)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=Path, default=HOME / "data/wildchat_0000.parquet")
    ap.add_argument("--model", type=Path,
                    default=HOME / "models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    ap.add_argument("--server-bin", type=Path,
                    default=HOME / "tools/llama.cpp/build/bin/llama-server")
    ap.add_argument("--port", type=int, default=8089)
    ap.add_argument("--out-dir", type=Path, default=REPO / "data")
    ap.add_argument("--doc", type=Path, default=REPO / "docs/DATASET.md")
    args = ap.parse_args()
    log = lambda msg: print(msg, flush=True)

    log(f"Loading {args.shard}")
    rows = load_rows(args.shard)
    shard_rows = len(rows)
    log(f"  rows: {shard_rows}")

    log("Filters:")
    rows, filter_counts = apply_filters(rows, log)

    log(f"Tokenizing {len(rows)} prompts with {args.model.name} via llama-server")
    with TokenizerServer(args.server_bin, args.model, args.port) as tok:
        for i, r in enumerate(rows, 1):
            r["token_count"] = tok.count(r["content"])
            if i % 5000 == 0:
                log(f"  {i}/{len(rows)}")
    log("  llama-server stopped")

    n_tokenized = len(rows)
    eligible = [r for r in rows if r["token_count"] <= MAX_TOKENS]
    n_too_long = n_tokenized - len(eligible)
    log(f"  dropped > {MAX_TOKENS} tokens: {n_too_long}  eligible: {len(eligible)}")

    rng = np.random.default_rng(SEED)
    sample, bucket_info = [], []
    log(f"Stratified sample (seed {SEED}, up to {PER_BUCKET} per bucket):")
    for name, _, _ in BUCKETS:
        members = sorted((r for r in eligible if bucket_of(r["token_count"]) == name),
                         key=lambda r: (r["conversation_hash"], r["turn_identifier"]))
        if len(members) > PER_BUCKET:
            idx = sorted(rng.choice(len(members), size=PER_BUCKET, replace=False))
            chosen = [members[i] for i in idx]
        else:
            chosen = members
        for r in chosen:
            r["bucket"] = name
        sample.extend(chosen)
        st = stats([r["token_count"] for r in chosen])
        bucket_info.append({"name": name, "eligible": len(members), "sampled": len(chosen),
                            "min": st["min"], "median": st["median"], "max": st["max"]})
        note = "  (fewer than target, took all)" if len(chosen) < PER_BUCKET else ""
        log(f"  {name:<10} eligible {len(members):>6}  sampled {len(chosen):>3}  "
            f"tokens min/median/max {st['min']}/{st['median']}/{st['max']}{note}")

    sample_stats = stats([r["token_count"] for r in sample])
    log(f"  total sampled {sample_stats['n']}  tokens min/median/max "
        f"{sample_stats['min']}/{sample_stats['median']}/{sample_stats['max']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.csv"
    prompts_path = args.out_dir / "prompts.jsonl"
    with open(manifest_path, "w", newline="") as mf, open(prompts_path, "w") as pf:
        w = csv.writer(mf, lineterminator="\n")
        w.writerow(["prompt_id", "conversation_hash", "turn_identifier",
                    "token_count", "bucket", "source"])
        for i, r in enumerate(sample):
            pid = f"wc{i:04d}"
            w.writerow([pid, r["conversation_hash"], r["turn_identifier"],
                        r["token_count"], r["bucket"], "external"])
            pf.write(json.dumps({"prompt_id": pid, "prompt": r["content"]},
                                ensure_ascii=False) + "\n")

    write_dataset_md(args.doc, {
        "shard_name": args.shard.name, "shard_rows": shard_rows,
        "shard_sha256": sha256_file(args.shard), "filter_counts": filter_counts,
        "model_name": args.model.name, "n_tokenized": n_tokenized,
        "n_too_long": n_too_long, "n_eligible": len(eligible),
        "buckets": bucket_info, "sample_stats": sample_stats,
    })
    log(f"Wrote {manifest_path}, {prompts_path}, {args.doc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
