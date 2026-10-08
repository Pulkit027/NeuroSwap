#!/usr/bin/env python3
"""Run an experiment plan: many scripts/run_one.py runs, resumable.

A plan (JSON, in experiments/) fixes the prompt selection, memory limits,
strategies, repetitions and run order. Every run gets a stable run_id
(sha256 of prompt_id|memory_limit|strategy|pin_fraction|repetition), the run
list is shuffled with the plan's order seed, and each run is executed as a
run_one.py subprocess.

Resuming: run_ids already present in the output file are skipped, including
errored ones unless --retry-errors is given (a retried run adds a new line;
analysis should use the last record per run_id).

Output safety: run_one writes into a temporary file; only a finished record is
appended to the output, in one write with SIGINT blocked, then fsynced. On
Ctrl+C or SIGTERM the current run_one is interrupted, its partial record is
discarded and the run's cgroup is checked and removed if left behind.

Plans with a memory limit need root (run by the user with sudo). The runner
refuses to start when the 1-minute load average is above 1.0 unless --force.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUN_ONE = REPO / "scripts/run_one.py"
CGROUP = Path("/sys/fs/cgroup/neuroswap_run")  # must match run_one.py
STRATEGIES = {"baseline", "prefetch", "pin"}
PLAN_KEYS = {"plan_name", "description", "prompts", "memory_limits", "strategies",
             "repetitions", "order_seed", "repack", "output"}
MAX_LOAD_1MIN = 1.0
CHILD_STOP_TIMEOUT_S = 60

EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_INTERRUPTED = 0, 1, 2, 130


class PlanError(Exception):
    pass


# ---------------------------------------------------------------- plan

def load_plan(path):
    try:
        plan = json.loads(Path(path).read_text())
    except (OSError, ValueError) as e:
        raise PlanError(f"cannot read plan {path}: {e}")
    unknown = set(plan) - PLAN_KEYS
    missing = PLAN_KEYS - {"description"} - set(plan)
    if unknown or missing:
        raise PlanError(f"plan keys: unknown {sorted(unknown)}, missing {sorted(missing)}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", str(plan["plan_name"])):
        raise PlanError("plan_name must match [A-Za-z0-9_.-]+")
    prompts = plan["prompts"]
    if set(prompts) != {"per_bucket", "seed", "manifest"} or \
            not isinstance(prompts["per_bucket"], int) or prompts["per_bucket"] < 1:
        raise PlanError('prompts must be {"per_bucket": int >= 1, "seed": int, "manifest": path}')
    for limit in plan["memory_limits"]:
        if limit != "none" and not re.fullmatch(r"\d+[KMG]?", str(limit)):
            raise PlanError(f"bad memory limit {limit!r}")
    if not plan["memory_limits"] or len(set(plan["memory_limits"])) != len(plan["memory_limits"]):
        raise PlanError("memory_limits must be a non-empty list without duplicates")
    seen = set()
    for item in plan["strategies"]:
        strategy, fraction = item.get("strategy"), item.get("fraction")
        if strategy not in STRATEGIES or set(item) - {"strategy", "fraction"}:
            raise PlanError(f"bad strategy entry {item}")
        if strategy == "pin":
            if not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
                raise PlanError(f"pin needs 0 < fraction <= 1: {item}")
        elif fraction is not None:
            raise PlanError(f"fraction is only valid for pin: {item}")
        key = (strategy, fraction)
        if key in seen:
            raise PlanError(f"duplicate strategy entry {item}")
        seen.add(key)
    if not isinstance(plan["repetitions"], int) or plan["repetitions"] < 1:
        raise PlanError("repetitions must be an int >= 1")
    if not isinstance(plan["order_seed"], int):
        raise PlanError("order_seed must be an int")
    if plan["repack"] not in ("on", "off"):
        raise PlanError("repack must be on or off")
    return plan


def select_prompts(plan):
    """N prompt_ids per bucket, buckets in manifest order, sampled with the plan seed."""
    manifest = REPO / plan["prompts"]["manifest"]
    by_bucket = {}
    with open(manifest, newline="") as f:
        for row in csv.DictReader(f):
            by_bucket.setdefault(row["bucket"], []).append(row["prompt_id"])
    rng = random.Random(plan["prompts"]["seed"])
    selected = []
    for bucket, ids in by_bucket.items():
        n = plan["prompts"]["per_bucket"]
        if n > len(ids):
            raise PlanError(f"bucket {bucket} has only {len(ids)} prompts, plan wants {n}")
        selected += [(pid, bucket) for pid in sorted(rng.sample(sorted(ids), n))]
    return selected


def run_id(prompt_id, memory_limit, strategy, fraction, repetition):
    fraction_text = "-" if fraction is None else f"{float(fraction):.6f}"
    key = f"{prompt_id}|{memory_limit}|{strategy}|{fraction_text}|{repetition}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def expand(plan):
    """All runs of the plan, shuffled with order_seed."""
    runs = []
    for prompt_id, bucket in select_prompts(plan):
        for limit in plan["memory_limits"]:
            for item in plan["strategies"]:
                for rep in range(plan["repetitions"]):
                    fraction = item.get("fraction")
                    runs.append({"run_id": run_id(prompt_id, limit, item["strategy"], fraction, rep),
                                 "prompt_id": prompt_id, "bucket": bucket, "memory_limit": limit,
                                 "strategy": item["strategy"], "pin_fraction": fraction,
                                 "repetition": rep})
    if len({r["run_id"] for r in runs}) != len(runs):
        raise PlanError("run_id collision")
    random.Random(plan["order_seed"]).shuffle(runs)
    return runs


def label(run):
    strategy = run["strategy"] + (f" {run['pin_fraction']}" if run["pin_fraction"] is not None else "")
    return f"{run['prompt_id']} {run['bucket']:>9} {run['memory_limit']:>5} {strategy:<10} rep{run['repetition']}"


# ---------------------------------------------------------------- output file

def give_back_to_invoking_user(paths):
    """Under sudo, chown files we created back to SUDO_UID/SUDO_GID (not recursive)."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if os.geteuid() != 0 or uid is None or gid is None:
        return
    for p in paths:
        try:
            os.chown(p, int(uid), int(gid), follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"run_experiment: warning: could not chown {p}: {e}", file=sys.stderr)


def load_records(out, repair):
    """Records of the output file. A trailing incomplete line (crash mid-write) is
    moved to <out>.partial and truncated when repair is true; any other invalid
    line is an error."""
    if not out.exists():
        return []
    data = out.read_bytes()
    lines = data.split(b"\n")
    tail = lines.pop()  # b"" when the file ends with a newline
    records = []
    for number, line in enumerate(lines, 1):
        try:
            records.append(json.loads(line))
        except ValueError:
            raise PlanError(f"{out}: line {number} is not valid JSON; fix the file by hand")
    if tail:
        if not repair:
            print(f"run_experiment: {out} ends with an incomplete line (would be repaired)")
            return records
        partial = out.with_name(out.name + ".partial")
        with open(partial, "ab") as f:
            f.write(tail + b"\n")
        with open(out, "r+b") as f:
            f.truncate(len(data) - len(tail))
        print(f"run_experiment: moved an incomplete last line of {out} to {partial}")
        give_back_to_invoking_user([partial])
    return records


def append_record(out, record):
    """One write of one complete line, SIGINT/SIGTERM blocked, then fsync."""
    data = (json.dumps(record) + "\n").encode()
    blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
    try:
        fd = os.open(out, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            written = os.write(fd, data)
            if written != len(data):
                raise OSError(f"short write to {out}: {written} of {len(data)} bytes")
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, blocked)


def write_status(path, status):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(status, indent=2) + "\n")
    os.replace(tmp, path)


def done_run_ids(records, retry_errors):
    done = set()
    for r in records:
        if "run_id" in r and (not retry_errors or r.get("error") is None):
            done.add(r["run_id"])
    return done


# ---------------------------------------------------------------- execution

def cleanup_cgroup():
    """Remove the run cgroup if a run_one left it behind; True if it is gone."""
    if not CGROUP.exists():
        return True
    try:
        procs = (CGROUP / "cgroup.procs").read_text().split()
    except OSError:
        procs = []
    if procs:
        print(f"run_experiment: {CGROUP} still has processes {procs}; not removed", file=sys.stderr)
        return False
    try:
        CGROUP.rmdir()
        print(f"run_experiment: removed leftover {CGROUP}")
        return True
    except OSError as e:
        print(f"run_experiment: could not remove {CGROUP}: {e}", file=sys.stderr)
        return False


def stop_child(proc):
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)  # run_one's finally blocks remove its cgroup
        try:
            proc.wait(timeout=CHILD_STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def on_sigterm(signum, frame):
    raise KeyboardInterrupt


def fmt_s(seconds):
    if seconds is None:
        return "--"
    seconds = int(round(seconds))
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def run_one_cmd(args, plan, run, out):
    cmd = [args.python, str(args.run_one), "--prompt-id", run["prompt_id"],
           "--memory-limit", run["memory_limit"], "--strategy", run["strategy"],
           "--repack", plan["repack"], "--run-index", str(run["repetition"]), "--out", str(out)]
    if run["pin_fraction"] is not None:
        cmd += ["--pin-fraction", repr(float(run["pin_fraction"]))]
    return cmd


def execute(args, plan, runs, out, todo):
    status_path = out.with_name(out.name + ".status.json")
    status = {"plan_name": plan["plan_name"], "output": str(out), "total_runs": len(runs),
              "to_run_this_session": len(todo), "done_this_session": 0, "errors_this_session": 0,
              "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "state": "running", "current": None, "mean_run_s": None, "eta_s": None}
    durations = []
    signal.signal(signal.SIGTERM, on_sigterm)
    t_session = time.monotonic()

    def update(**changes):
        status.update(changes, updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        write_status(status_path, status)
        give_back_to_invoking_user([out, status_path])

    update()
    for index, run in enumerate(todo, 1):
        update(current={"index": index, **run})
        with tempfile.TemporaryDirectory(prefix="neuroswap_exp_") as tmp:
            tmp_out = Path(tmp) / "record.jsonl"
            t0 = time.monotonic()
            proc = subprocess.Popen(run_one_cmd(args, plan, run, tmp_out))
            try:
                rc = proc.wait()
            except KeyboardInterrupt:
                stop_child(proc)
                rc = None
            elapsed = time.monotonic() - t0
            lines = tmp_out.read_text().splitlines() if tmp_out.exists() else []

            if rc is None or rc < 0:
                # Interrupted (Ctrl+C, SIGTERM, or run_one killed by a signal):
                # its record, if any, is incomplete and is discarded.
                cleanup_cgroup()
                update(state="interrupted", current=None)
                print(f"\nrun_experiment: interrupted during {run['run_id']}; its partial record "
                      f"was discarded. {index - 1} run(s) recorded this session. "
                      f"Rerun the same command to resume.", file=sys.stderr)
                return EXIT_INTERRUPTED
            if rc not in (0, 1) or len(lines) != 1:
                cleanup_cgroup()
                update(state="failed", current=None)
                print(f"run_experiment: run_one exited {rc} with {len(lines)} record line(s) for "
                      f"{run['run_id']}; stopping (nothing written for this run).", file=sys.stderr)
                return EXIT_FAILED

            record = json.loads(lines[0])
            record["run_id"] = run["run_id"]
            record["plan_name"] = plan["plan_name"]
            append_record(out, record)

        durations.append(elapsed)
        mean = sum(durations) / len(durations)
        remaining = len(todo) - index
        errors = status["errors_this_session"] + (record.get("error") is not None)
        update(done_this_session=index, errors_this_session=errors, current=None,
               mean_run_s=round(mean, 1), eta_s=round(mean * remaining),
               last_run_id=run["run_id"])
        outcome = "ok" if record.get("error") is None else "ERROR " + str(record["error"])[:80]
        print(f"[{index}/{len(todo)}] {run['run_id']} {label(run)} {elapsed:6.1f}s {outcome} | "
              f"elapsed {fmt_s(time.monotonic() - t_session)} ETA {fmt_s(mean * remaining)}",
              flush=True)

    update(state="finished")
    return EXIT_OK


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", type=Path, help="plan JSON, e.g. experiments/pilot.json")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and run list; run nothing")
    ap.add_argument("--retry-errors", action="store_true", help="rerun run_ids whose records have an error")
    ap.add_argument("--force", action="store_true", help=f"start even if load average > {MAX_LOAD_1MIN}")
    ap.add_argument("--run-one", type=Path, default=RUN_ONE, help=argparse.SUPPRESS)  # test stub hook
    ap.add_argument("--python", default=sys.executable, help=argparse.SUPPRESS)
    args = ap.parse_args()

    try:
        plan = load_plan(args.plan)
        runs = expand(plan)
        out = (REPO / plan["output"]).resolve()
        records = load_records(out, repair=not args.dry_run)
    except PlanError as e:
        print(f"run_experiment: {e}", file=sys.stderr)
        return EXIT_REFUSED

    done = done_run_ids(records, args.retry_errors)
    todo = [r for r in runs if r["run_id"] not in done]
    errored = {r["run_id"] for r in records if r.get("error") is not None} - \
        {r["run_id"] for r in records if r.get("error") is None}
    by_bucket = {}
    for prompt_id, bucket in select_prompts(plan):  # manifest bucket order
        by_bucket.setdefault(bucket, []).append(prompt_id)
    strategies = ", ".join(s["strategy"] + (f" {s['fraction']}" if "fraction" in s else "")
                           for s in plan["strategies"])
    print(f"plan          {plan['plan_name']} ({args.plan})")
    if plan.get("description"):
        print(f"description   {plan['description']}")
    for bucket, ids in by_bucket.items():
        print(f"prompts       {bucket:>9}: {' '.join(ids)}")
    print(f"selection     {plan['prompts']['per_bucket']} per bucket, seed {plan['prompts']['seed']}")
    print(f"memory limits {', '.join(plan['memory_limits'])}  (limited runs are cold; none runs are warm)")
    print(f"strategies    {strategies}")
    print(f"repetitions   {plan['repetitions']}, order seed {plan['order_seed']}, repack {plan['repack']}")
    print(f"output        {out}")
    print(f"runs          {len(runs)} total, {len(runs) - len(todo)} already recorded, "
          f"{len(todo)} to run" + (f" ({len(errored)} errored run_ids, rerun with --retry-errors)"
                                   if errored and not args.retry_errors else ""))

    if args.dry_run:
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)  # quiet when piped into head
        print(f"\n{'#':>4} {'run_id':16} {'prompt':6} {'bucket':>9} {'limit':>5} {'strategy':<10} rep  state")
        for i, r in enumerate(runs, 1):
            state = "recorded" if r["run_id"] in done else "todo"
            print(f"{i:>4} {r['run_id']} {label(r)}  {state}")
        return EXIT_OK

    needs_root = any(limit != "none" for limit in plan["memory_limits"])
    if needs_root and os.geteuid() != 0:
        print("run_experiment: this plan has memory-limited runs, which need root. Run it yourself:\n"
              f"  sudo {sys.executable} {' '.join(sys.argv)}", file=sys.stderr)
        return EXIT_REFUSED
    load = os.getloadavg()[0]
    if load > MAX_LOAD_1MIN:
        print(f"run_experiment: warning: 1-minute load average is {load:.2f} (> {MAX_LOAD_1MIN}); "
              "the machine is not idle", file=sys.stderr)
        if not args.force:
            print("run_experiment: refusing to start; wait for the machine to be idle or pass --force",
                  file=sys.stderr)
            return EXIT_REFUSED
    if not cleanup_cgroup():
        return EXIT_REFUSED
    if not todo:
        print("run_experiment: nothing to run")
        return EXIT_OK

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        return execute(args, plan, runs, out, todo)
    except KeyboardInterrupt:
        # Between runs: every recorded line is complete (appends block SIGINT).
        print("\nrun_experiment: interrupted between runs; rerun the same command to resume.",
              file=sys.stderr)
        return EXIT_INTERRUPTED
    finally:
        cleanup_cgroup()


if __name__ == "__main__":
    sys.exit(main())
