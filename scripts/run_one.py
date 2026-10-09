#!/usr/bin/env python3
"""Run one manifest prompt through llama.cpp and append one telemetry record.

The prompt is fed raw (no chat template, -no-cnv), as in Phase 1. Unlike
Phase 1, weight repacking is off by default (--no-repack), so all weights stay
file-backed; --repack on restores llama.cpp's default.
Strategies: baseline runs llama.cpp alone. prefetch and pin first start the
C++ helper build/neuroswap_strategy (--hold) in the same cgroup, wait for its
READY line, run llama.cpp exactly as for baseline, then release the helper.
A helper refusal (exit 3) or failure is recorded as an error and no inference
runs; there is no fallback to baseline.

Runs with a memory limit are cold by default (sync + drop_caches first), so the
model's page-cache pages are charged to the run's cgroup; --warm turns that off.
--memory-limit other than "none" and cold runs need root (cgroup v2 writes and
/proc/sys/vm/drop_caches). They are run by the user, never by automation.

The output record never contains prompt text or generated text.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import pwd
import re
import resource
import select
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def invoking_user_home():
    """Home of the user who invoked us, also under sudo (where ~ is /root)."""
    sudo_user = os.environ.get("SUDO_USER")
    if os.geteuid() == 0 and sudo_user:
        return Path(pwd.getpwnam(sudo_user).pw_dir)
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


# Repo-relative paths (data, results) come from this file's location.
REPO = Path(__file__).resolve().parent.parent
USER_HOME = invoking_user_home()
DEFAULT_LLAMA_BIN = USER_HOME / "tools/llama.cpp/build/bin/llama-completion"
DEFAULT_MODEL = USER_HOME / "models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
CGROUP = Path("/sys/fs/cgroup/neuroswap_run")
SHA_CACHE = REPO / "results/raw/llm/.model_sha256.json"
STRATEGIES = ["baseline", "prefetch", "pin"]
HELPER_BIN = REPO / "build/neuroswap_strategy"
PREFETCH_CHUNK_MIB = "8"        # madvise(WILLNEED) per chunk; 32 MiB plateaued at 8 MiB
PREFETCH_WAIT_RESIDENT = "99"   # per-chunk target
PREFETCH_WAIT_TIMEOUT_S = "120"  # whole chunk loop
PREFETCH_WAIT_PLATEAU_S = "3"
HELPER_READY_TIMEOUT_S = 300   # helper setup incl. the prefetch residency wait
HELPER_RELEASE_TIMEOUT_S = 30
SAMPLE_INTERVAL_S = 0.1
STATUS_KEYS = ["VmRSS", "VmHWM", "RssAnon", "RssFile"]

PERF_PATTERNS = {
    "load_time_ms": r"load time =\s*([\d.]+) ms",
    "prompt_eval": r"prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second",
    "eval": r"\beval time =\s*([\d.]+) ms /\s*(\d+) runs.*?([\d.]+) tokens per second",
    "total_time": r"total time =\s*([\d.]+) ms /\s*(\d+) tokens",
}


def die(msg):
    print(f"run_one: {msg}", file=sys.stderr)
    sys.exit(2)


def load_prompt(prompt_id):
    meta = None
    with open(REPO / "data/manifest.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["prompt_id"] == prompt_id:
                meta = row
                break
    if meta is None:
        die(f"prompt_id {prompt_id} not in data/manifest.csv")
    with open(REPO / "data/prompts.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            if rec["prompt_id"] == prompt_id:
                return meta, rec["prompt"]
    die(f"prompt_id {prompt_id} not in data/prompts.jsonl (regenerate with build_prompt_manifest.py)")


def model_sha256(model):
    # Cached by size+mtime: hashing reads the whole model and would warm the
    # page cache, so it must never happen between drop_caches and the run.
    st = model.stat()
    key = {"path": str(model), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    try:
        cached = json.loads(SHA_CACHE.read_text())
        if {k: cached.get(k) for k in key} == key:
            return cached["sha256"]
    except (OSError, ValueError, KeyError):
        pass
    h = hashlib.sha256()
    with open(model, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    SHA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    SHA_CACHE.write_text(json.dumps({**key, "sha256": h.hexdigest()}))
    return h.hexdigest()


def llama_commit(llama_bin):
    # The binary lives inside the llama.cpp checkout (build/bin). safe.directory
    # lets git read a checkout owned by the invoking user when we run as root.
    src = llama_bin.resolve().parent
    try:
        return subprocess.run(["git", "-c", "safe.directory=*", "-C", str(src),
                               "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_meminfo_available_kb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    return None


def read_psi():
    out = {}
    try:
        with open("/proc/pressure/memory") as f:
            for line in f:
                kind, *fields = line.split()
                out[kind] = {k: float(v) if "." in v else int(v)
                             for k, v in (x.split("=") for x in fields)}
    except OSError:
        return None
    return out


def system_snapshot():
    return {"mem_available_kb": read_meminfo_available_kb(), "psi_memory": read_psi()}


def sample_proc(pid):
    """One sample of /proc/<pid>/{stat,status,io}; None if the process is gone."""
    s = {"t": time.monotonic()}
    try:
        with open(f"/proc/{pid}/stat") as f:
            rest = f.read().rsplit(")", 1)[1].split()
        # rest[0] is field 3 (state) in proc(5) numbering.
        s.update(state=rest[0], minflt=int(rest[7]), majflt=int(rest[9]),
                 utime_ticks=int(rest[11]), stime_ticks=int(rest[12]),
                 rss_pages=int(rest[21]))
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                key, _, val = line.partition(":")
                if key in STATUS_KEYS:
                    s[key + "_kb"] = int(val.split()[0])
    except (OSError, IndexError, ValueError):
        return None
    # An exiting process has released its mm: status loses the Vm*/Rss* lines
    # and io becomes unreadable. Such a sample describes teardown, not the run.
    if s["state"] in ("Z", "X") or "VmRSS_kb" not in s:
        return None
    try:
        with open(f"/proc/{pid}/io") as f:
            for line in f:
                key, _, val = line.partition(":")
                s["io_" + key] = int(val)
    except (OSError, ValueError):
        pass
    return s


def parse_perf(stderr_text):
    out = {k: None for k in [
        "load_time_ms", "prompt_eval_ms", "prompt_eval_tokens", "prompt_eval_tps",
        "eval_ms", "eval_runs", "eval_tps", "total_time_ms", "total_tokens"]}
    lines = [l for l in stderr_text.splitlines() if "common_perf_print" in l]
    text = "\n".join(lines)
    if m := re.search(PERF_PATTERNS["load_time_ms"], text):
        out["load_time_ms"] = float(m.group(1))
    if m := re.search(PERF_PATTERNS["prompt_eval"], text):
        out.update(prompt_eval_ms=float(m.group(1)), prompt_eval_tokens=int(m.group(2)),
                   prompt_eval_tps=float(m.group(3)))
    # "prompt eval time" also contains "eval time"; match only on its own line.
    for line in lines:
        if "prompt eval" in line:
            continue
        if m := re.search(PERF_PATTERNS["eval"], line):
            out.update(eval_ms=float(m.group(1)), eval_runs=int(m.group(2)),
                       eval_tps=float(m.group(3)))
    if m := re.search(PERF_PATTERNS["total_time"], text):
        out.update(total_time_ms=float(m.group(1)), total_tokens=int(m.group(2)))
    out["perf_lines_found"] = len(lines)
    return out


def cgroup_setup(limit):
    try:
        CGROUP.mkdir()
    except FileExistsError:
        die(f"{CGROUP} already exists (stale from an earlier run?); remove it with rmdir first")
    try:
        (CGROUP / "memory.max").write_text(limit)
        (CGROUP / "memory.swap.max").write_text("0")
    except OSError:
        cgroup_teardown()
        raise


def cgroup_read_events():
    try:
        return {k: int(v) for k, v in
                (l.split() for l in (CGROUP / "memory.events").read_text().splitlines())}
    except OSError:
        return None


def cgroup_teardown():
    for _ in range(50):
        try:
            CGROUP.rmdir()
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.1)  # the reaped child may linger briefly in cgroup.procs
    print(f"run_one: warning: could not remove {CGROUP}", file=sys.stderr)


def join_cgroup():
    # Runs in the child between fork and exec, so llama.cpp starts inside the cgroup.
    with open(CGROUP / "cgroup.procs", "w") as f:
        f.write("0")


def give_back_to_invoking_user(paths):
    """Under sudo, chown files we created back to SUDO_UID/SUDO_GID (not recursive)."""
    uid, gid = os.environ.get("SUDO_UID"), os.environ.get("SUDO_GID")
    if os.geteuid() != 0 or uid is None or gid is None:
        return
    for p in dict.fromkeys(paths):
        try:
            os.chown(p, int(uid), int(gid), follow_symlinks=False)
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"run_one: warning: could not chown {p}: {e}", file=sys.stderr)


def drop_caches():
    os.sync()
    with open("/proc/sys/vm/drop_caches", "w") as f:
        f.write("3")


def read_line(pipe, deadline):
    """One line from a pipe, "" on EOF, None if the deadline passes first."""
    fd, buf = pipe.fileno(), b""
    while not buf.endswith(b"\n"):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
            return None
        chunk = os.read(fd, 4096)
        if not chunk:
            break
        buf += chunk
    return buf.decode(errors="replace")


def read_json_file(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def start_helper(args, tmp, use_cgroup, record):
    """Start the strategy helper and wait for READY.

    Returns (process, setup_seconds); process is None if the helper refused or
    failed, in which case record["error"] and record["helper_report"] are set.
    """
    report_path = tmp / "helper_report.json"
    cmd = [str(HELPER_BIN), "--model", str(args.model), "--strategy", args.strategy,
           "--hold", "--report", str(report_path)]
    if use_cgroup:
        cmd += ["--cgroup", str(CGROUP)]
    if args.strategy == "prefetch":
        cmd += ["--prefetch-chunk-mib", PREFETCH_CHUNK_MIB, "--wait-resident", PREFETCH_WAIT_RESIDENT,
                "--wait-timeout", PREFETCH_WAIT_TIMEOUT_S, "--wait-plateau-s", PREFETCH_WAIT_PLATEAU_S]
    if args.strategy == "pin":
        cmd += ["--fraction", repr(args.pin_fraction)]
    t0 = time.monotonic()
    with open(tmp / "helper_stderr.txt", "w") as err_f:
        # Joins the cgroup before exec too, so the helper's own startup is charged there.
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err_f,
                                preexec_fn=join_cgroup if use_cgroup else None)
    line = read_line(proc.stdout, t0 + HELPER_READY_TIMEOUT_S)
    setup_s = time.monotonic() - t0
    if line is not None and line.startswith("READY"):
        return proc, setup_s

    if line is None and proc.poll() is None:
        proc.kill()
        reason = f"no READY within {HELPER_READY_TIMEOUT_S}s, killed"
    else:
        reason = None
    code = proc.wait()
    record["helper_report"] = read_json_file(report_path)
    message = (record["helper_report"] or {}).get("error") or reason or \
        " | ".join((tmp / "helper_stderr.txt").read_text().splitlines()[-3:])
    kind = "refused by pin_cap" if code == 3 else "failed"
    record["error"] = f"strategy helper {kind} (exit {code}): {message}; no inference run"
    return None, setup_s


def release_helper(proc, tmp, record):
    """Close the helper's stdin so it releases its pages, then read its final report."""
    died_early = proc.poll()
    try:
        proc.stdin.close()
    except OSError:
        pass
    try:
        code = proc.wait(timeout=HELPER_RELEASE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        proc.kill()
        code = proc.wait()
    record["helper_report"] = read_json_file(tmp / "helper_report.json")
    if record.get("error") is None:
        if died_early is not None:
            record["error"] = f"strategy helper exited during inference with {died_early}"
        elif code != 0:
            record["error"] = f"strategy helper exited with {code} on release"


def run(args, prompt_text, record):
    use_cgroup = args.memory_limit != "none"
    use_helper = args.strategy != "baseline"
    record["model_sha256"] = model_sha256(args.model)  # before drop_caches, see model_sha256()
    if args.cold:
        drop_caches()
    if use_cgroup:
        cgroup_setup(args.memory_limit)

    helper = None
    try:
        with tempfile.TemporaryDirectory(prefix="neuroswap_run_") as tmp:
            tmp = Path(tmp)
            setup_s = 0.0
            if use_helper:
                helper, setup_s = start_helper(args, tmp, use_cgroup, record)
                record["strategy_setup_ms"] = round(setup_s * 1000, 3)
                if helper is None:
                    # No inference: keep the usual record shape, with nothing measured.
                    record.update(command=None, system_start=None, wall_s=None, system_end=None,
                                  exit_code=None, rusage=None, samples=[], n_samples=0,
                                  llama_perf=None, total_wall_s=round(setup_s, 3))
                    if use_cgroup:
                        record["cgroup_memory_events"] = cgroup_read_events()
                    return
            else:
                record["strategy_setup_ms"] = 0.0

            prompt_file = tmp / "prompt.txt"
            prompt_file.write_text(prompt_text)
            cmd = [str(args.llama_bin), "-m", str(args.model), "-f", str(prompt_file), "-n", "64",
                   "-c", "2048", "-t", "4", "--seed", "42", "--temp", "0", "-no-cnv"]
            if args.repack == "off":
                cmd.append("--no-repack")
            record["command"] = [c if c != str(prompt_file) else "<prompt_file>" for c in cmd]
            # Generated text goes to a temp file that is deleted with tmp; it echoes the prompt.
            with open(tmp / "stdout.txt", "w") as out_f, \
                 open(tmp / "stderr.txt", "w+") as err_f:
                samples = []
                status = rusage = None
                record["system_start"] = system_snapshot()
                t0 = time.monotonic()
                try:
                    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=out_f,
                                            stderr=err_f,
                                            preexec_fn=join_cgroup if use_cgroup else None)
                    deadline = t0 + args.timeout
                    while True:
                        pid, status, rusage = os.wait4(proc.pid, os.WNOHANG)
                        if pid:
                            break
                        if (s := sample_proc(proc.pid)) is not None:
                            s["t"] = round(s["t"] - t0, 3)
                            samples.append(s)
                        if time.monotonic() > deadline:
                            proc.kill()
                            record["error"] = f"timeout after {args.timeout}s, killed"
                            _, status, rusage = os.wait4(proc.pid, 0)
                            break
                        time.sleep(SAMPLE_INTERVAL_S)
                    proc.returncode = os.waitstatus_to_exitcode(status)
                finally:
                    record["wall_s"] = round(time.monotonic() - t0, 3)
                    record["inference_wall_s"] = record["wall_s"]
                    record["total_wall_s"] = round(setup_s + record["wall_s"], 3)
                    record["system_end"] = system_snapshot()
                    if use_cgroup:
                        record["cgroup_memory_events"] = cgroup_read_events()

                err_f.seek(0)
                stderr_text = err_f.read()

            if helper is not None:
                release_helper(helper, tmp, record)
    finally:
        if helper is not None and helper.poll() is None:
            helper.kill()
            helper.wait()
        if use_cgroup:
            cgroup_teardown()

    record["exit_code"] = proc.returncode
    record["rusage"] = {"ru_minflt": rusage.ru_minflt, "ru_majflt": rusage.ru_majflt,
                        "ru_maxrss_kb": rusage.ru_maxrss,
                        "ru_utime_s": rusage.ru_utime, "ru_stime_s": rusage.ru_stime}
    record["samples"] = samples
    record["n_samples"] = len(samples)
    record["llama_perf"] = parse_perf(stderr_text)
    if record.get("error") is None and proc.returncode != 0:
        if proc.returncode < 0:
            record["error"] = f"killed by signal {signal.Signals(-proc.returncode).name}"
        else:
            record["error"] = f"llama-completion exited with {proc.returncode}"
        events = record.get("cgroup_memory_events") or {}
        if events.get("oom_kill"):
            record["error"] += f" (cgroup oom_kill={events['oom_kill']})"
        record["stderr_tail"] = stderr_text.splitlines()[-5:]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--memory-limit", required=True,
                    help='cgroup memory.max value, e.g. 1200M, or "none"')
    ap.add_argument("--strategy", required=True, choices=STRATEGIES)
    ap.add_argument("--pin-fraction", type=float,
                    help="pin only: fraction (0 < F <= 1) of the per-token weight set to mlock")
    ap.add_argument("--repack", choices=["on", "off"], default="off",
                    help="llama.cpp weight repacking; off (default) passes --no-repack "
                         "so all weights stay file-backed")
    ap.add_argument("--run-index", type=int, required=True)
    ap.add_argument("--cold", action="store_true",
                    help="sync and drop page cache first (root); default whenever --memory-limit "
                         "is not none, where this flag is a no-op alias")
    ap.add_argument("--warm", action="store_true",
                    help="with --memory-limit: skip the default cache drop (the model's cached "
                         "pages may then be charged to another cgroup)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL,
                    help=f"GGUF model (default {DEFAULT_MODEL})")
    ap.add_argument("--llama-bin", type=Path, default=DEFAULT_LLAMA_BIN,
                    help=f"llama-completion binary (default {DEFAULT_LLAMA_BIN})")
    ap.add_argument("--timeout", type=float, default=900, help="seconds before kill")
    args = ap.parse_args()

    if args.memory_limit != "none" and not re.fullmatch(r"\d+[KMG]?|max", args.memory_limit):
        die(f"bad --memory-limit {args.memory_limit!r}; use e.g. 1200M or none")
    if args.cold and args.warm:
        die("--cold and --warm are mutually exclusive")
    # Effective setting: limited runs are cold unless --warm; unlimited runs only with --cold.
    args.cold = (not args.warm) if args.memory_limit != "none" else args.cold
    needs_root = [f for f, on in [("--memory-limit " + args.memory_limit, args.memory_limit != "none"),
                                  ("cold start (drop_caches)", args.cold)] if on]
    if needs_root and os.geteuid() != 0:
        die(f"{' and '.join(needs_root)} require root (cgroup / drop_caches). "
            "Run this command yourself with sudo, using the .venv python.")

    # Fail on missing inputs before any cgroup is created or cache dropped.
    if not args.model.is_file():
        die(f"model not found: {args.model} (pass --model)")
    if not (args.llama_bin.is_file() and os.access(args.llama_bin, os.X_OK)):
        die(f"llama-completion not found or not executable: {args.llama_bin} (pass --llama-bin)")
    if args.strategy == "pin":
        if args.pin_fraction is None or not 0 < args.pin_fraction <= 1:
            die("--strategy pin requires --pin-fraction with 0 < F <= 1")
    elif args.pin_fraction is not None:
        die("--pin-fraction is only valid with --strategy pin")
    if args.strategy != "baseline" and not (HELPER_BIN.is_file() and os.access(HELPER_BIN, os.X_OK)):
        die(f"strategy helper not found or not executable: {HELPER_BIN} (build it with cmake --build build)")

    meta, prompt_text = load_prompt(args.prompt_id)
    record = {
        "source": "neuroswap_measured",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "prompt_id": args.prompt_id,
        "conversation_hash": meta["conversation_hash"],
        "bucket": meta["bucket"],
        "prompt_tokens": int(meta["token_count"]),
        "prompt_format": "raw text, no chat template (-no-cnv)",
        "strategy": args.strategy,
        "repack": args.repack,
        "memory_limit": args.memory_limit,
        "cold": args.cold,
        "run_index": args.run_index,
        "llama_cpp_commit": llama_commit(args.llama_bin),
        "model": args.model.name,
        "cgroup_memory_events": None,  # stays null when --memory-limit none
        "error": None,
        "strategy_setup_ms": None,     # helper setup + residency wait; 0 for baseline
        "helper_report": None,         # neuroswap_strategy's JSON report, unmodified
        "inference_wall_s": None,      # llama-completion only (same as wall_s)
        "total_wall_s": None,          # strategy setup + inference
        "prefetch_chunk_mib": int(PREFETCH_CHUNK_MIB) if args.strategy == "prefetch" else None,
    }
    try:
        run(args, prompt_text, record)
    except Exception as e:  # still write a record, never retry silently
        record["error"] = record.get("error") or f"{type(e).__name__}: {e}"
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(record) + "\n")
        give_back_to_invoking_user([args.out, args.out.parent, SHA_CACHE.parent, SHA_CACHE])
    if record["error"]:
        print(f"run_one: {args.prompt_id} run {args.run_index}: ERROR {record['error']}",
              file=sys.stderr)
        return 1
    perf = record["llama_perf"]
    print(f"run_one: {args.prompt_id} run {args.run_index}: ok, wall {record['wall_s']}s, "
          f"eval {perf['eval_tps']} tok/s, majflt {record['rusage']['ru_majflt']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
