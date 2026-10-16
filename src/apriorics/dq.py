#!/usr/bin/env python3
"""
dq — queue `dvc repro` runs with per-run parameter overrides.

No git refs, no experiments, no checkout, no restore: jobs run one after
another in the current workspace on the current branch. Each job applies
its -S overrides on top of the *current* params files (overrides compound
across the queue) and runs `dvc repro` in place.

With --commit, every run is recorded as a git commit of dvc.lock plus the
params files it touched — git history becomes your run log.

Override syntax (-S, repeatable):
  key=value                  nested keys via dots:  train.lr=0.01
  file.yaml:key=value        target a params file other than params.yaml
  +key=value                 create the key if missing
  ~key                       delete the key
  a=1,b=2                    several assignments in one token
Values: 1  1.5  true/false  null  'text'  [1,2,3]
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

DEFAULT_PARAMS = "params.yaml"

# --- yaml (ruamel.yaml keeps comments/formatting; falls back to PyYAML) -----
try:
    from ruamel.yaml import YAML as _RYAML
    _y = _RYAML()
    _y.preserve_quotes = True
    _y.width = 1 << 16

    def _yload(text: str):
        return _y.load(text) or {}

    def _ydump(data) -> str:
        import io
        buf = io.StringIO()
        _y.dump(data, buf)
        return buf.getvalue()
except ImportError:
    import yaml

    def _yload(text: str):
        return yaml.safe_load(text) or {}

    def _ydump(data) -> str:
        return yaml.safe_dump(data, sort_keys=False)

# --- repo layout ------------------------------------------------------------
def _find_root() -> Path:
    p = Path.cwd().resolve()
    for d in (p, *p.parents):
        if (d / "dvc.yaml").is_file() or (d / ".dvc").is_dir():
            return d
    sys.exit("dq: error: run inside a DVC repo (no dvc.yaml or .dvc found)")

ROOT = _find_root()
STATE = ROOT / ".dq"
QUEUE = STATE / "queue.json"
os.chdir(ROOT)

def now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")

def lock() -> Path:
    STATE.mkdir(exist_ok=True)
    lp = STATE / "run.lock"
    try:
        fd = os.open(lp, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        sys.exit(f"dq: error: {lp} exists — another dq run may be active; delete if stale")
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    return lp

def unlock(lp: Path) -> None:
    lp.unlink(missing_ok=True)

def load_queue() -> dict:
    if not QUEUE.exists():
        return {"version": 2, "jobs": []}
    q = json.loads(QUEUE.read_text())
    q.setdefault("jobs", [])
    return q

def save_queue(q: dict) -> None:
    QUEUE.parent.mkdir(exist_ok=True)
    tmp = STATE / "queue.json.tmp"
    tmp.write_text(json.dumps(q, indent=2, default=str))
    tmp.replace(QUEUE)

# --- override parsing ---------------------------------------------------------
def _split_top(s: str, sep: str = ",") -> list:
    """Split on sep, ignoring separators inside quotes/brackets."""
    parts, buf, depth, quote = [], [], 0, None
    for ch in s:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
            buf.append(ch)
        elif ch in "[{(":
            depth += 1
            buf.append(ch)
        elif ch in "]})":
            depth -= 1
            buf.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    parts.append("".join(buf))
    return parts

def parse_value(raw: str):
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        return raw[1:-1]
    low = raw.lower()
    if low in ("null", "none", "~"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        return [parse_value(p) for p in _split_top(inner)] if inner else []
    return raw

def _is_filename(s: str) -> bool:
    return s.endswith((".yaml", ".yml", ".json", ".toml")) or "/" in s

def _valstr(v) -> str:
    if isinstance(v, str):
        if v == "" or any(c in v for c in ",[]{}'\"") or v != v.strip():
            return "'" + v.replace("'", "\\'") + "'"
        return v
    return json.dumps(v)

def parse_override(token: str) -> list:
    ops = []
    for part in _split_top(token):
        part = part.strip()
        if not part:
            continue
        fname = DEFAULT_PARAMS
        head, sep, rest = part.partition(":")
        if sep and "=" not in head and _is_filename(head):
            fname, part = head, rest
        if part.startswith("~"):
            key = part[1:].strip()
            if not key:
                sys.exit(f"dq: error: bad override {token!r}")
            ops.append({"file": fname, "op": "del", "key": key, "value": None})
            continue
        if part.startswith("+"):
            part = part[1:]
        key, eq, raw = part.partition("=")
        key = key.strip()
        if not eq or not key:
            sys.exit(f"dq: error: overrides look like key=value, got {part!r}")
        ops.append({"file": fname, "op": "set", "key": key, "value": parse_value(raw)})
    if not ops:
        sys.exit(f"dq: error: empty override {token!r}")
    return ops

def fmt_override(o: dict) -> str:
    pfx = "" if o["file"] == DEFAULT_PARAMS else o["file"] + ":"
    if o["op"] == "del":
        return f"{pfx}~{o['key']}" if pfx else f"~{o['key']}"
    return f"{pfx}{o['key']}={_valstr(o['value'])}"

_MISSING = object()

def _get_path(data, keys):
    cur = data
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return _MISSING
        cur = cur[k]
    return cur

def _set_path(data: dict, keys: list, value, create: bool, where: str) -> None:
    cur = data
    for k in keys[:-1]:
        nxt = cur.get(k)
        if nxt is None:
            if not create:
                sys.exit(f"dq: error: {where}: key {k!r} does not exist (prefix with + to create)")
            nxt = {}
            cur[k] = nxt
        if not isinstance(nxt, dict):
            sys.exit(f"dq: error: {where}: {k!r} is not a mapping")
        cur = nxt
    if keys[-1] not in cur and not create:
        sys.exit(f"dq: error: {where}: key {'.'.join(keys)} does not exist (prefix with + to create)")
    cur[keys[-1]] = value

def _del_path(data: dict, keys: list, where: str) -> None:
    cur = data
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            sys.exit(f"dq: error: {where}: no such key {'.'.join(keys)}")
        cur = nxt
    if keys[-1] not in cur:
        sys.exit(f"dq: error: {where}: no such key {'.'.join(keys)}")
    del cur[keys[-1]]

def apply_overrides(overrides: list) -> None:
    """Write overrides into the real params files, in place."""
    by_file: dict = {}
    for ov in overrides:
        by_file.setdefault(ov["file"], []).append(ov)
    for rel, ovs in by_file.items():
        path = ROOT / rel
        data = _yload(path.read_text()) if path.exists() else {}
        for ov in ovs:
            where = fmt_override(ov)
            keys = ov["key"].split(".")
            if ov["op"] == "del":
                _del_path(data, keys, where)
            else:
                _set_path(data, keys, ov["value"], ov["op"] == "add", where)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_ydump(data))

# --- git helpers ---------------------------------------------------------------
def _git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True)

def _tracked(rel: str) -> bool:
    return bool(_git("ls-files", "--", rel).stdout.strip())

def git_commit() -> str | None:
    r = _git("rev-parse", "--short", "HEAD")
    return r.stdout.strip() or None

def _commit_message(job: dict, rc, repro_cmd: list) -> tuple:
    label = job["name"] or ", ".join(fmt_override(o) for o in job["overrides"]) or "baseline"
    if len(label) > 48:
        label = label[:45] + "..."
    subject = f"dq: {label}"
    if rc != 0:
        subject += f" [rc={rc}]"
    lines = [f"job: {job['id']}", f"command: {' '.join(repro_cmd)}"]
    if job["overrides"]:
        lines.append("overrides:")
        lines += [f"  {fmt_override(o)}" for o in job["overrides"]]
    return subject, "\n".join(lines)

def _commit_run(job: dict, rc, repro_cmd: list, extras: list) -> str | None:
    paths = {"dvc.lock", *(ov["file"] for ov in job["overrides"]), *(extras or [])}
    paths = [p for p in sorted(paths) if (ROOT / p).exists() or _tracked(p)]
    if not paths:
        print("dq: warning: nothing to commit (no dvc.lock/params present)", file=sys.stderr)
        return None
    subject, body = _commit_message(job, rc, repro_cmd)
    r = _git("status", "--porcelain", "--", *paths)
    if r.returncode != 0:
        print(f"dq: warning: git status failed: {r.stderr.strip()}", file=sys.stderr)
        return None
    if not r.stdout.strip():
        print("dq: no changes to commit (run was a no-op)")
        return None
    _git("add", "--", *paths)  # pick up newly created files too
    r = _git("commit", "-m", subject, "-m", body, "--", *paths)
    if r.returncode != 0:
        print(f"dq: warning: git commit failed: {(r.stderr or r.stdout).strip()}", file=sys.stderr)
        return None
    sha = git_commit()
    print(f"dq: committed {sha or '?'} — {subject}")
    return sha

# --- misc ---------------------------------------------------------------------
def _params_diff() -> None:
    try:
        r = subprocess.run(["dvc", "params", "diff"], capture_output=True,
                           text=True, timeout=120)
        if r.stdout.strip():
            print(r.stdout.rstrip())
    except Exception:
        pass

def _new_job(name, overrides: list) -> dict:
    return {
        "id": uuid.uuid4().hex[:8],
        "name": name,
        "overrides": overrides,
        "status": "pending",
        "queued_at": now(),
        "started_at": None,
        "ended_at": None,
        "returncode": None,
        "error": None,
        "commit": None,
    }

# --- commands -------------------------------------------------------------------
def cmd_add(args) -> None:
    lp = lock()
    try:
        q = load_queue()
        if args.name and any(j.get("name") == args.name for j in q["jobs"]):
            sys.exit(f"dq: error: job name {args.name!r} already queued")
        overrides = []
        for tok in args.set_param:
            overrides.extend(parse_override(tok))
        job = _new_job(args.name, overrides)
        q["jobs"].append(job)
        save_queue(q)
        ovs = " ".join(fmt_override(o) for o in overrides) or "(no overrides)"
        print(f"dq: queued {job['id']}" + (f" ({job['name']})" if job["name"] else "") + f": {ovs}")
    finally:
        unlock(lp)

def _parse_dim(tok: str):
    fname = DEFAULT_PARAMS
    head, sep, rest = tok.partition(":")
    if sep and "=" not in head and _is_filename(head):
        fname, tok = head, rest
    key, eq, raw = tok.partition("=")
    key = key.strip()
    if not eq or not key:
        sys.exit(f"dq: error: sweep dims look like key=v1,v2,... got {tok!r}")
    if key[0] in "+~":
        sys.exit("dq: error: +/~ not supported in sweep dims")
    values = [parse_value(v) for v in _split_top(raw)]
    if not values:
        sys.exit(f"dq: error: no values for sweep dim {key!r}")
    return fname, key, values

def cmd_sweep(args) -> None:
    if not args.set_param:
        sys.exit("dq: error: sweep needs at least one -S key=v1,v2,...")
    lp = lock()
    try:
        q = load_queue()
        dims = [_parse_dim(t) for t in args.set_param]
        total = 1
        for _, _, vals in dims:
            total *= len(vals)
        if total > args.max_jobs:
            sys.exit(f"dq: error: sweep would create {total} jobs (max {args.max_jobs})")
        for combo in itertools.product(*(vals for _, _, vals in dims)):
            overrides = [{"file": f, "op": "set", "key": k, "value": v}
                         for (f, k, _), v in zip(dims, combo)]
            name = ",".join(f"{k}={_valstr(v)}" for (_, k, _), v in zip(dims, combo))
            base, i = name, 2
            while any(j.get("name") == name for j in q["jobs"]):
                name = f"{base}#{i}"
                i += 1
            q["jobs"].append(_new_job(name, overrides))
        save_queue(q)
        print(f"dq: queued {total} jobs")
    finally:
        unlock(lp)

def _dry(pending: list) -> None:
    """Simulate the whole queue in memory; print old -> new per key."""
    mem: dict = {}

    def load(rel):
        if rel not in mem:
            p = ROOT / rel
            mem[rel] = _yload(p.read_text()) if p.exists() else {}
        return mem[rel]

    for job in pending:
        print(f"\n=== [{job['id']}] {job['name'] or '(unnamed)'}")
        if not job["overrides"]:
            print("    (no overrides)")
            continue
        try:
            for ov in job["overrides"]:
                data = load(ov["file"])
                keys = ov["key"].split(".")
                old = _get_path(data, keys)
                where = fmt_override(ov)
                if ov["op"] == "del":
                    _del_path(data, keys, where)
                    print(f"    {where}: deleted (was {_valstr(old)})")
                else:
                    _set_path(data, keys, ov["value"], ov["op"] == "add", where)
                    if old is _MISSING:
                        print(f"    {where}: (new) = {_valstr(ov['value'])}")
                    else:
                        print(f"    {where}: {_valstr(old)} -> {_valstr(ov['value'])}")
        except SystemExit as e:
            print(f"    would fail: {e}", file=sys.stderr)
    print("\ndq: dry run — nothing was written; jobs stay pending")

def _run_one(job: dict, args, q: dict, repro_cmd: list) -> None:
    job["status"] = "running"
    job["started_at"] = now()
    save_queue(q)

    print(f"\n=== [{job['id']}] {job['name'] or '(unnamed)'}")
    for ov in job["overrides"]:
        print(f"    {fmt_override(ov)}")

    rc, err, sha = None, None, None
    try:
        apply_overrides(job["overrides"])
        if not args.no_diff:
            _params_diff()
        rc = subprocess.run(repro_cmd).returncode
    except KeyboardInterrupt:
        err = "interrupted"
        raise
    except (SystemExit, Exception) as e:
        err = str(e) or type(e).__name__
    finally:
        if err is None and args.commit:
            sha = _commit_run(job, rc, repro_cmd, args.commit_extra)
        job["returncode"] = rc
        job["error"] = err
        job["ended_at"] = now()
        job["commit"] = sha
        job["status"] = "done" if (err is None and rc == 0) else "failed"
        save_queue(q)

def cmd_run(args) -> None:
    if shutil.which("dvc") is None:
        sys.exit("dq: error: dvc not found on PATH")
    if args.commit:
        if shutil.which("git") is None:
            sys.exit("dq: error: git not found on PATH (--commit requires git)")
        r = _git("rev-parse", "--is-inside-work-tree")
        if r.returncode != 0 or r.stdout.strip() != "true":
            sys.exit("dq: error: --commit requires the repo to be inside a git work tree")

    ra = list(args.repro_args or [])
    if ra and ra[0] == "--":
        ra = ra[1:]
    repro_cmd = ["dvc", "repro", *ra]

    lp = lock()
    try:
        q = load_queue()
        for j in q["jobs"]:
            if j["status"] == "running":  # crashed previous invocation
                print(f"dq: re-queueing job {j['id']} left 'running'")
                j["status"] = "pending"
        pending = [j for j in q["jobs"] if j["status"] == "pending"]
        if not pending:
            print("dq: nothing to run (no pending jobs)")
            return
        save_queue(q)

        if args.dry:
            _dry(pending)
            return

        try:
            for job in pending:
                _run_one(job, args, q, repro_cmd)
                if job["status"] == "failed" and not args.keep_going:
                    print("dq: stopping on failure; remaining jobs stay pending "
                          "(--keep-going to override)", file=sys.stderr)
                    break
        except KeyboardInterrupt:
            print("\ndq: interrupted — workspace left exactly as it is; "
                  "remaining jobs stay pending", file=sys.stderr)
        finally:
            save_queue(q)

        counts: dict = {}
        for j in q["jobs"]:
            counts[j["status"]] = counts.get(j["status"], 0) + 1
        summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
        print(f"\ndq: batch finished ({summary}) — see 'dq ls' and 'git log'")
    finally:
        unlock(lp)

def cmd_ls(args) -> None:
    q = load_queue()
    if not q["jobs"]:
        print("dq: queue is empty")
        return
    for j in q["jobs"]:
        ovs = " ".join(fmt_override(o) for o in j["overrides"]) or "(no overrides)"
        name = j["name"] or "-"
        rc = "-" if j.get("returncode") is None else str(j["returncode"])
        sha = j.get("commit") or "-"
        print(f"{j['status']:>8}  {j['id']}  {name:<24} rc={rc:>3}  {sha:<8} {ovs}")
        if j.get("error"):
            print(f"{'':>8}  error: {j['error']}")

def cmd_drop(args) -> None:
    lp = lock()
    try:
        q = load_queue()
        keep, dropped = [], []
        for j in q["jobs"]:
            (dropped if args.job in (j["id"], j["name"]) else keep).append(j)
        if not dropped:
            sys.exit(f"dq: error: no job with id or name {args.job!r}")
        q["jobs"] = keep
        save_queue(q)
        for j in dropped:
            print(f"dq: dropped {j['id']} ({j['name'] or 'unnamed'})")
    finally:
        unlock(lp)

def cmd_clear(args) -> None:
    lp = lock()
    try:
        QUEUE.unlink(missing_ok=True)
        print("dq: queue cleared")
    finally:
        unlock(lp)

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        prog="dq",
        description="Queue `dvc repro` runs with per-run parameter overrides — "
                    "no git refs, no checkout, no restore. Overrides compound "
                    "across the queue; the workspace just moves forward.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("add", help="queue one job")
    p.add_argument("-S", "--set-param", action="append", default=[], metavar="OVR",
                   help="e.g. train.lr=0.01 or configs/model.yaml:depth=8 (repeatable)")
    p.add_argument("-n", "--name", help="job name (used in the commit message)")
    p.set_defaults(fn=cmd_add)

    p = sub.add_parser("sweep", help="queue the cartesian product of comma-separated values")
    p.add_argument("-S", "--set-param", action="append", default=[], metavar="OVR",
                   help="key=v1,v2,... — bare commas are alternatives; bracketed lists "
                        "are single values, e.g. -S layers='[64,128]'")
    p.add_argument("--max-jobs", type=int, default=200)
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("run", help="run all pending jobs sequentially")
    p.add_argument("--commit", action="store_true",
                   help="after every run, git commit dvc.lock + the params files "
                        "this job touched (pathspec-limited commit; failed runs get "
                        "[rc=N] in the subject)")
    p.add_argument("--commit-extra", action="append", default=[], metavar="PATH",
                   help="extra path to include in each run's commit, e.g. metrics.json "
                        "(repeatable)")
    p.add_argument("--keep-going", action="store_true", help="continue after failures")
    p.add_argument("--dry", action="store_true",
                   help="simulate the queue in memory: show old -> new per key, "
                        "don't write anything (jobs stay pending)")
    p.add_argument("--no-diff", action="store_true",
                   help="skip the `dvc params diff` printout before each repro")
    p.add_argument("repro_args", nargs=argparse.REMAINDER, metavar="-- ARGS",
                   help="extra args for dvc repro, e.g.: dq run -- -v")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("ls", help="list jobs (with commit shas once run)")
    p.set_defaults(fn=cmd_ls)

    p = sub.add_parser("drop", help="remove a job by id or name")
    p.add_argument("job")
    p.set_defaults(fn=cmd_drop)

    p = sub.add_parser("clear", help="drop all queued jobs")
    p.set_defaults(fn=cmd_clear)

    args = ap.parse_args(argv)
    args.fn(args)

if __name__ == "__main__":
    main()
