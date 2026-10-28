#!/usr/bin/env python3
"""perf/EXP-0001 orchestrator. Frozen plan_commit d6e3a243. Do not edit production code."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(os.environ.get("ROOT", "/data/sgl_pr/runs/perf/EXP-0001"))
WT = Path(os.environ.get("WT", "/data/sgl_pr/worktrees/perf"))
PY = os.environ.get("PY", "/opt/sglang/bin/python3")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "30000"))
EXPECT_SHA = "faa56e2b5c5c5dee60eb5dd7b6251944a21a6104"
PLAN_COMMIT = "d6e3a243de880df22194457397d382906773ff0f"
MODEL = "/data1/models/DeepSeek-V4.1-Flash"
SCRIPT_DIR = Path(__file__).resolve().parent
READY_TIMEOUT_S = 30 * 60
STALL_S = 10 * 60

CASES = [
    {"id": "A", "isl": 8192, "osl": 1, "c": 1, "n": 8, "warmup": 2, "kind": "prefill"},
    {"id": "B", "isl": 20480, "osl": 1, "c": 1, "n": 4, "warmup": 2, "kind": "prefill"},
    {"id": "C", "isl": 2048, "osl": 256, "c": 1, "n": 8, "warmup": 2, "kind": "decode"},
    {"id": "D", "isl": 2048, "osl": 256, "c": 8, "n": 32, "warmup": 16, "kind": "decode"},
]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def step(n: int, msg: str) -> None:
    log(f"step={n} {msg}")


def run(cmd: List[str], cwd: Optional[str] = None, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    log("+ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, timeout=timeout, text=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def http_json(url: str, method: str = "GET", body: Optional[dict] = None, timeout: float = 60.0) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            code = resp.status
    except urllib.error.HTTPError as e:
        raw = e.read()
        code = e.code
    except Exception as e:
        return 0, {"error": str(e)}
    if not raw:
        return code, {}
    try:
        return code, json.loads(raw.decode())
    except Exception:
        return code, {"raw": raw[:4000].decode("utf-8", "replace")}


def pick_port() -> int:
    for p in [30000, 30010, 30020, 30030, 30100]:
        r = subprocess.run(["ss", "-lntp"], capture_output=True, text=True)
        if f":{p}" not in r.stdout:
            return p
    raise RuntimeError("no free port")


def gpus_busy() -> bool:
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
    )
    used = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line:
            used.append(int(float(line)))
    write_json(ROOT / "identity" / "gpu_mem_used.json", {"used_mib": used, "at": now()})
    return any(v > 256 for v in used)


def record_identity() -> Dict[str, Any]:
    step(1, "record_identity")
    ident_dir = ROOT / "identity"
    ident_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.check_output(["git", "-C", str(WT), "rev-parse", "HEAD"], text=True).strip()
    porcelain = subprocess.check_output(["git", "-C", str(WT), "status", "--porcelain"], text=True)
    live_sha = subprocess.check_output(
        ["git", "-C", "/sgl-workspace/sglang", "rev-parse", "HEAD"], text=True
    ).strip()
    (ident_dir / "git_status.txt").write_text(porcelain)
    env = {
        k: os.environ.get(k)
        for k in sorted(os.environ)
        if k.upper().startswith(("SGLANG", "NCCL", "CUDA", "FLASH", "DEEPGEMM"))
        or k in ("PYTHONPATH", "http_proxy", "https_proxy")
    }
    py_info = subprocess.check_output(
        [
            PY,
            "-c",
            "import sys,torch,os; os.environ['CUDA_VISIBLE_DEVICES']='9999'; import sgl_kernel,flashinfer;\n"
            "print('exe', sys.executable); print('torch', torch.__version__, torch.version.cuda);\n"
            "print('sgl_kernel', getattr(sgl_kernel,'__version__','?'));\n"
            "print('flashinfer', getattr(flashinfer,'__version__','?'));\n"
            "print('path0', sys.path[0])",
        ],
        env={**os.environ, "PYTHONPATH": str(WT / "python"), "CUDA_VISIBLE_DEVICES": "9999"},
        text=True,
    )
    (ident_dir / "python_packages.txt").write_text(py_info)
    smi = subprocess.check_output(["nvidia-smi"], text=True)
    (ident_dir / "nvidia-smi.txt").write_text(smi)
    drv = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version,name,compute_cap", "--format=csv"],
        text=True,
    )
    (ident_dir / "gpu.csv").write_text(drv)
    import_file = subprocess.check_output(
        [
            PY,
            "-c",
            "import sglang; print(sglang.__file__)",
        ],
        env={**os.environ, "PYTHONPATH": str(WT / "python"), "CUDA_VISIBLE_DEVICES": "9999"},
        text=True,
    ).strip()
    ident = {
        "recorded_at": now(),
        "plan_commit": PLAN_COMMIT,
        "code_sha": sha,
        "expect_sha": EXPECT_SHA,
        "live_sha": live_sha,
        "worktree": str(WT),
        "pythonpath": str(WT / "python"),
        "import_sglang": import_file,
        "python": PY,
        "model": MODEL,
        "dirty": bool(porcelain.strip()),
        "env_subset": env,
        "hostname": subprocess.check_output(["hostname"], text=True).strip(),
    }
    write_json(ident_dir / "identity.json", ident)
    if sha != EXPECT_SHA:
        raise SystemExit(f"SHA mismatch {sha}")
    if porcelain.strip():
        raise SystemExit("worktree dirty")
    if live_sha == sha and "/sgl-workspace/sglang/python" in import_file:
        raise SystemExit("importing live tree")
    if not import_file.startswith(str(WT)):
        raise SystemExit(f"import not from worktree: {import_file}")
    return ident


def wait_ready(port: int) -> bool:
    url = f"http://{HOST}:{port}/health"
    t0 = time.time()
    last = ""
    while time.time() - t0 < READY_TIMEOUT_S:
        code, body = http_json(url, timeout=5)
        last = f"code={code} body={body}"
        if code == 200:
            log(f"health 200 after {time.time()-t0:.0f}s")
            return True
        elapsed = time.time() - t0
        if int(elapsed) % 30 < 2:
            log(f"waiting health {elapsed:.0f}s {last}")
        time.sleep(5)
    log(f"ready timeout {last}")
    return False


def dump_server_info(port: int, dest: Path) -> Dict[str, Any]:
    code, body = http_json(f"http://{HOST}:{port}/server_info", timeout=120)
    write_json(dest, {"http_code": code, "body": body, "at": now()})
    return body if isinstance(body, dict) else {}


def bench_cmd(case: dict, rundir: Path, port: int, profile: bool, extra: List[str]) -> List[str]:
    cmd = [
        PY,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang",
        "--base-url",
        f"http://{HOST}:{port}",
        "--model",
        MODEL,
        "--tokenizer",
        MODEL,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(case["isl"]),
        "--random-output-len",
        str(case["osl"]),
        "--random-range-ratio",
        "0",
        "--seed",
        "42",
        "--temperature",
        "0",
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(case["c"]),
        "--num-prompts",
        str(case["n"] if not profile else extra_n(case)),
        "--warmup-requests",
        str(case["warmup"] if not profile else extra_warmup(case)),
        "--tokenize-prompt",
        "--flush-cache",
        "--output-details",
        "--disable-tqdm",
        "--output-file",
        str(rundir / "bench.json"),
    ]
    if profile:
        cmd += extra
    return cmd


def extra_n(case: dict) -> int:
    if case["id"] == "D":
        return 8
    return 1


def extra_warmup(case: dict) -> int:
    if case["kind"] == "prefill":
        return 10
    if case["id"] == "D":
        return 16
    return 10


def profile_flags(case: dict, rundir: Path) -> List[str]:
    out = rundir / "traces"
    out.mkdir(parents=True, exist_ok=True)
    if case["kind"] == "prefill":
        nsteps = 8
    else:
        nsteps = 16
    return [
        "--profile",
        "--profile-activities",
        "CPU",
        "GPU",
        "--profile-by-stage",
        "--profile-num-steps",
        str(nsteps),
        "--profile-output-dir",
        str(out),
        "--profile-prefix",
        f"exp0001-{case['id']}",
    ]


def run_bench(case: dict, rundir: Path, port: int, profile: bool) -> Dict[str, Any]:
    rundir.mkdir(parents=True, exist_ok=True)
    extra = profile_flags(case, rundir) if profile else []
    cmd = bench_cmd(case, rundir, port, profile, extra)
    (rundir / "cmd.txt").write_text(" ".join(cmd) + "\n")
    logf = rundir / "client.log"
    t0 = time.time()
    env = {
        **os.environ,
        "PYTHONPATH": str(WT / "python"),
        "SGLANG_PROFILE_V2": "0",
    }
    with logf.open("w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        last_size = 0
        last_change = time.time()
        while True:
            rc = proc.poll()
            sz = logf.stat().st_size if logf.exists() else 0
            if sz != last_size:
                last_size = sz
                last_change = time.time()
            if rc is not None:
                break
            if time.time() - last_change > STALL_S:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise RuntimeError(f"stalled {STALL_S}s in {rundir}")
            time.sleep(5)
    elapsed = time.time() - t0
    if profile:
        try:
            http_json(f"http://{HOST}:{port}/stop_profile", method="POST", timeout=180)
        except Exception as e:
            log(f"stop_profile after client: {e}")
    summary = {"rc": proc.returncode, "elapsed_s": elapsed, "cmd": cmd}
    bench_path = rundir / "bench.json"
    if bench_path.exists():
        try:
            lines = [ln for ln in bench_path.read_text().splitlines() if ln.strip()]
            summary["bench"] = json.loads(lines[-1]) if lines else {}
        except Exception as e:
            summary["bench_parse_error"] = str(e)
    write_json(rundir / "summary.json", summary)
    return summary


def metric_from_summary(summary: Dict[str, Any], case: dict) -> Dict[str, Any]:
    b = summary.get("bench") or {}
    completed = b.get("completed", b.get("completed_requests"))
    n = case["n"]
    ok = summary.get("rc") == 0 and isinstance(b, dict) and completed is not None and int(completed) == int(n)
    out = {
        "ok": bool(ok),
        "output_throughput": b.get("output_throughput"),
        "median_tpot_ms": b.get("median_tpot_ms"),
        "median_ttft_ms": b.get("median_ttft_ms"),
        "mean_tpot_ms": b.get("mean_tpot_ms"),
        "completed": completed,
        "num_prompts": n,
        "error": None if ok else "bench_failed_or_incomplete",
    }
    return out


def noise_ratio(vals: List[float]) -> Optional[float]:
    good = [v for v in vals if v is not None]
    if len(good) < 3:
        return None
    mid = sorted(good)[len(good) // 2]
    if mid == 0:
        return None
    return (max(good) - min(good)) / mid


def list_traces(rundir: Path) -> List[Path]:
    traces = list((rundir / "traces").rglob("*.trace.json.gz"))
    traces += list((rundir / "traces").rglob("*.trace.json"))
    return sorted(traces)


def trace_has_gpu(path: Path) -> bool:
    import gzip

    opener = gzip.open if path.suffix == ".gz" or path.name.endswith(".json.gz") else open
    try:
        with opener(path, "rt") as f:
            # scan first 8MB of text for kernel cats to avoid full parse of huge traces
            chunk = f.read(8 * 1024 * 1024)
        keys = ('"cat": "kernel"', '"cat":"kernel"', "gpu_memcpy", "gpu_memset", "cuda_runtime")
        if any(k in chunk for k in keys):
            return True
        # fallback full parse if small
        with opener(path, "rt") as f:
            data = json.load(f)
        events = data if isinstance(data, list) else data.get("traceEvents", [])
        for e in events:
            cat = str(e.get("cat") or "")
            if cat in ("kernel", "gpu_memcpy", "gpu_memset", "cuda_runtime", "gpu_user_annotation"):
                return True
        return False
    except Exception as e:
        log(f"trace_has_gpu {path}: {e}")
        return False


def validate_profile(case: dict, rundir: Path) -> Dict[str, Any]:
    traces = list_traces(rundir)
    by_rank = {}
    for p in traces:
        name = p.name
        gpu = trace_has_gpu(p)
        rec = {"path": str(p), "gpu": gpu, "size": p.stat().st_size}
        by_rank.setdefault(name, rec)
    tp = [p for p in traces if "TP-" in p.name]
    extend = [p for p in traces if "EXTEND" in p.name or "PREFILL" in p.name]
    decode = [p for p in traces if "DECODE" in p.name]
    ranks = set()
    for p in traces:
        for part in p.name.split("-"):
            if part.startswith("TP"):
                ranks.add(part)
    report = {
        "n_traces": len(traces),
        "n_extend": len(extend),
        "n_decode": len(decode),
        "ranks": sorted(ranks),
        "files": [str(p.relative_to(rundir)) for p in traces],
        "gpu_ok": all(trace_has_gpu(p) for p in traces) if traces else False,
    }
    if case["kind"] == "prefill":
        report["valid"] = report["gpu_ok"] and len(extend) >= 8 and len(ranks) >= 8
        report["need"] = "8-rank EXTEND/prefill GPU traces"
    else:
        report["valid"] = report["gpu_ok"] and len(decode) >= 8 and len(ranks) >= 8
        report["need"] = "8-rank DECODE GPU traces"
    write_json(rundir / "profile_validate.json", report)
    return report


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    step(0, "start EXP-0001")
    status: Dict[str, Any] = {
        "plan_commit": PLAN_COMMIT,
        "started_at": now(),
        "state": "running",
        "stop_reason": None,
        "cases": {},
    }
    write_json(ROOT / "status.json", status)

    try:
        ident = record_identity()
        if gpus_busy():
            status["state"] = "invalid"
            status["stop_reason"] = "launch_blocked_foreign_gpu"
            write_json(ROOT / "status.json", status)
            log("STOP: GPUs occupied by another service")
            return 2

        global PORT
        PORT = pick_port()
        os.environ["PORT"] = str(PORT)
        os.environ["HOST"] = HOST
        os.environ["ROOT"] = str(ROOT)
        ident["endpoint"] = f"http://{HOST}:{PORT}"
        write_json(ROOT / "identity" / "identity.json", ident)

        step(2, f"launch serve host={HOST} port={PORT}")
        r = run(["bash", str(SCRIPT_DIR / "serve.sh")])
        if r.returncode != 0:
            status["state"] = "invalid"
            status["stop_reason"] = f"serve_script_rc_{r.returncode}"
            write_json(ROOT / "status.json", status)
            return 3

        step(3, "wait_ready")
        if not wait_ready(PORT):
            status["state"] = "invalid"
            status["stop_reason"] = "serve_not_ready_30min"
            write_json(ROOT / "status.json", status)
            run(["bash", str(SCRIPT_DIR / "stop_serve.sh")])
            return 4

        info = dump_server_info(PORT, ROOT / "serve" / "server_info.json")
        write_json(ROOT / "serve" / "effective_args.json", info.get("server_args") or info)

        # timing
        s = 10
        for case in CASES:
            cid = case["id"]
            status["cases"].setdefault(cid, {"timing": [], "profile": []})
            metrics = []
            for rep in range(1, 4):
                rundir = ROOT / cid / f"timing_r{rep}"
                step(s, f"timing {cid} r{rep}")
                s += 1
                try:
                    summary = run_bench(case, rundir, PORT, profile=False)
                except Exception as e:
                    status["state"] = "invalid"
                    status["stop_reason"] = f"timing_{cid}_r{rep}_{e}"
                    write_json(ROOT / "status.json", status)
                    run(["bash", str(SCRIPT_DIR / "stop_serve.sh")])
                    return 5
                m = metric_from_summary(summary, case)
                m["run"] = f"timing_r{rep}"
                metrics.append(m)
                status["cases"][cid]["timing"].append(m)
                write_json(ROOT / "status.json", status)
                if not m["ok"]:
                    status["state"] = "invalid"
                    status["stop_reason"] = f"timing_request_error_{cid}_r{rep}"
                    write_json(ROOT / "status.json", status)
                    run(["bash", str(SCRIPT_DIR / "stop_serve.sh")])
                    return 6
            thr = [m["output_throughput"] for m in metrics]
            tpot = [m["median_tpot_ms"] for m in metrics]
            nr_thr = noise_ratio(thr)
            nr_tpot = noise_ratio(tpot) if case["kind"] == "decode" else None
            noisy = (nr_thr is not None and nr_thr > 0.10) or (
                case["kind"] == "decode" and nr_tpot is not None and nr_tpot > 0.10
            )
            status["cases"][cid]["noise_throughput"] = nr_thr
            status["cases"][cid]["noise_tpot"] = nr_tpot
            if noisy:
                log(f"{cid} noise >10%, extra 2 rounds")
                for rep in range(4, 6):
                    rundir = ROOT / cid / f"timing_r{rep}"
                    step(s, f"timing {cid} extra r{rep}")
                    s += 1
                    summary = run_bench(case, rundir, PORT, profile=False)
                    m = metric_from_summary(summary, case)
                    m["run"] = f"timing_r{rep}"
                    metrics.append(m)
                    status["cases"][cid]["timing"].append(m)
                thr = [m["output_throughput"] for m in metrics if m.get("output_throughput") is not None]
                tpot = [m["median_tpot_ms"] for m in metrics if m.get("median_tpot_ms") is not None]
                nr_thr = noise_ratio(thr)
                nr_tpot = noise_ratio(tpot) if case["kind"] == "decode" else None
                status["cases"][cid]["noise_throughput"] = nr_thr
                status["cases"][cid]["noise_tpot"] = nr_tpot
                still = (nr_thr is not None and nr_thr > 0.10) or (
                    case["kind"] == "decode" and nr_tpot is not None and nr_tpot > 0.10
                )
                status["cases"][cid]["noise_unconverged"] = still
            write_json(ROOT / "status.json", status)

        # profiles
        for case in CASES:
            cid = case["id"]
            attempts = ["profile_r1", "profile_r2"]
            ok = False
            for i, tag in enumerate(attempts):
                rundir = ROOT / cid / tag
                step(s, f"profile {cid} {tag}")
                s += 1
                try:
                    run_bench(case, rundir, PORT, profile=True)
                except Exception as e:
                    log(f"profile exception {e}")
                    (rundir / "error.txt").write_text(traceback.format_exc())
                report = validate_profile(case, rundir)
                status["cases"][cid]["profile"].append(report)
                write_json(ROOT / "status.json", status)
                if report.get("valid"):
                    ok = True
                    break
                if i == 0:
                    log(f"{cid} profile invalid, retry once")
            status["cases"][cid]["profile_ok"] = ok

        status["state"] = "captured"
        status["finished_at"] = now()
        write_json(ROOT / "status.json", status)
        step(90, "stop serve")
        run(["bash", str(SCRIPT_DIR / "stop_serve.sh")])
        step(100, "capture complete")
        return 0
    except Exception:
        status["state"] = "invalid"
        status["stop_reason"] = "orchestrator_exception"
        status["traceback"] = traceback.format_exc()
        write_json(ROOT / "status.json", status)
        log(status["traceback"])
        run(["bash", str(SCRIPT_DIR / "stop_serve.sh")])
        return 1


if __name__ == "__main__":
    sys.exit(main())
