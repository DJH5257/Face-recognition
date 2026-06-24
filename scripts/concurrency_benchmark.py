#!/usr/bin/env python3
"""并发压测脚本：用真实核验路径估算单实例承载能力。

使用方式（依赖 backend/app 已经准备好测试模板和 enrollment）：

    python scripts/concurrency_benchmark.py \
        --base-url http://127.0.0.1:8000 \
        --internal-api-key change-me-face-internal-key \
        --subject-type DOCTOR --subject-id doc-001 --scene attendance \
        --template-id <tpl> --frames-fixture tests/fixtures/frames.json \
        --concurrencies 5 10 15 --repeats 1

如果只想测槽位是否会排队（不发真实人脸帧），可以加 `--probe-only`，
脚本会直接打 /v1/internal/slot-probe（仅占用核验槽位，不跑模型），用来
观察排队与 503 行为，不会真的触发模型推理。

目标只有一个：让你知道当前机器在 5/10/15 并发下：成功多少、失败多少、
平均耗时多少、有没有触发 503。绝对不会"压垮"服务（最大并发与服务端一致）。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CallResult:
    ok: bool
    status: int
    elapsed_ms: float
    detail: str = ""
    body: Optional[dict] = None


def _post_json(url: str, headers: dict, payload: dict, timeout: float) -> CallResult:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    req.add_header("Content-Type", "application/json")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            raw = resp.read().decode("utf-8", errors="replace")
            body = json.loads(raw) if raw else {}
            return CallResult(ok=True, status=resp.status, elapsed_ms=elapsed_ms, body=body)
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return CallResult(ok=False, status=exc.code, elapsed_ms=elapsed_ms, detail=body[:160])
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return CallResult(ok=False, status=0, elapsed_ms=elapsed_ms, detail=str(exc)[:160])


def _get_json(url: str, headers: dict, timeout: float) -> CallResult:
    req = urllib.request.Request(url, method="GET")
    for k, v in headers.items():
        req.add_header(k, v)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            return CallResult(ok=True, status=resp.status, elapsed_ms=elapsed_ms)
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return CallResult(ok=False, status=exc.code, elapsed_ms=elapsed_ms, detail=str(exc)[:160])
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return CallResult(ok=False, status=0, elapsed_ms=elapsed_ms, detail=str(exc)[:160])


def run_round(make_call, concurrency: int) -> List[CallResult]:
    results: List[CallResult] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(make_call, idx) for idx in range(concurrency)]
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


def summarize(results: List[CallResult]) -> dict:
    total = len(results)
    ok = [r for r in results if r.ok]
    fail = [r for r in results if not r.ok]
    busy = [r for r in fail if r.status == 503]
    latencies = [r.elapsed_ms for r in results]
    summary = {
        "total": total,
        "success": len(ok),
        "fail": len(fail),
        "busy_503": len(busy),
        "p50_ms": round(statistics.median(latencies), 1) if latencies else 0,
        "avg_ms": round(statistics.mean(latencies), 1) if latencies else 0,
        "max_ms": round(max(latencies), 1) if latencies else 0,
    }
    # 抽样列出失败原因，便于排查
    samples = [
        {"status": r.status, "detail": r.detail}
        for r in fail[:3]
    ]
    if samples:
        summary["fail_samples"] = samples
    return summary


def fetch_stats(base_url: str, headers: dict) -> Optional[dict]:
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/v1/internal/runtime-stats")
        for k, v in headers.items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Face verify concurrency benchmark")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--internal-api-key", default="")
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--repeats", type=int, default=1, help="每个并发量重复几轮")
    parser.add_argument("--timeout", type=float, default=120.0, help="单次请求超时秒")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="只打 /v1/internal/slot-probe，用于观察排队语义，不触发真实推理",
    )
    parser.add_argument("--probe-hold-ms", type=int, default=250, help="probe-only 每个请求占用槽位毫秒数")
    parser.add_argument("--subject-type", default="")
    parser.add_argument("--subject-id", default="")
    parser.add_argument("--scene", default="")
    parser.add_argument("--frames-fixture", default="", help="包含 frames 列表的 JSON 文件路径")
    args = parser.parse_args(argv)

    base = args.base_url.rstrip("/")
    internal_headers = {"X-Face-Api-Key": args.internal_api_key} if args.internal_api_key else {}

    if args.probe_only:
        if not internal_headers:
            print("probe-only 模式需要 --internal-api-key", file=sys.stderr)
            return 2

        def make_probe(_idx: int) -> CallResult:
            return _post_json(
                base + "/v1/internal/slot-probe",
                internal_headers,
                {"hold_ms": args.probe_hold_ms},
                args.timeout,
            )

        run_func = make_probe
    else:
        for required in ("subject_type", "subject_id", "scene", "frames_fixture"):
            if not getattr(args, required):
                print(
                    "完整核验压测需要 --subject-type/--subject-id/--scene/--frames-fixture",
                    file=sys.stderr,
                )
                return 2
        with open(args.frames_fixture, "r", encoding="utf-8") as fp:
            frames_payload = json.load(fp)
        frames = frames_payload.get("frames") if isinstance(frames_payload, dict) else frames_payload
        if not isinstance(frames, list) or not frames:
            print("--frames-fixture 必须是 frames 列表或包含 frames 字段的 JSON", file=sys.stderr)
            return 2

        def make_full(_idx: int) -> CallResult:
            # 1) 创建核验会话
            request_id = f"bench-{_idx}-{int(time.time()*1000)}"
            session_payload = {
                "request_id": request_id,
                "subject_type": args.subject_type,
                "subject_id": args.subject_id,
                "scene": args.scene,
                "business_event_id": request_id,
            }
            r1 = _post_json(
                base + "/v1/internal/verification-sessions",
                internal_headers,
                session_payload,
                args.timeout,
            )
            if not r1.ok:
                return r1
            session = r1.body or {}
            session_id = session.get("session_id")
            upload_token = session.get("upload_token")
            if not session_id or not upload_token:
                return CallResult(
                    ok=False,
                    status=0,
                    elapsed_ms=r1.elapsed_ms,
                    detail="create session response missing session_id/upload_token",
                )
            verify_headers = {"Authorization": "Bearer " + upload_token}
            return _post_json(
                base + "/v1/verification-sessions/" + session_id + "/verify",
                verify_headers,
                {"frames": frames},
                args.timeout,
            )

        run_func = make_full

    print(f"==> base_url={base}, probe_only={args.probe_only}")
    pre = fetch_stats(base, internal_headers)
    if pre is not None:
        print(f"    runtime-stats(before): {pre}")

    for concurrency in args.concurrencies:
        for round_idx in range(args.repeats):
            start = time.perf_counter()
            results = run_round(run_func, concurrency)
            elapsed = (time.perf_counter() - start) * 1000.0
            summary = summarize(results)
            summary["concurrency"] = concurrency
            summary["round"] = round_idx + 1
            summary["wallclock_ms"] = round(elapsed, 1)
            print(json.dumps(summary, ensure_ascii=False))

    post = fetch_stats(base, internal_headers)
    if post is not None:
        print(f"    runtime-stats(after): {post}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
