#!/usr/bin/env python3
"""Regression test cho OCR auto-detect — chạy auto trên tất cả sample.

Mục đích: tránh "fix lỗi này lòi ra lỗi khác" — mỗi lần đổi engine.py /
paragraph_merger.py / main.py thì chạy script này để verify không regression
trên các case đã từng được fix.

Usage:
    # Chạy regression test, so sánh với baseline (nếu có):
    python3 scripts/test_ocr_samples.py
    python3 scripts/test_ocr_samples.py --api http://127.0.0.1:9003

    # Sau khi đã verify thủ công kết quả OK, đặt baseline mới:
    python3 scripts/test_ocr_samples.py --update-baseline

Output ghi vào `logs/ocr_regression/`:
    <timestamp>.json: snapshot full result của run này.
    latest.json: copy của run mới nhất.
    baseline.json: golden output (manually pinned) — để diff.
    summary.log: append-only text log mọi run (timestamp + diff summary).

Diff rules (flag REGRESSION nếu):
    - detected_lang đổi.
    - reading_order đổi.
    - n_blocks chênh > 30% so với baseline.
    - full_text hash khác (text recognized đổi đáng kể).
Cảnh báo (WARN) nếu n_blocks chênh 10–30%.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "ocr_service" / "samples_ocr"
LOGS_DIR = ROOT / "logs" / "ocr_regression"
BASELINE_PATH = LOGS_DIR / "baseline.json"
LATEST_PATH = LOGS_DIR / "latest.json"
SUMMARY_LOG = LOGS_DIR / "summary.log"

DEFAULT_API = "http://127.0.0.1:9003"
SAMPLE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
HTTP_TIMEOUT_S = 60


# ANSI color (skip nếu stdout không phải tty — log file readable)
def _c(code: str) -> str:
    return code if sys.stdout.isatty() else ""


C_RED = _c("\033[31m")
C_GREEN = _c("\033[32m")
C_YELLOW = _c("\033[33m")
C_BLUE = _c("\033[34m")
C_DIM = _c("\033[2m")
C_BOLD = _c("\033[1m")
C_RST = _c("\033[0m")


def _post_multipart(api: str, file_path: Path) -> dict:
    """POST /v1/ocr/upload với multipart/form-data (manual encode để không cần requests dep)."""
    boundary = f"----vbk{int(time.time()*1000)}"
    body_parts: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        body_parts.append(f"--{boundary}\r\n".encode())
        body_parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body_parts.append(value.encode())
        body_parts.append(b"\r\n")

    def add_file(name: str, path: Path) -> None:
        body_parts.append(f"--{boundary}\r\n".encode())
        body_parts.append(
            f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'.encode()
        )
        body_parts.append(b"Content-Type: application/octet-stream\r\n\r\n")
        body_parts.append(path.read_bytes())
        body_parts.append(b"\r\n")

    add_file("file", file_path)
    add_field("lang", "auto")
    body_parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(body_parts)

    url = api.rstrip("/") + "/v1/ocr/upload"
    req = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode())


def _summarize_response(resp: dict) -> dict:
    """Trích summary deterministic để diff (bỏ field bay-bổng như request_id)."""
    blocks = resp.get("blocks") or []
    full_text = resp.get("full_text") or ""
    n_lines = sum(len(b.get("lines") or []) for b in blocks)
    n_words = sum(
        len(line.get("words") or [])
        for b in blocks
        for line in (b.get("lines") or [])
    )
    block_texts = [b.get("text", "") for b in blocks]
    block_avg_conf = (
        sum(b.get("confidence", 0.0) for b in blocks) / len(blocks)
        if blocks else 0.0
    )
    return {
        "detected_lang": resp.get("detected_lang"),
        "reading_order": resp.get("reading_order"),
        "image_size": [resp.get("image_width"), resp.get("image_height")],
        "n_blocks": len(blocks),
        "n_lines": n_lines,
        "n_words": n_words,
        "processing_time_ms": resp.get("processing_time_ms"),
        "avg_confidence": round(block_avg_conf, 3),
        "full_text_hash": hashlib.md5(full_text.encode("utf-8")).hexdigest()[:12],
        "full_text_preview": full_text[:120],
        "block_texts": block_texts,
    }


def _list_samples() -> list[Path]:
    if not SAMPLES_DIR.exists():
        sys.exit(f"[ERR] Samples dir không tồn tại: {SAMPLES_DIR}")
    samples = sorted(
        p for p in SAMPLES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in SAMPLE_EXTS
    )
    if not samples:
        sys.exit(f"[ERR] Không có sample nào trong {SAMPLES_DIR}")
    return samples


def _diff_against_baseline(current: dict, baseline: dict | None) -> tuple[list[str], list[str]]:
    """Return (regressions, warnings). Empty = no regression."""
    regressions: list[str] = []
    warnings: list[str] = []
    if baseline is None:
        return regressions, warnings

    cur_results: dict[str, dict] = {r["filename"]: r for r in current["results"]}
    base_results: dict[str, dict] = {r["filename"]: r for r in baseline["results"]}

    for fname, cur in cur_results.items():
        base = base_results.get(fname)
        if base is None:
            warnings.append(f"  {fname}: NEW sample (no baseline)")
            continue
        if "error" in cur:
            regressions.append(f"  {fname}: ERROR — {cur['error']}")
            continue
        if "error" in base:
            warnings.append(f"  {fname}: baseline có error, hiện tại OK")
            continue

        cs, bs = cur["summary"], base["summary"]
        if cs["detected_lang"] != bs["detected_lang"]:
            regressions.append(
                f"  {fname}: detected_lang {bs['detected_lang']!r} → {cs['detected_lang']!r}"
            )
        if cs["reading_order"] != bs["reading_order"]:
            regressions.append(
                f"  {fname}: reading_order {bs['reading_order']!r} → {cs['reading_order']!r}"
            )
        if cs["full_text_hash"] != bs["full_text_hash"]:
            regressions.append(
                f"  {fname}: full_text_hash {bs['full_text_hash']} → {cs['full_text_hash']}"
            )

        # n_blocks tolerance
        b_blocks = bs["n_blocks"]
        c_blocks = cs["n_blocks"]
        if b_blocks > 0:
            ratio = abs(c_blocks - b_blocks) / b_blocks
            if ratio > 0.30:
                regressions.append(
                    f"  {fname}: n_blocks {b_blocks} → {c_blocks} (chênh {ratio*100:.0f}%)"
                )
            elif ratio > 0.10:
                warnings.append(
                    f"  {fname}: n_blocks {b_blocks} → {c_blocks} (chênh {ratio*100:.0f}%)"
                )

    # Sample bị xoá
    for fname in base_results:
        if fname not in cur_results:
            warnings.append(f"  {fname}: REMOVED — không còn trong samples_ocr")

    return regressions, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default=os.environ.get("OCR_API", DEFAULT_API),
                        help=f"OCR API endpoint (default: {DEFAULT_API})")
    parser.add_argument("--update-baseline", action="store_true",
                        help="Sau khi chạy, ghi đè baseline bằng kết quả lần này")
    parser.add_argument("--quiet", action="store_true",
                        help="Chỉ in summary cuối, bỏ log per-sample")
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    samples = _list_samples()

    print(f"{C_BOLD}OCR Regression Test{C_RST}")
    print(f"  API:     {args.api}")
    print(f"  Samples: {len(samples)} file ({SAMPLES_DIR})")
    print(f"  Logs:    {LOGS_DIR}")
    print()

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    results: list[dict] = []
    total_ms = 0

    col_w = max(len(s.name) for s in samples) + 2
    if not args.quiet:
        print(
            f"  {'sample':{col_w}}  {'lang':10s} {'order':5s} "
            f"{'blocks':>6s} {'lines':>6s} {'words':>6s} {'ms':>6s}  preview"
        )
        print(f"  {'-' * (col_w + 60)}")

    for sample in samples:
        entry: dict = {"filename": sample.name}
        try:
            t0 = time.time()
            resp = _post_multipart(args.api, sample)
            elapsed_ms = int((time.time() - t0) * 1000)
            summary = _summarize_response(resp)
            summary["client_total_ms"] = elapsed_ms
            entry["summary"] = summary
            total_ms += elapsed_ms
            if not args.quiet:
                preview = summary["full_text_preview"].replace("\n", " ")[:50]
                print(
                    f"  {sample.name:{col_w}}  "
                    f"{(summary['detected_lang'] or '?'):10s} "
                    f"{(summary['reading_order'] or '?'):5s} "
                    f"{summary['n_blocks']:>6d} "
                    f"{summary['n_lines']:>6d} "
                    f"{summary['n_words']:>6d} "
                    f"{summary['processing_time_ms']:>6d}  "
                    f"{preview!r}"
                )
        except (HTTPError, URLError, TimeoutError) as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  {C_RED}{sample.name}: ERROR — {entry['error']}{C_RST}")
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  {C_RED}{sample.name}: ERROR — {entry['error']}{C_RST}")
        results.append(entry)

    finished_at = dt.datetime.now(dt.timezone.utc).isoformat()
    snapshot = {
        "started_at": started_at,
        "finished_at": finished_at,
        "api": args.api,
        "n_samples": len(samples),
        "total_client_ms": total_ms,
        "results": results,
    }

    # Ghi snapshot có timestamp
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snap_path = LOGS_DIR / f"{ts}.json"
    snap_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    LATEST_PATH.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")

    # Diff với baseline
    baseline = None
    if BASELINE_PATH.exists():
        try:
            baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"{C_YELLOW}[WARN] Không đọc được baseline: {exc}{C_RST}")
    regressions, warnings = _diff_against_baseline(snapshot, baseline)

    print()
    print(f"{C_BOLD}Summary:{C_RST}")
    print(f"  Total time: {total_ms} ms (avg {total_ms // max(1, len(samples))} ms / sample)")
    print(f"  Snapshot:   {snap_path}")
    if baseline is None:
        print(f"  {C_DIM}Chưa có baseline.json — chạy với --update-baseline để pin.{C_RST}")
    else:
        if regressions:
            print(f"  {C_RED}REGRESSION ({len(regressions)}):{C_RST}")
            for r in regressions:
                print(f"  {C_RED}{r}{C_RST}")
        if warnings:
            print(f"  {C_YELLOW}WARN ({len(warnings)}):{C_RST}")
            for w in warnings:
                print(f"  {C_YELLOW}{w}{C_RST}")
        if not regressions and not warnings:
            print(f"  {C_GREEN}OK — không regression, khớp baseline.{C_RST}")

    # Append summary.log
    with SUMMARY_LOG.open("a", encoding="utf-8") as f:
        f.write(f"\n=== {finished_at} ===\n")
        f.write(f"api={args.api} samples={len(samples)} total_ms={total_ms}\n")
        f.write(f"snapshot={snap_path.name}\n")
        if baseline is not None:
            f.write(f"regressions={len(regressions)} warnings={len(warnings)}\n")
            for r in regressions:
                f.write(f"REGRESSION:{r}\n")
            for w in warnings:
                f.write(f"WARN:{w}\n")
        else:
            f.write("baseline=NONE\n")

    if args.update_baseline:
        shutil.copy(snap_path, BASELINE_PATH)
        print(f"  {C_GREEN}Baseline updated: {BASELINE_PATH}{C_RST}")

    # Exit code: 1 nếu có regression, 0 nếu OK
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
