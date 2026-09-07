#!/usr/bin/env python3
"""Ref2AV 速度对照：Gemini 多轮 vs 本地 Omni LoRA（混合路由）。

用法：
  python scripts/bench_ref2av_backends.py
  python scripts/bench_ref2av_backends.py --skip-gemini
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.omni_lora import health as omni_health  # noqa: E402
from src.pipeline import enhance  # noqa: E402

REF_DIR = (
    Path("/kwkj-k8s/zwb/0903qwen-ir/MiniMax-H3-Prompt-Rewriter-LoRA-Omni/assets/examples/ref2av")
)
INTENT = (
    "Create a 10-second podcast recording session. "
    "Use <Picture 1> for the host in an orange patterned shirt and studio background. "
    "Use <Picture 2> for the guest in a red shirt opposite the host. "
    "The host speaks animatedly while the guest listens."
)


def _run_once(backend: str, out_dir: Path) -> dict:
    """跑一条 r2va 并返回 timing 摘要。"""
    t0 = time.perf_counter()
    rec = enhance(
        "r2va",
        INTENT,
        reference_images=[
            str(REF_DIR / "picture_1.jpg"),
            str(REF_DIR / "picture_2.jpg"),
        ],
        duration=10,
        skill_router="hybrid",
        mechanism_router="hybrid",
        enable_verify=False,
        backend=backend,
        resolution="16:9",
        out_dir=out_dir,
    )
    wall = time.perf_counter() - t0
    prompt = rec.get("prompt") or ""
    return {
        "backend": backend,
        "wall_sec": round(wall, 3),
        "timing": rec.get("timing"),
        "omni_lora": rec.get("omni_lora"),
        "schema_ok": (rec.get("verify") or {}).get("schema_ok"),
        "prompt_chars": len(prompt),
        "prompt_preview": prompt[:240],
        "out_dir": str(out_dir),
    }


def main() -> int:
    """对照 gemini / omni_lora 耗时。"""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-gemini", action="store_true", help="只测 Omni（无 Gemini Key 时）")
    p.add_argument("--warmup", action="store_true", help="Omni 先预热一次不计时")
    args = p.parse_args()

    out_root = ROOT / "runs" / f"bench_ref2av_{time.strftime('%Y%m%d_%H%M%S')}"
    out_root.mkdir(parents=True, exist_ok=True)
    results: dict = {"case": "ref2av_podcast", "skill_router": "hybrid", "rows": []}

    try:
        h = omni_health()
        print("omni health:", h)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Omni LoRA 未就绪 ({exc})；请先 cd /kwkj-k8s/zwb/0903qwen-ir && ./serve.sh")
        return 1

    if args.warmup:
        print("warmup omni…")
        _run_once("omni_lora", out_root / "warmup_omni")

    print("=== omni_lora ===")
    omni_row = _run_once("omni_lora", out_root / "omni_lora")
    results["rows"].append(omni_row)
    print(json.dumps(omni_row, ensure_ascii=False, indent=2))

    if not args.skip_gemini:
        print("=== gemini ===")
        try:
            gem_row = _run_once("gemini", out_root / "gemini")
            results["rows"].append(gem_row)
            print(json.dumps(gem_row, ensure_ascii=False, indent=2))
            if gem_row["wall_sec"] > 0:
                speedup = gem_row["wall_sec"] / max(omni_row["wall_sec"], 1e-6)
                results["speedup_omni_vs_gemini"] = round(speedup, 3)
                results["meets_2x"] = bool(speedup >= 2.0)
                print(f"speedup={speedup:.2f}x  meets_2x={results['meets_2x']}")
        except Exception as exc:  # noqa: BLE001
            results["gemini_error"] = str(exc)
            print(f"gemini FAILED: {exc}")

    summary = out_root / "summary.json"
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"summary → {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
