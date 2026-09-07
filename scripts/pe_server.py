#!/usr/bin/env python3
"""Prompt Enhancement (PE) HTTP service: enhance prompts without generating video."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import enhance  # noqa: E402
from src.runlog import activate, deactivate, write_meta  # noqa: E402
from src.skill import ALL_MODES  # noqa: E402

MAX_BODY_BYTES = 1_000_000
ROUTER_MODES = {"off", "keyword", "hybrid", "llm"}


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} 必须是字符串数组")
    return [item.strip() for item in value if item.strip()]


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} 必须是字符串")
    return value.strip() or None


def execute_pe(payload: dict[str, Any], *, out_root: Path, log_root: Path) -> dict[str, Any]:
    """Validate one PE request, run the enhancer, and return an API-friendly record."""
    if not isinstance(payload, dict):
        raise ValueError("请求体必须是 JSON 对象")
    mode = str(payload.get("mode") or "t2va").strip().lower()
    if mode not in ALL_MODES:
        raise ValueError(f"mode 须为 {' / '.join(ALL_MODES)}")
    intent = str(payload.get("intent") or "").strip()
    if not intent:
        raise ValueError("intent 不能为空")

    duration_raw = payload.get("duration")
    duration = None if duration_raw is None else int(duration_raw)
    if duration is not None and not 4 <= duration <= 15:
        raise ValueError("duration 须为 4–15 秒")
    enable_verify = payload.get("verify", True)
    if not isinstance(enable_verify, bool):
        raise ValueError("verify 必须是 JSON boolean")
    verify_intent_llm = payload.get("verify_intent_llm")
    if verify_intent_llm is not None and not isinstance(verify_intent_llm, bool):
        raise ValueError("verify_intent_llm 必须是 JSON boolean")

    profile = str(payload.get("profile") or "fast").strip().lower()
    if profile not in {"fast", "quality"}:
        raise ValueError("profile 须为 fast / quality")
    default_router = "keyword" if profile == "fast" else "hybrid"
    skill_router = str(payload.get("skill_router") or default_router).strip().lower()
    mechanism_router = str(payload.get("mechanism_router") or default_router).strip().lower()
    if skill_router not in ROUTER_MODES:
        raise ValueError("skill_router 须为 off / keyword / hybrid / llm")
    if mechanism_router not in ROUTER_MODES:
        raise ValueError("mechanism_router 须为 off / keyword / hybrid / llm")

    run_id = f"pe_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    out_dir = Path(out_root) / run_id
    log_dir = Path(log_root) / run_id
    started = time.perf_counter()
    activate(log_dir)
    try:
        rec = enhance(
            mode,
            intent,
            first_frame=_optional_string(payload.get("first_frame"), "first_frame"),
            last_frame=_optional_string(payload.get("last_frame"), "last_frame"),
            reference_images=_string_list(payload.get("reference_images"), "reference_images") or None,
            reference_videos=_string_list(payload.get("reference_videos"), "reference_videos") or None,
            reference_audios=_string_list(payload.get("reference_audios"), "reference_audios") or None,
            duration=duration,
            out_dir=out_dir,
            skills=_string_list(payload.get("skills"), "skills") or None,
            skill_router=skill_router,
            mechanisms=_string_list(payload.get("mechanisms"), "mechanisms") or None,
            mechanism_router=mechanism_router,
            enable_verify=enable_verify,
            verify_intent_llm=verify_intent_llm,
        )
        elapsed = time.perf_counter() - started
        api_calls = len(list(log_dir.glob("*_request.txt")))
        result = {
            "request_id": run_id,
            "mode": mode,
            "profile": profile,
            "intent": intent,
            "duration": rec.get("duration"),
            "prompt": rec.get("prompt") or "",
            "contract": rec.get("contract") or {},
            "verify": rec.get("verify") or {},
            "metrics": {"elapsed_sec": round(elapsed, 3), "api_calls": api_calls},
            "artifacts": {"out_dir": str(out_dir), "log_dir": str(log_dir)},
        }
        if payload.get("include_intermediates") is True:
            result["expanded"] = rec.get("expanded") or ""
            result["elaborated"] = rec.get("elaborated") or ""
        write_meta({
            "request_id": run_id,
            "mode": mode,
            "intent": intent,
            "metrics": result["metrics"],
            "out_dir": str(out_dir),
            "log_dir": str(log_dir),
        })
        return result
    except Exception as exc:
        write_meta({"request_id": run_id, "mode": mode, "intent": intent, "error": str(exc)})
        raise
    finally:
        deactivate()


class PEHandler(BaseHTTPRequestHandler):
    out_root = ROOT / "runs" / "pe"
    log_root = ROOT / "log" / "pe"
    server_version = "ContextIR-PE/1.0"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        expected = os.environ.get("PE_API_KEY", "")
        return not expected or secrets.compare_digest(
            self.headers.get("Authorization", ""), f"Bearer {expected}"
        )

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "service": "pe"})
        else:
            self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/v1/pe", "/v1/enhance"):
            self._send_json(404, {"error": "not_found", "message": f"不支持的端点: {self.path}"})
            return
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError("请求体为空或超过 1 MB")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            result = execute_pe(payload, out_root=self.out_root, log_root=self.log_root)
            self._send_json(200, result)
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": "bad_request", "message": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": "pe_failed", "message": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[pe] {self.address_string()} {fmt % args}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="ContextIR Prompt Enhancement HTTP service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs" / "pe")
    parser.add_argument("--log-root", type=Path, default=ROOT / "log" / "pe")
    args = parser.parse_args()
    PEHandler.out_root = args.out_root
    PEHandler.log_root = args.log_root
    server = ThreadingHTTPServer((args.host, args.port), PEHandler)
    print(f"PE listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
