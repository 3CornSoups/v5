#!/usr/bin/env python3
"""本地 Qwen2.5-Omni-7B + H3 LoRA 改写客户端（HTTP → :8910）。"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any

import requests

from .config import CONFIGS, load_yaml

REFERENCE_LABEL_RE = re.compile(
    r"<?\b(Picture|Video|Audio)\s+(\d+)\b>?", re.IGNORECASE
)

_MODE_TO_TASK = {
    "t2va": "t2av",
    "t2av": "t2av",
    "i2va": "i2av",
    "i2av": "i2av",
    "l2va": "l2av",
    "l2av": "l2av",
    "fl2va": "fl2av",
    "fl2av": "fl2av",
    "r2va": "ref2av",
    "ref2va": "ref2av",
    "ref2av": "ref2av",
}


def omni_lora_settings() -> dict[str, Any]:
    """合并 omni_lora.yaml 与环境变量。"""
    path = CONFIGS / "omni_lora.yaml"
    cfg: dict[str, Any] = {}
    if path.is_file():
        cfg = load_yaml("omni_lora")
    base = (
        os.environ.get("OMNI_LORA_BASE_URL")
        or str(cfg.get("base_url") or "http://127.0.0.1:8910")
    ).rstrip("/")
    return {
        "base_url": base,
        "timeout_sec": float(
            os.environ.get("OMNI_LORA_TIMEOUT_SEC") or cfg.get("timeout_sec") or 600
        ),
        "default_resolution": str(cfg.get("default_resolution") or "16:9"),
        "served_model_name": str(
            cfg.get("served_model_name") or "qwen2.5-omni-7b-h3-lora"
        ),
    }


def map_mode_to_task(mode: str) -> str:
    """V5 mode → Omni LoRA task。"""
    key = (mode or "").strip().lower()
    if key not in _MODE_TO_TASK:
        raise ValueError(f"不支持的 mode: {mode}")
    return _MODE_TO_TASK[key]


def build_references(
    task: str,
    *,
    first_frame: str | None = None,
    last_frame: str | None = None,
    reference_images: list[str] | None = None,
    reference_videos: list[str] | None = None,
    reference_audios: list[str] | None = None,
) -> list[dict[str, str]]:
    """把 V5 媒体参数整理为 Omni references 列表（绝对路径）。"""
    refs: list[dict[str, str]] = []

    def _add(kind: str, path: str) -> None:
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"参考媒体不存在: {p}")
        refs.append({"type": kind, "path": str(p)})

    if task == "i2av":
        if not first_frame:
            raise ValueError("i2av 需要 first_frame")
        _add("image", first_frame)
    elif task == "l2av":
        if not last_frame:
            raise ValueError("l2av 需要 last_frame")
        _add("image", last_frame)
    elif task == "fl2av":
        if not first_frame or not last_frame:
            raise ValueError("fl2av 需要 first_frame 与 last_frame")
        _add("image", first_frame)
        _add("image", last_frame)
    elif task == "ref2av":
        for p in reference_images or []:
            _add("image", p)
        for p in reference_videos or []:
            _add("video", p)
        for p in reference_audios or []:
            _add("audio", p)
        if not refs:
            raise ValueError("ref2av 至少需要一张图或一段视频")
    return refs


def ensure_ref_labels_in_prompt(prompt: str, references: list[dict[str, str]]) -> str:
    """Ref2AV：若用户意图未点名标签，自动补全 Use <Picture N> … 语句。"""
    if not references:
        return prompt
    counts = {"Picture": 0, "Video": 0, "Audio": 0}
    labels: list[str] = []
    for ref in references:
        kind = ref["type"]
        prefix = {"image": "Picture", "video": "Video", "audio": "Audio"}[kind]
        counts[prefix] += 1
        labels.append(f"<{prefix} {counts[prefix]}>")

    mentioned = set()
    for m in REFERENCE_LABEL_RE.finditer(prompt):
        mentioned.add(f"<{m.group(1).capitalize()} {int(m.group(2))}>")
    missing = [lb for lb in labels if lb not in mentioned]
    if not missing:
        return prompt
    lines = [
        prompt.strip(),
        "",
        "Reference usage (must retain identity / composition from each asset):",
    ]
    for lb, ref in zip(labels, references):
        lines.append(f"- Use {lb} from file {Path(ref['path']).name} ({ref['type']}).")
    return "\n".join(lines)


def resolve_resolution(task: str, resolution: str | None, default: str = "16:9") -> str:
    """归一化分辨率；Ref2AV 仅允许 16:9 / 9:16。"""
    ratio = (resolution or default or "16:9").strip()
    if task == "ref2av" and ratio not in {"16:9", "9:16"}:
        return "16:9" if ratio != "9:16" else ratio
    if task == "t2av" and ratio == "adaptive":
        return "16:9"
    return ratio


def health(timeout: float = 3.0) -> dict[str, Any]:
    """探测本地 Omni LoRA 服务。"""
    cfg = omni_lora_settings()
    url = f"{cfg['base_url']}/health"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def rewrite(
    *,
    mode: str,
    intent: str,
    duration: int,
    first_frame: str | None = None,
    last_frame: str | None = None,
    reference_images: list[str] | None = None,
    reference_videos: list[str] | None = None,
    reference_audios: list[str] | None = None,
    resolution: str | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """调用本地 Omni LoRA `/v1/rewrite`，返回含 enhanced_prompt 与耗时的结果。"""
    cfg = omni_lora_settings()
    task = map_mode_to_task(mode)
    refs = build_references(
        task,
        first_frame=first_frame,
        last_frame=last_frame,
        reference_images=reference_images,
        reference_videos=reference_videos,
        reference_audios=reference_audios,
    )
    prompt = intent.strip()
    if task == "ref2av":
        prompt = ensure_ref_labels_in_prompt(prompt, refs)
    ratio = resolve_resolution(task, resolution, cfg["default_resolution"])
    body: dict[str, Any] = {
        "task": task,
        "prompt": prompt,
        "duration": int(duration),
        "resolution": ratio,
        "references": refs,
    }
    if request_id:
        body["id"] = request_id

    url = f"{cfg['base_url']}/v1/rewrite"
    t0 = time.perf_counter()
    resp = requests.post(url, json=body, timeout=cfg["timeout_sec"])
    latency = time.perf_counter() - t0
    if resp.status_code >= 400:
        raise RuntimeError(f"Omni LoRA HTTP {resp.status_code}: {resp.text[:800]}")
    data = resp.json()
    data["client_latency_sec"] = round(latency, 3)
    data["backend"] = "omni_lora"
    data["endpoint"] = url
    return data
