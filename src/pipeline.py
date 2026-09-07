"""T2VA / I2VA / FL2VA / L2VA / R2VA 多步 Gemini 编排；格式化注入官方 skill 指南。"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .complexity import format_complexity_budget_block
from .config import ROOT, gemini_settings, h3_settings, load_prompt
from .contract import (
    extract_dialogue_lines,
    parse_intent,
    parse_intent_deterministic,
    resolve_dialogue_speakers,
)
from .gemini import chat
from .media import user_parts
from .omni_lora import rewrite as omni_lora_rewrite
from .report import write_report
from .skill import (
    ALL_MODES,
    GRID_SCAN_INSTRUCTION,
    KEYFRAME_MODES,
    compose_format_system,
    ensure_alignment_prefix,
    expand_hint,
    grid_coverage_gap,
    grid_keep_subjects_note,
)
from .mechanism_router import (
    ROUTER_MODES as MECHANISM_ROUTER_MODES,
    mechanism_block_for_user,
    select_mechanisms,
    writing_blocks_for_user,
)
from .skill_router import ROUTER_MODES, select_style_skills, style_block_for_user
from .verify import verify_and_fix
from .video import generate_video

PE_BACKENDS = ("gemini", "omni_lora")

CANVAS_RE = re.compile(
    r"(?:,\s*)?(?:aspect ratio|canvas size|分辨率|帧率|画幅)\s*[:=为是]?\s*"
    r"(?:16:9|9:16|21:9|4:3|3:4|360P|480P|540P|720P|768P|1080P|2K|4K|"
    r"\d+x\d+|\d+(?:\.\d+)?\s*fps)|"
    r"(?:16:9|9:16|21:9|4:3|3:4)[ \t]*(?:aspect ratio|横屏|竖屏)?|"
    r"\b(?:360P|480P|540P|720P|768P|1080P|2K|4K|1280x720|1920x1080|"
    r"\d+(?:\.\d+)?[ \t]*fps)\b",
    re.I,
)
DURATION_RE = re.compile(r"(?:约|大概)?\s*(\d{1,2})\s*秒")


def infer_duration(intent: str, fallback: int = 5) -> int:
    """从短意图里的「约 N 秒」推断时长；夹到 4–15。"""
    m = DURATION_RE.search(intent or "")
    n = int(m.group(1)) if m else fallback
    return max(4, min(15, n))


def strip_canvas(text: str) -> str:
    """去掉误写入字段的画幅/分辨率/帧率，保留段落换行结构。"""
    cleaned = CANVAS_RE.sub(" ", text)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip() + "\n"


def _locked_dialogue_block(intent: str, contract: Any = None) -> str | None:
    """把用户意图里的真实台词与屏上文字列成锁定清单，供扩写/补细节/格式化原句抄写。"""
    if contract is not None:
        spoken = list(getattr(contract, "dialogue", []) or [])
        onscreen = list(getattr(contract, "onscreen_text", []) or [])
    else:
        from .contract import extract_dialogue_lines
        from .verify import extract_locked_onscreen
        spoken = extract_dialogue_lines(intent)
        onscreen = extract_locked_onscreen(intent)

    parts: list[str] = []
    if spoken:
        body = "\n".join(
            f"- #{idx} speaker_id={line.speaker_id or 'UNKNOWN'}; "
            f"speaker={line.speaker or 'UNKNOWN'}; language={line.language or 'UNKNOWN'}; "
            f"speech_type={getattr(line, 'speech_type', 'dialogue')}; "
            f"speaker_visible={str(getattr(line, 'speaker_visible', True)).lower()}; "
            f"lip_sync={getattr(line, 'lip_sync', 'required')}; "
            f"character_match={getattr(line, 'character_match', '') or 'NONE'}; "
            f"exact_text={json.dumps(line.text, ensure_ascii=False)}"
            for idx, line in enumerate(spoken, 1)
        )
        parts.append(
            "Locked spoken lines / VOICE CONTRACT — emit exactly one <d>[Language] exact_text</d> block per numbered line, "
            "in this order and with the same punctuation. This numbered list is ONE immutable global audible-event timeline "
            "across all shots and across dialogue, voiceover/inner thought, and narration. The Nth <d> block in the complete output "
            "must be entry #N; never group or reorder events by speaker, speech_type, character, shot, dramatic importance, or audio asset. "
            "Use the required speaker_id immediately before its block. Do not translate, paraphrase, deduplicate, merge, split, or add dialogue. "
            "The JSON quotation marks surrounding exact_text are metadata delimiters, not spoken characters: "
            "never copy those wrapper quotes into <d> unless quotation marks are actually part of exact_text. "
            "Honor speech_type exactly: dialogue requires visible lip sync; voiceover uses the exact phrase "
            "'says in an off-screen voiceover' and keeps the matched character's lips completely closed; "
            "narration keeps the narrator off-screen, creates no visible narrator, and gives no character lip sync. "
            "Never use [Mandarin]; Chinese uses [Chinese], English uses [English].\n"
            + body
        )
    if onscreen:
        body = "\n".join(f"- {line}" for line in onscreen)
        parts.append(
            "Locked on-screen lines from the user's intent. Keep each line verbatim as on-screen text "
            "in the original language. Do not translate or invent extra captions.\n"
            + body
        )
    if not parts:
        return None
    return "\n\n".join(parts)


def _expand_user(
    intent: str,
    *,
    inventory: str | None = None,
    mode: str,
    writing_block: str | None = None,
    contract_block: str | None = None,
    complexity_block: str | None = None,
    contract: Any = None,
) -> str:
    """构造扩写 USER：短意图 + Intent Contract + 官方模式写作路径 + 可选库存与写法块。"""
    lines = [
        f"Mode: {mode}. Expand the short intent. Do not output MiniMax fields yet.",
        f"Writing path: {expand_hint(mode)}",
        "",
        "Short intent:",
        intent.strip(),
    ]
    if contract_block:
        lines.extend(["", contract_block.strip()])
    if complexity_block:
        lines.extend(["", complexity_block.strip()])
    locked = _locked_dialogue_block(intent, contract=contract)
    if locked:
        lines.extend(["", locked])
    if inventory:
        lines.extend(["", "Reference inventory:", inventory.strip()])
        keep = grid_keep_subjects_note(inventory)
        if keep:
            lines.extend(["", keep])
    if writing_block:
        lines.extend(["", writing_block.rstrip()])
    return "\n".join(lines)


def _format_user(
    mode: str,
    scene: str,
    *,
    inventory: str | None,
    duration: int | None,
    intent: str = "",
    contract_block: str | None = None,
    contract: Any = None,
) -> str:
    """构造共用格式化 USER：模式 + Intent Contract + 场景稿 + 可选库存与时长约束。"""
    lines = [
        f"MODE={mode}",
        "Serialize the scene note into the MiniMax-H3 fields for this MODE.",
        "Follow the appended official writing guide. Do not mention aspect ratio, resolution, fps, or canvas size.",
    ]
    if duration is not None:
        lines.append(
            f"Duration hint: {duration:g} seconds. Keep cut timestamps inside this length. "
            "Do not write the duration into the core fields. "
            f"If MODE is fl2va or l2va, the alignment line MUST use S.SS = {float(duration):.2f}."
        )
    if contract_block:
        lines.extend(["", contract_block.strip()])
    lines.extend(["", "Scene note:", scene.strip()])
    locked = _locked_dialogue_block(intent, contract=contract)
    if locked:
        lines.extend(["", locked])
    if inventory:
        lines.extend(["", "Reference inventory:", inventory.strip()])
        keep = grid_keep_subjects_note(inventory)
        if keep:
            lines.extend(["", keep])
    return "\n".join(lines)


def _append_grid_scan(text: str) -> str:
    """在感知 USER 文本末尾加上宫格扫全说明。"""
    return text.rstrip() + "\n\n" + GRID_SCAN_INSTRUCTION


def _rescan_if_grid_incomplete(
    system: str,
    inventory: str,
    *,
    text: str,
    images: list[str] | None,
    videos: list[str] | None = None,
    audios: list[str] | None = None,
) -> tuple[str, str | None]:
    """宫格声明与格子笔记不一致时再扫一次；返回 (库存, 补扫阶段名或 None)。"""
    gap = grid_coverage_gap(inventory)
    if not gap:
        return inventory, None
    follow = (
        f"{text}\n\nPrevious inventory (incomplete):\n{inventory.strip()}\n\n{gap}"
    )
    scanned = chat(
        system,
        user_parts(follow, images=images, videos=videos, audios=audios),
        stage="perceive",
    )
    return scanned, "perceive_grid_rescan"


def _perceive_keyframes(
    mode: str,
    *,
    first_frame: str | None,
    last_frame: str | None,
    duration: int,
) -> tuple[str, str | None]:
    """对 I2VA/FL2VA/L2VA 的静帧做事实库存；宫格漏格时补扫。"""
    system = load_prompt("perceive_image")
    if mode == "i2va":
        text = (
            "Mode: I2VA. Attached image is <Picture 1>, the FIRST frame at 0.00s / [Shot 1]. "
            "Describe visible facts only."
        )
        images = [first_frame] if first_frame else None
    elif mode == "fl2va":
        text = (
            "Mode: FL2VA. Two images in order:\n"
            "<Picture 1> = FIRST frame at 0.00s (attached first).\n"
            f"<Picture 2> = LAST frame at {float(duration):.2f}s (attached second).\n"
            "Describe each image separately. Do not invent the path between them."
        )
        images = [p for p in (first_frame, last_frame) if p]
    else:
        text = (
            "Mode: L2VA. Attached image is <Picture 1>, the LAST frame of the clip "
            f"(lands at about {float(duration):.2f}s). It does NOT belong to Shot 1. "
            "Describe the landing state only."
        )
        images = [last_frame] if last_frame else None
    text = _append_grid_scan(text)
    inventory = chat(system, user_parts(text, images=images), stage="perceive")
    return _rescan_if_grid_incomplete(system, inventory, text=text, images=images)


def _elaborate_user(
    expanded: str,
    inventory: str | None,
    intent: str = "",
    *,
    contract_block: str | None = None,
    complexity_block: str | None = None,
    contract: Any = None,
) -> str:
    """构造补细节 USER：Intent Contract + 复杂度预算 + 扩写稿 + 可选库存。"""
    lines = [
        "Make the scene note below concrete and physically plausible. "
        "Match detail depth to the COMPLEXITY BUDGET when provided; do not pad a simple "
        "single-shot clip beyond its budget, and do not undershoot a multi-beat scene."
    ]
    if contract_block:
        lines.extend(["", contract_block.strip()])
    if complexity_block:
        lines.extend(["", complexity_block.strip()])
    if inventory:
        lines.extend(["", "Reference inventory:", inventory.strip()])
    lines.extend(["", "Scene note:", expanded.strip()])
    locked = _locked_dialogue_block(intent, contract=contract)
    if locked:
        lines.extend(["", locked])
    return "\n".join(lines)


def _resolve_pe_backend(backend: str | None) -> str:
    """解析 PE 后端：参数 / 环境变量 PE_BACKEND / 默认 gemini。"""
    import os

    raw = (backend or os.environ.get("PE_BACKEND") or "gemini").strip().lower()
    if raw not in PE_BACKENDS:
        raise ValueError(f"backend 须为 {' / '.join(PE_BACKENDS)}，收到 {backend!r}")
    return raw


def _enhance_omni_lora(
    mode: str,
    intent: str,
    *,
    first_frame: str | None,
    last_frame: str | None,
    images: list[str],
    videos: list[str],
    audios: list[str],
    duration: int,
    out_dir: Path | None,
    skills: list[str] | None,
    skill_router: str,
    mechanisms: list[str] | None,
    mechanism_router: str,
    resolution: str | None = None,
) -> dict[str, Any]:
    """Omni LoRA 单次改写快路径：跳过多轮 Gemini，保留 hybrid 关键词路由元数据。

    目标场景：混合路由下 Ref2AV（r2va）推理加速；质量依赖 LoRA 改写效果。
    """
    import time

    t0 = time.perf_counter()
    steps: list[dict[str, Any]] = []

    # hybrid：仅关键词路由（不打 LLM），记录命中 skill/mechanism，不注入多轮扩写。
    router_mode = (skill_router or "hybrid").strip().lower()
    mech_router_mode = (mechanism_router or "hybrid").strip().lower()
    if router_mode not in ROUTER_MODES:
        raise ValueError(f"skill_router 须为 {' / '.join(ROUTER_MODES)}")
    if mech_router_mode not in MECHANISM_ROUTER_MODES:
        raise ValueError(f"mechanism_router 须为 {' / '.join(MECHANISM_ROUTER_MODES)}")

