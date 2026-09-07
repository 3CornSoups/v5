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

    style_sel = select_style_skills(
        intent,
        inventory=None,
        forced=skills,
        router="keyword" if router_mode in {"hybrid", "llm"} else router_mode,
        classify=None,
    )
    mech_sel = select_mechanisms(
        intent,
        inventory=None,
        forced=mechanisms,
        router="keyword" if mech_router_mode in {"hybrid", "llm"} else mech_router_mode,
        classify=None,
    )
    steps.append(
        {
            "stage": "skill_route",
            "source": f"omni_lora+{style_sel.source}",
            "skills": style_sel.ids,
            "scores": style_sel.scores,
            "threshold": style_sel.threshold,
            "note": "omni_lora 快路径下 hybrid 降为 keyword，避免额外 LLM 往返",
        }
    )
    if mech_sel.ids:
        steps.append(
            {
                "stage": "mechanism_route",
                "text": f"source=omni_lora+{mech_sel.source}; mechanisms={', '.join(mech_sel.ids)}",
            }
        )

    omni = omni_lora_rewrite(
        mode=mode,
        intent=intent,
        duration=duration,
        first_frame=first_frame,
        last_frame=last_frame,
        reference_images=images or None,
        reference_videos=videos or None,
        reference_audios=audios or None,
        resolution=resolution,
    )
    prompt = (omni.get("enhanced_prompt") or "").strip()
    if not prompt:
        raise RuntimeError("Omni LoRA 返回空 enhanced_prompt")
    prompt = ensure_alignment_prefix(mode, strip_canvas(prompt), duration)
    steps.append(
        {
            "stage": "omni_lora_rewrite",
            "task": omni.get("task"),
            "schema_ok": omni.get("schema_ok"),
            "server_latency_sec": omni.get("latency_sec"),
            "client_latency_sec": omni.get("client_latency_sec"),
            "text": prompt,
        }
    )

    total_sec = round(time.perf_counter() - t0, 3)
    record: dict[str, Any] = {
        "mode": mode,
        "backend": "omni_lora",
        "intent": intent,
        "duration": duration,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "reference_images": images if mode == "r2va" else [],
        "i2va_first_frame": first_frame if mode == "i2va" else None,
        "reference_videos": videos,
        "reference_audios": audios,
        "inventory": None,
        "contract": None,
        "expanded": None,
        "elaborated": None,
        "style_skills": style_sel.ids,
        "style_skill_source": f"omni_lora+{style_sel.source}",
        "style_skill_scores": style_sel.scores,
        "style_skill_threshold": style_sel.threshold,
        "mechanisms": mech_sel.ids,
        "mechanism_source": f"omni_lora+{mech_sel.source}",
        "prompt_official": prompt,
        "prompt": prompt,
        "verify": {
            "status": "skipped",
            "fixed": False,
            "issues": [],
            "schema_ok": bool(omni.get("schema_ok")),
        },
        "omni_lora": {
            "schema_ok": omni.get("schema_ok"),
            "latency_sec": omni.get("latency_sec"),
            "client_latency_sec": omni.get("client_latency_sec"),
            "endpoint": omni.get("endpoint"),
            "effective_duration": omni.get("effective_duration"),
            "references": omni.get("references"),
        },
        "timing": {"total_sec": total_sec, "http_calls": 1},
        "steps": steps,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
        (out_dir / "run.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        record["out_dir"] = str(out_dir)
    return record


def enhance(
    mode: str,
    intent: str,
    *,
    first_frame: str | None = None,
    last_frame: str | None = None,
    reference_images: list[str] | None = None,
    reference_videos: list[str] | None = None,
    reference_audios: list[str] | None = None,
    duration: int | None = None,
    out_dir: Path | None = None,
    skills: list[str] | None = None,
    skill_router: str = "hybrid",
    mechanisms: list[str] | None = None,
    mechanism_router: str = "hybrid",
    enable_verify: bool = True,
    verify_intent_llm: bool | None = None,
    backend: str | None = None,
    resolution: str | None = None,
) -> dict[str, Any]:
    """
    跑完感知（若需要）→ 风格/机制路由 → 扩写 → 补细节 → 注入官方指南后格式化。

    skills: 强制加载的风格 skill id。
    skill_router: off / keyword / hybrid / llm。
    hybrid / llm：前置模型为各 skill 打 0~1 分，仅加载 >= match_threshold（默认 0.8）；
    keyword：仍可只用触发词；off：只用强制 id。
    mechanisms: 强制加载的 T8 Creative DNA 机制 id。
    mechanism_router: 机制路由模式，默认同 skill_router（机制侧仍为关键词优先 hybrid）。
    backend: gemini（默认多轮）| omni_lora（本地 Omni+LoRA 单次改写）。

    Returns:
        含 prompt、各步原文、mode
    """
    mode = mode.lower().strip()
    if mode not in ALL_MODES:
        raise ValueError(f"mode 须为 {' / '.join(ALL_MODES)}")
    intent = (intent or "").strip()
    if not intent:
        raise ValueError("短意图为空")

    images = list(reference_images or [])
    videos = list(reference_videos or [])
    audios = list(reference_audios or [])
    if mode == "i2va" and not first_frame:
        raise ValueError("i2va 需要 --first-frame")
    if mode == "fl2va" and (not first_frame or not last_frame):
        raise ValueError("fl2va 需要同时提供 --first-frame 与 --last-frame")
    if mode == "l2va" and not last_frame:
        raise ValueError("l2va 需要 --last-frame")
    if mode == "r2va":
        if not images and not videos:
            raise ValueError("r2va 须至少 1 张参考图或 1 段参考视频")
        if len(images) > 9:
            raise ValueError("r2va 参考图数量 ≤ 9")
        if len(videos) > 3:
            raise ValueError("r2va 参考视频数量 ≤ 3")
        if len(audios) > 3:
            raise ValueError("r2va 参考音频数量 ≤ 3")

    pe_backend = _resolve_pe_backend(backend)
    steps: list[dict[str, Any]] = []
    dur = duration if duration is not None else infer_duration(intent)
    inventory: str | None = None

    if pe_backend == "omni_lora":
        return _enhance_omni_lora(
            mode,
            intent,
            first_frame=first_frame,
            last_frame=last_frame,
            images=images,
            videos=videos,
            audios=audios,
            duration=dur,
            out_dir=out_dir,
            skills=skills,
            skill_router=skill_router,
            mechanisms=mechanisms,
            mechanism_router=mechanism_router,
            resolution=resolution,
        )

    import time as _time

    _t0 = _time.perf_counter()

    if mode in KEYFRAME_MODES:
        inventory, rescan = _perceive_keyframes(
            mode,
            first_frame=first_frame,
            last_frame=last_frame,
            duration=dur,
        )
        steps.append({"stage": "perceive_image", "text": inventory})
        if rescan:
            steps.append({"stage": rescan, "text": inventory})
    elif mode == "r2va":
        labels = []
        for i, p in enumerate(images, 1):
            labels.append(f"<Picture {i}> = {p}")
        for i, p in enumerate(videos, 1):
            labels.append(f"<Video {i}> = {p}")
        for i, p in enumerate(audios, 1):
            labels.append(f"<Audio {i}> = {p}")
        system = load_prompt("perceive_refs")
        text = _append_grid_scan(
            "Inventory attached assets in this order:\n" + "\n".join(labels)
        )
        inventory = chat(
            system,
            user_parts(
                text,
                images=images or None,
                videos=videos or None,
                audios=audios or None,
            ),
            stage="perceive",
        )
        steps.append({"stage": "perceive_refs", "text": inventory})
        rescanned, rescan = _rescan_if_grid_incomplete(
            system,
            inventory,
            text=text,
            images=images or None,
            videos=videos or None,
            audios=audios or None,
        )
        if rescan:
            inventory = rescanned
            steps.append({"stage": rescan, "text": inventory})

    # Intent Contract：感知后、路由前；以 01_parse_intent (LLM) 为唯一主真理源，只抽取不推断。
    contract = parse_intent(intent, mode=mode, inventory=inventory, chat=chat, use_llm=True)
    steps.append({"stage": "contract", "text": contract.format_for_prompt()})
    # 时长只有一个真值：显式 API 参数优先，并同步回 Intent Contract。
    # 否则 contract 中默认的 5 秒会继续进入 expand / format，覆盖调用方传入的时长。
    if duration is not None:
        contract.duration_sec = float(dur)
    elif contract.duration_sec:
        dur = int(contract.duration_sec)

    router_mode = (skill_router or "hybrid").strip().lower()
    if router_mode not in ROUTER_MODES:
        raise ValueError(f"skill_router 须为 {' / '.join(ROUTER_MODES)}")

    mech_router_mode = (mechanism_router or "hybrid").strip().lower()
    if mech_router_mode not in MECHANISM_ROUTER_MODES:
        raise ValueError(f"mechanism_router 须为 {' / '.join(MECHANISM_ROUTER_MODES)}")

    # 并发执行风格路由与机制路由（减少串行 LLM 往返延迟）
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as router_pool:
        f_style = router_pool.submit(
            select_style_skills,
            intent,
            inventory=inventory,
            forced=skills,
            router=router_mode,
            classify=chat if router_mode in {"hybrid", "llm"} else None,
            explicit_style=contract.explicit_style,
            explicit_negatives=list(contract.explicit_negatives or []),
        )
        f_mech = router_pool.submit(
            select_mechanisms,
            intent,
            inventory=inventory,
            forced=mechanisms,
            router=mech_router_mode,
            classify=chat if mech_router_mode in {"hybrid", "llm"} else None,
        )
        style_sel = f_style.result()
        mech_sel = f_mech.result()

    style_block = style_block_for_user(style_sel)
    extra_guides = style_sel.overlay_pairs()
    if style_sel.ids or style_sel.scores:
        steps.append(
            {
                "stage": "skill_route",
                "source": style_sel.source,
                "skills": style_sel.ids,
                "scores": style_sel.scores,
                "threshold": style_sel.threshold,
            }
        )

    mechanism_block = mechanism_block_for_user(mech_sel)
    writing_block = writing_blocks_for_user(style_block, mechanism_block)
    if mech_sel.ids:
        steps.append(
            {
                "stage": "mechanism_route",
                "text": f"source={mech_sel.source}; mechanisms={', '.join(mech_sel.ids)}",
            }
        )

    contract_block = contract.format_for_prompt()
    complexity_block = format_complexity_budget_block(contract)

    # 先扩展意图骨架，再用独立 elaborate 阶段补足可执行的物理、镜头与声画细节。
    expand_sys = load_prompt("expand_intent")
    expanded = chat(
        expand_sys,
        _expand_user(
            intent,
            inventory=inventory,
            mode=mode,
            writing_block=writing_block,
            contract_block=contract_block,
            complexity_block=complexity_block,
            contract=contract,
        ),
        stage="expand",
    )
    steps.append({"stage": "expand", "text": expanded})
    elaborated = chat(
        load_prompt("elaborate"),
        _elaborate_user(
            expanded,
            inventory,
            intent,
            contract_block=contract_block,
            complexity_block=complexity_block,
            contract=contract,
        ),
        stage="elaborate",
    )
    steps.append({"stage": "elaborate", "text": elaborated})

    format_sys = compose_format_system(mode, load_prompt("format_h3"), extra_guides)
    format_text = _format_user(
        mode,
        elaborated,
        inventory=inventory,
        duration=dur,
        intent=intent,
        contract_block=contract_block,
        contract=contract,
    )
    format_user = format_text
    raw_prompt = chat(format_sys, format_user, stage="format")
    official_prompt = raw_prompt
    prompt = ensure_alignment_prefix(mode, strip_canvas(raw_prompt), dur)
    steps.append({"stage": "format", "text": prompt})

    # 确定性规则硬修复与快速清洗（毫秒级开销，无额外网络往返）
    if mode in KEYFRAME_MODES:
        frame_images = [p for p in (first_frame, last_frame) if p]
        verify_imgs, verify_vids, verify_auds = len(frame_images), 0, 0
    else:
        verify_imgs = len(images) if mode == "r2va" else 0
        verify_vids = len(videos) if mode == "r2va" else 0
        verify_auds = len(audios) if mode == "r2va" else 0

    verify_result = verify_and_fix(
        mode,
        prompt,
        duration=dur,
        images=verify_imgs,
        videos=verify_vids,
        audios=verify_auds,
        chat=chat if (enable_verify and verify_intent_llm) else None,
        asset_fix_chat=chat if enable_verify else None,
        intent=intent,
        inventory=inventory,
        check_intent_llm=False,
        max_fix_rounds=1 if (enable_verify and verify_intent_llm) else 0,
        contract=contract,
        max_fidelity_fix_rounds=0,
    )
    prompt = verify_result["prompt"]
    # 清洗最终 prompt：严禁带有 MODE= 前缀或 markdown 块
    prompt = re.sub(r"^MODE=[a-z0-9_]+\s*\n+", "", prompt.strip(), flags=re.I).strip()
    if prompt.startswith("```"):
        prompt = prompt.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    import time as _time

    # pe_backend 分支进入本路径前未计时：用 steps 里请求次数近似 http_calls
    http_calls = sum(
        1
        for s in steps
        if s.get("stage")
        in {
            "perceive_image",
            "perceive_refs",
            "contract",
            "skill_route",
            "mechanism_route",
            "expand",
            "elaborate",
            "format",
        }
        or str(s.get("stage", "")).startswith("rescan")
    )

    record: dict[str, Any] = {
        "mode": mode,
        "backend": "gemini",
        "intent": intent,
        "duration": dur,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "reference_images": images if mode == "r2va" else [],
        "i2va_first_frame": first_frame if mode == "i2va" else None,
        "reference_videos": videos,
        "reference_audios": audios,
        "inventory": inventory,
        "contract": contract.to_dict(),
        "expanded": expanded,
        "elaborated": elaborated,
        "style_skills": style_sel.ids,
        "style_skill_source": style_sel.source,
        "style_skill_scores": style_sel.scores,
        "style_skill_threshold": style_sel.threshold,
        "mechanisms": mech_sel.ids,
        "mechanism_source": mech_sel.source,
        "prompt_official": official_prompt,
        "prompt": prompt,
        "verify": verify_result,
        "timing": {
            "total_sec": round(_time.perf_counter() - _t0, 3),
            "http_calls_estimate": http_calls,
        },
        "steps": steps,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (out_dir / "prompt_official_raw.txt").write_text(
            official_prompt.strip() + "\n",
            encoding="utf-8",
        )
        (out_dir / "expanded.txt").write_text(expanded.strip() + "\n", encoding="utf-8")
        (out_dir / "elaborated.txt").write_text(elaborated.strip() + "\n", encoding="utf-8")
        if inventory:
            (out_dir / "inventory.txt").write_text(inventory.strip() + "\n", encoding="utf-8")
        (out_dir / "contract.json").write_text(
            json.dumps(contract.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        slim = {k: v for k, v in record.items() if k != "steps"}
        slim["steps"] = steps
        (out_dir / "run.json").write_text(
            json.dumps(slim, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        record["out_dir"] = str(out_dir)
    return record


def run_job(
    mode: str,
    intent: str,
    *,
    first_frame: str | None = None,
    last_frame: str | None = None,
    reference_images: list[str] | None = None,
    reference_videos: list[str] | None = None,
    reference_audios: list[str] | None = None,
    duration: int | None = None,
    ratio: str | None = None,
    resolution: str | None = None,
    out_dir: Path | None = None,
    make_video: bool = True,
    wait_video: bool = True,
    compare_video: bool = False,
    skills: list[str] | None = None,
    skill_router: str = "hybrid",
    mechanisms: list[str] | None = None,
    mechanism_router: str = "hybrid",
    enable_verify: bool = True,
    verify_intent_llm: bool | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """增强 prompt，可选调用 H3 出片。画幅/分辨率只进视频 API。"""
    h3 = h3_settings()
    dur = duration if duration is not None else infer_duration(intent, h3["default_duration"])
    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "runs" / f"{mode}_{stamp}"
    rec = enhance(
        mode,
        intent,
        first_frame=first_frame,
        last_frame=last_frame,
        reference_images=reference_images,
        reference_videos=reference_videos,
        reference_audios=reference_audios,
        duration=dur,
        out_dir=out_dir,
        skills=skills,
        skill_router=skill_router,
        mechanisms=mechanisms,
        mechanism_router=mechanism_router,
        enable_verify=enable_verify,
        verify_intent_llm=verify_intent_llm,
        backend=backend,
        resolution=ratio,
    )
    rec["ratio_api"] = ratio or (h3["default_ratio"] if mode == "t2va" else "adaptive")
    rec["resolution_api"] = resolution or h3["default_resolution"]
    rec["make_video"] = make_video
    rec["compare_video"] = compare_video

    prompt_official = rec.get("prompt_official") or ""
    prompt_local = rec.get("prompt") or ""
    video_official: dict[str, Any] | None = None
    video_local: dict[str, Any] | None = None
    if make_video:
        video_local_path = Path(out_dir) / "out_local.mp4"
        video_local_res = generate_video(
            mode,
            prompt_local,
            duration=dur,
            ratio=ratio,
            resolution=resolution,
            first_frame=first_frame,
            last_frame=last_frame,
            reference_images=reference_images,
            reference_videos=reference_videos,
            reference_audios=reference_audios,
            output=video_local_path,
            wait=wait_video,
        )
        video_local = {k: v for k, v in video_local_res.items() if k != "task"}
        video_local["task_status"] = (video_local_res.get("task") or {}).get("status")
        rec["video"] = video_local

        if compare_video:
            video_official_path = Path(out_dir) / "out_official.mp4"
            video_official_res = generate_video(
                mode,
                prompt_official,
                duration=dur,
                ratio=ratio,
                resolution=resolution,
                first_frame=first_frame,
                last_frame=last_frame,
                reference_images=reference_images,
                reference_videos=reference_videos,
                reference_audios=reference_audios,
                output=video_official_path,
                wait=wait_video,
            )
            video_official = {k: v for k, v in video_official_res.items() if k != "task"}
            video_official["task_status"] = (video_official_res.get("task") or {}).get("status")
            rec["video_official"] = video_official

        run_path = Path(out_dir) / "run.json"
        if run_path.is_file():
            dumped = json.loads(run_path.read_text(encoding="utf-8"))
            dumped["video"] = rec.get("video")
            dumped["video_official"] = rec.get("video_official")
            dumped["ratio_api"] = rec["ratio_api"]
            dumped["resolution_api"] = rec["resolution_api"]
            run_path.write_text(json.dumps(dumped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 无论是否出片，都写出提示词对比报告（可选视频对比会包含对应视频结果）。
    write_report(
        out_dir,
        record=rec,
        prompt_official=prompt_official,
        prompt_local=prompt_local,
        video_official=video_official,
        video_local=video_local,
    )
    return rec
