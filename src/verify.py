"""提示词质量校验：确定性规则硬校验 + 可选 LLM 修复。

规则层不产生额外 HTTP 调用；只有检测到 error 且调用方提供 chat 时才触发
LLM 修复（stage="verify"），修复后重新校验，最多 max_fix_rounds 轮。

各规则对应的质量问题：
- 字段结构 / 对齐句 / 画幅残留  → 结构不稳
- 时间戳单调性                 → 画面诡异（时序混乱）
- 标签编号 / 标签使用          → 丢失参考素材
- <d>[Language] 语言匹配       → 台词发音紊乱
- 禁止 [Mandarin]              → 台词发音紊乱
- 用户原句必须进 <d>           → 台词被翻译 / 漏句
- 屏上文字原句必须出现         → 字幕/标题被翻译或丢弃
- 有对白却 soundscape=N/A      → 音视频同步薄弱
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable

from .config import load_prompt

BASE_FIELDS = ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music")
R2VA_FIELDS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)
KEYFRAME_MODES = ("i2va", "fl2va", "l2va")

_ALIGN_PREFIXES = (
    "For the target video, at 0.00 seconds",
    "How the reference pictures align with the target video",
)
_SHOT_TS_RE = re.compile(r"\[Shot\s+(\d+)\]\s*At\s+(\d{2}):(\d{2})\.(\d{3})", re.I)
_SHOT_TS_CANDIDATE_RE = re.compile(
    r"\[Shot\s+(\d+)\]\s*At\s+((?:(?:\d{1,2}):)?\d{1,2}(?:\.\d{1,3})?s?)",
    re.I,
)
# Subject 也纳入标签定义匹配：subject_definitions 行首定义以 <Subject N> 开头。
_LABEL_RE = re.compile(r"<(Subject|Picture|Video|Audio)\s+(\d+)>", re.I)
_DLANG_RE = re.compile(r"<d>\s*\[([^\]\n]+)\]\s*(.*?)</d>", re.S | re.I)
_SOUNDSCAPE_RE = re.compile(
    r"overall_soundscape\s*:\s*(.*?)(?=\n\s*\n(?:non_diegetic_music|[a-z_]+\s*:)|\Z)",
    re.S | re.I,
)
_ONSCREEN_CUE_RE = re.compile(
    r"(字幕|标题|花字|屏上|屏幕文字|屏幕显示|角标|logo\b|CTA|on[- ]?screen|subtitle|caption|title\s*text)",
    re.I,
)

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

_QUOTE_RES = (
    re.compile(r"「([^」]+)」"),
    re.compile(r"『([^』]+)』"),
    re.compile(r"‘([^’]+)’"),
    # ASCII apostrophes inside contractions are content, not closing quotes.
    re.compile(r"(?<![A-Za-z0-9])'((?:[^'\n]|(?<=[A-Za-z])'(?=[A-Za-z]))+)'(?![A-Za-z0-9])"),
    re.compile(r"“([^”]+)”"),
    re.compile(r'"([^"]+)"'),
)
_PUNCT_NORM = str.maketrans(
    {
        "。": ".",
        "．": ".",
        "！": "!",
        "？": "?",
        "，": ",",
        "、": ",",
        "：": ":",
        "；": ";",
        "…": ".",
        "—": "-",
        "～": "~",
        "「": "",
        "」": "",
        "『": "",
        "』": "",
        '"': "",
        "“": "",
        "”": "",
    }
)

# 画幅/分辨率/帧率残留（strip_canvas 已先清理，这里是最终安全网）。
_CANVAS_RE = re.compile(
    r"(?i)\b\d+(?:\.\d+)?\s*fps\b"
    r"|\b(?:360P|480P|540P|720P|768P|1080P|2K|4K|8K|1280x720|1920x1080)\b"
    r"|\b(?:16:9|9:16|21:9|4:3|3:4)\s*(?:aspect\s*ratio|横屏|竖屏)?"
)


@dataclass(frozen=True)
class VerifyIssue:
    """一条校验问题。severity: error 阻断 / warning 提示。"""

    code: str
    severity: str
    message: str


def check_field_structure(mode: str, prompt: str) -> list[VerifyIssue]:
    """三字段/六段必须存在且顺序正确。"""
    text = (prompt or "").strip()
    fields = R2VA_FIELDS if mode == "r2va" else BASE_FIELDS
    issues: list[VerifyIssue] = []
    positions: list[int] = []
    for f in fields:
        idx = text.find(f + ":")
        positions.append(idx)
        if idx == -1:
            issues.append(VerifyIssue("field_missing", "error", f"缺少字段: {f}"))
    if all(p >= 0 for p in positions) and positions != sorted(positions):
        issues.append(VerifyIssue("field_order", "error", "字段顺序不符合官方骨架"))
    return issues


def check_alignment_line(mode: str, prompt: str, duration: int) -> list[VerifyIssue]:
    """关键帧模式：首行必须是对齐句，且 S.SS 与时长一致（两位小数）。"""
    if mode not in KEYFRAME_MODES:
        return []
    text = (prompt or "").strip()
    first_line = text.splitlines()[0] if text.splitlines() else ""
    issues: list[VerifyIssue] = []
    if not first_line.startswith(_ALIGN_PREFIXES):
        issues.append(
            VerifyIssue("align_missing", "error", "首行不是官方对齐句（For the target video... / How the reference pictures align...）")
        )
        return issues
    if mode == "i2va":
        return issues
    sss = f"{float(duration):.2f}"
    if f"{sss}-second" not in first_line:
        issues.append(
            VerifyIssue("align_duration", "error", f"对齐句 S.SS 应为 {sss}，与出片时长 {duration}s 一致")
        )
    return issues


def normalize_timestamps(prompt: str) -> str:
    """把常见的 Shot 时间写法确定性归一为 ``At MM:SS.mmm``。"""
    def replace_match(match: re.Match[str]) -> str:
        shot_no, raw = match.groups()
        token = raw[:-1] if raw.lower().endswith("s") else raw
        if ":" in token:
            minutes_raw, seconds_raw = token.split(":", 1)
        else:
            minutes_raw, seconds_raw = "0", token
        if "." in seconds_raw:
            seconds_int, fraction = seconds_raw.split(".", 1)
        else:
            seconds_int, fraction = seconds_raw, ""
        minutes = int(minutes_raw)
        seconds = int(seconds_int)
        millis = int((fraction + "000")[:3])
        minutes += seconds // 60
        seconds %= 60
        return f"[Shot {int(shot_no)}] At {minutes:02d}:{seconds:02d}.{millis:03d}"

    return _SHOT_TS_CANDIDATE_RE.sub(replace_match, prompt or "")


def check_timestamps(prompt: str, duration: int) -> list[VerifyIssue]:
    """Shot 时间戳须格式正确、严格递增且不超过视频时长。"""
    issues: list[VerifyIssue] = []
    for shot_no, raw in _SHOT_TS_CANDIDATE_RE.findall(prompt or ""):
        full = f"[Shot {shot_no}] At {raw}"
        if _SHOT_TS_RE.fullmatch(full) is None:
            issues.append(
                VerifyIssue(
                    "shot_time_malformed",
                    "error",
                    f"[Shot {shot_no}] 时间戳格式错误: {raw}；应为 MM:SS.mmm",
                )
            )
    prev_secs = -1.0
    for shot_no, mm, ss, mmm in _SHOT_TS_RE.findall(prompt or ""):
        secs = float(mm) * 60 + float(ss) + float(mmm) / 1000
        if secs <= prev_secs:
            issues.append(
                VerifyIssue(
                    "shot_time_not_increasing",
                    "error",
                    f"[Shot {shot_no}] 时间戳 {mm}:{ss}.{mmm} 未严格递增（前一个时间点为 {prev_secs:.3f}s）",
                )
            )
        if secs > duration + 0.001:
            issues.append(
                VerifyIssue(
                    "shot_time_over_duration",
                    "error",
                    f"[Shot {shot_no}] 时间戳 {mm}:{ss}.{mmm} 超过时长 {duration}s",
                )
            )
        prev_secs = secs
    return issues


def check_label_numbers(
    prompt: str,
    *,
    images: int = 0,
    videos: int = 0,
    audios: int = 0,
) -> list[VerifyIssue]:
    """参考标签编号不能超过实际素材数量（防止发明不存在的素材）。"""
    issues: list[VerifyIssue] = []
    # 每段参考视频都可以独立贡献一条同步音轨，因此 Audio 编号上限包含 videos。
    for kind, limit in (("Picture", images), ("Video", videos), ("Audio", audios + videos)):
        for n in {int(num) for k, num in _LABEL_RE.findall(prompt or "") if k == kind}:
            if n > limit:
                issues.append(
                    VerifyIssue(
                        "label_overrun",
                        "error",
                        f"<{kind} {n}> 超出实际素材数（{limit}），不要发明未上传的素材",
                    )
                )
    return issues


def check_asset_citations(
    prompt: str,
    *,
    images: int = 0,
    videos: int = 0,
    audios: int = 0,
) -> list[VerifyIssue]:
    """所有实际上传的素材必须至少以对应标签出现一次。"""
    text = prompt or ""
    issues: list[VerifyIssue] = []
    for kind, count in (("Picture", images), ("Video", videos), ("Audio", audios)):
        for number in range(1, count + 1):
            label = f"<{kind} {number}>"
            if re.search(rf"<{kind}\s+{number}>", text, re.I) is None:
                issues.append(
                    VerifyIssue(
                        "asset_uncited",
                        "error",
                        f"已上传素材 {label} 未在提示词中引用",
                    )
                )
    return issues


def check_label_usage(prompt: str) -> list[VerifyIssue]:
    """r2va：subject_definitions 里「行首独立定义」的标签必须在正文被引用。

    只有行首以 <Subject N> / <Picture N> / <Video N> / <Audio N> 开头的行才视为
    独立定义；定义行内嵌的素材来源引用（如 <Subject 1> is ... in <Picture 1>）
    只说明出处，不单独算作定义，不参与该检查。
    """
    text = (prompt or "").strip()
    m = re.search(r"subject_definitions:\s*(.*?)(?=\n\w+:|$)", text, re.S)
    if not m:
        return []
    defined: set[str] = set()
    for line in m.group(1).splitlines():
        lbl = _LABEL_RE.match(line.strip())
        if lbl:
            defined.add(f"<{lbl.group(1)} {lbl.group(2)}>")
    body = text[m.end() :]
    return [
        VerifyIssue("label_unused", "warning", f"subject_definitions 定义了但正文未引用: {lbl}")
        for lbl in sorted(defined)
        if lbl not in body
    ]


def extract_locked_dialogue(intent: str) -> list[str]:
    """从用户意图抽出有序台词；保留重复次数，排除屏上文字。"""
    return [line for _, line in extract_locked_dialogue_occurrences(intent)]


def extract_locked_dialogue_occurrences(intent: str) -> list[tuple[int, str]]:
    """按出现顺序返回对白位置与原文；重复台词不能去重。"""
    return [(start, line) for start, _end, line in extract_locked_dialogue_spans(intent)]


STYLE_KEYWORD_RE = re.compile(
    r"^(?:国风3D(?:渲染)?|3D渲染|水墨风格?|赛博朋克|二次元|皮影戏|定格动画|粘土风|真人实拍|电影质感|动漫风格?|像素风|写实风格?)$",
    re.I,
)
STYLE_META_SECTION_RE = re.compile(
    r"(?:\n\s*(?:Visual style|美术规范|风格说明|美术提示词|Style description)\s*[:：].*)\Z",
    re.I | re.S,
)


def extract_locked_dialogue_spans(intent: str) -> list[tuple[int, int, str]]:
    """按出现顺序返回对白的起止位置与原文，供前后置说话人归属使用。"""
    clean_intent = STYLE_META_SECTION_RE.sub("", intent or "")
    onscreen = set(extract_locked_onscreen(clean_intent))
    hits: list[tuple[int, int, str]] = []
    for rx in _QUOTE_RES:
        for match in rx.finditer(clean_intent):
            line = (match.group(1) or "").strip()
            if line in onscreen:
                continue
            prefix = clean_intent[max(0, match.start() - 35) : match.start()]
            if re.search(
                r"(?:风格|画风|题材|渲染|style|美术|质感|规范|专为|限定于|关键词|锁定|要求|画幅|比例|参数|属性|格式|结构)\s*[:：#\-_（(\[、“]*\s*$",
                prefix,
                re.I,
            ):
                continue
            if _is_lockable_line(line):
                hits.append((match.start(), match.end(), line))
    hits.sort(key=lambda item: item[0])
    return hits


def extract_locked_onscreen(intent: str) -> list[str]:
    """抽出紧跟字幕/标题/招牌等线索的引号文案，作为屏上文字锁定。"""
    text = STYLE_META_SECTION_RE.sub("", intent or "")
    hits: list[tuple[int, str]] = []
    # 线索与引号之间只允许很短间隔，避免把后面的对白误收进来。
    # 含门头/霓虹/写着/旁注/屏显等：避免屏上字被误判为对白（FC3 要求进 <d>）。
    cue = (
        r"(?:字幕|标题|花字|屏上|屏幕文字|屏幕显示|角标|logo|CTA|"
        r"on[- ]?screen|subtitle|caption|title\s*text|"
        r"写着|旁注|标注|显示|镌刻|印着|刻着|投影|滚动|"
        r"霓虹|门头|招牌|大屏|屏幕|大字|提示|屏显|界面|高亮|"
        r"胸牌|说明牌|广告牌|绣|浮出|弹出|白字|"
        r"中文提示|中文界面|浮出中文|弹出中文|下行?为|"
        r"尾帧|终态|机房|侧板|屏幕上|界面上)"
    )
    patterns = (
        re.compile(
            cue + r"[^「」\"“”\n]{0,16}[「\"“]([^」\"”]+)[」\"”]",
            re.I,
        ),
    )
    for rx in patterns:
        for match in rx.finditer(text):
            line = (match.group(1) or "").strip()
            if _is_lockable_line(line):
                hits.append((match.start(1), line))
    hits.sort(key=lambda item: item[0])
    seen: set[str] = set()
    lines: list[str] = []
    for _, line in hits:
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return lines


def _is_lockable_line(line: str) -> bool:
    """过滤路径、空串、纯标点、艺术风格词，保留对白/屏上字（含纯数字如楼层「18」）。"""
    text = (line or "").strip()
    if len(text) < 1:
        return False
    if re.search(r"\.(png|jpg|jpeg|webp|mp4|mov|wav)\b", text, re.I):
        return False
    if not (_CJK_RE.search(text) or _LATIN_RE.search(text) or re.search(r"\d", text)):
        return False
    if STYLE_KEYWORD_RE.match(text) or text.endswith("风格") or text.endswith("画风"):
        return False
    return True


def _norm_dialogue(text: str) -> str:
    """对白比对用：去掉空白、统一中英文标点。"""
    compact = re.sub(r"\s+", "", (text or "").strip())
    compact = compact.translate(_PUNCT_NORM)
    return compact.replace("...", ".")


def check_dialogue_verbatim(prompt: str, locked: list[str]) -> list[VerifyIssue]:
    """用户意图里的锁定台词必须原句出现在某个 <d> 内，不得翻译或漏写。"""
    if not locked:
        return []
    inners = [c.strip() for _, c in _DLANG_RE.findall(prompt or "")]
    joined = "\n".join(_norm_dialogue(x) for x in inners)
    issues: list[VerifyIssue] = []
    for line in locked:
        needle = _norm_dialogue(line)
        if needle and needle in joined:
            continue
        issues.append(
            VerifyIssue(
                "dialogue_verbatim_missing",
                "error",
                f"用户原句未出现在 <d> 内（禁止翻译或漏写）: 「{line}」",
            )
        )
    return issues


def check_onscreen_verbatim(prompt: str, locked: list[str]) -> list[VerifyIssue]:
    """锁定屏上文字必须原句出现在提示词正文（不要求进 <d>）。"""
    if not locked:
        return []
    haystack = _norm_dialogue(prompt or "")
    issues: list[VerifyIssue] = []
    for line in locked:
        needle = _norm_dialogue(line)
        if needle and needle in haystack:
            continue
        issues.append(
            VerifyIssue(
                "onscreen_verbatim_missing",
                "error",
                f"屏上文字原句缺失（禁止翻译或漏写）: 「{line}」",
            )
        )
    return issues


def check_speech_soundscape(prompt: str, locked_dialogue: list[str]) -> list[VerifyIssue]:
    """有锁定对白时 overall_soundscape 不应写成纯 N/A。"""
    if not locked_dialogue:
        return []
    match = _SOUNDSCAPE_RE.search(prompt or "")
    if not match:
        return []
    body = (match.group(1) or "").strip()
    if re.fullmatch(r"N/?A\.?", body, flags=re.I):
        return [
            VerifyIssue(
                "av_sync_soundscape_empty",
                "warning",
                "存在对白但 overall_soundscape 为 N/A，音画同步描述可能不足",
            )
        ]
    return []


def check_dialogue_language(prompt: str) -> list[VerifyIssue]:
    """<d>[Lang] 内容</d>：支持动态语言标签，校验标签与内容文字冲突（如 [English] 包中文）。"""
    issues: list[VerifyIssue] = []
    for lang, content in _DLANG_RE.findall(prompt or ""):
        key = lang.strip().lower()
        body = content.strip()
        if not key:
            issues.append(VerifyIssue("dialogue_empty_lang", "error", "<d>[] 缺少语言标签"))
            continue
        if not body:
            issues.append(VerifyIssue("dialogue_empty_content", "error", f"<d>[{lang}] 对白内容为空"))
            continue
        if key in ("english", "french", "german", "spanish") and _CJK_RE.search(body):
            snippet = body[:30].replace("\n", " ")
            issues.append(
                VerifyIssue(
                    "dialogue_lang_mismatch",
                    "error",
                    f"<d>[{lang}] 内容包含中文字符但语言标签为 {lang}: 「{snippet}...」",
                )
            )
    return issues


def check_dialogue_contract(prompt: str, dialogue: list[Any]) -> list[VerifyIssue]:
    """严格校验 Voice Contract：结构、次数、顺序、原字符、语言与 S-ID。"""
    text = prompt or ""
    matches = list(_DLANG_RE.finditer(text))
    issues: list[VerifyIssue] = []
    open_count = len(re.findall(r"<d\b", text, re.I))
    close_count = len(re.findall(r"</d>", text, re.I))
    if open_count != close_count or open_count != len(matches):
        issues.append(
            VerifyIssue(
                "dialogue_markup_invalid",
                "error",
                f"<d> 包裹不完整或格式错误: open={open_count}, close={close_count}, parsed={len(matches)}",
            )
        )

    expected = list(dialogue or [])
    if len(matches) != len(expected):
        issues.append(
            VerifyIssue(
                "dialogue_count_mismatch",
                "error",
                f"对白次数不符: expected={len(expected)}, actual={len(matches)}",
            )
        )

    for idx, (match, item) in enumerate(zip(matches, expected), 1):
        lang = (match.group(1) or "").strip()
        content = (match.group(2) or "").strip()
        exact = str(getattr(item, "text", "") or "")
        if content != exact:
            issues.append(
                VerifyIssue(
                    "dialogue_exact_mismatch",
                    "error",
                    f"第{idx}句对白必须逐字符逐标点一致: expected={exact!r}, actual={content!r}",
                )
            )
        expected_lang = str(getattr(item, "language", "") or "")
        if expected_lang and lang.lower() != expected_lang.lower():
            issues.append(
                VerifyIssue(
                    "dialogue_contract_language",
                    "error",
                    f"第{idx}句语言标签应为 [{expected_lang}]，实际为 [{lang}]",
                )
            )
        prefix = text[max(0, match.start() - 240) : match.start()]
        suffix = text[match.end() : min(len(text), match.end() + 200)]
        expected_sid = str(getattr(item, "speaker_id", "") or "")
        if expected_sid:
            ids = re.findall(r"\((S\d+)\)", prefix, re.I)
            actual_sid = ids[-1].upper() if ids else ""
            if actual_sid != expected_sid.upper():
                issues.append(
                    VerifyIssue(
                        "dialogue_speaker_mismatch",
                        "error",
                        f"第{idx}句说话人应为 ({expected_sid})，实际为 ({actual_sid or 'missing'})",
                    )
                )

        speech_type = str(getattr(item, "speech_type", "dialogue") or "dialogue").lower()
        has_offscreen_phrase = "says in an off-screen voiceover" in prefix.lower()
        has_closed_lips = bool(
            re.search(r"lips? (?:remain|remains|stay|stays|are) completely closed", suffix, re.I)
        )
        if speech_type == "dialogue" and has_offscreen_phrase:
            issues.append(
                VerifyIssue(
                    "dialogue_delivery_mismatch",
                    "error",
                    f"第{idx}句类型为 dialogue，不应渲染成 off-screen voiceover",
                )
            )
        elif speech_type == "voiceover" and (not has_offscreen_phrase or not has_closed_lips):
            issues.append(
                VerifyIssue(
                    "dialogue_delivery_mismatch",
                    "error",
                    f"第{idx}句类型为 voiceover，必须使用固定 off-screen voiceover 短语并声明匹配角色嘴唇完全闭合",
                )
            )
        elif speech_type == "narration" and (not has_offscreen_phrase or has_closed_lips):
            issues.append(
                VerifyIssue(
                    "dialogue_delivery_mismatch",
                    "error",
                    f"第{idx}句类型为 narration，必须保持独立旁白在画外且不得给画面人物绑定闭嘴/口型动作",
                )
            )
    return issues


def check_canvas_residue(prompt: str) -> list[VerifyIssue]:
    """画幅/分辨率/帧率等生产参数不应出现在提示词里。"""
    matches = [m.group(0) for m in _CANVAS_RE.finditer(prompt or "")]
    return [
        VerifyIssue("canvas_residue", "warning", f"残留画幅/分辨率/帧率词: {m}")
        for m in matches
    ]


def fix_asset_citations_with_llm(
    prompt: str,
    issues: list[VerifyIssue],
    *,
    inventory: str | None,
    contract: Any,
    chat: Callable[..., str],
) -> str:
    """定向补回完全未引用的上传素材，不改变字段骨架或发明素材事实。"""
    missing = [issue.message for issue in issues if issue.code == "asset_uncited"]
    if not missing:
        return prompt
    system = load_prompt("fix_asset_citations")
    user_lines = [
        "Missing uploaded asset citations:",
        *[f"- {message}" for message in missing],
    ]
    if contract is not None:
        user_lines.extend(["", contract.format_for_prompt()])
    if inventory:
        user_lines.extend(["", "Reference inventory (authoritative facts):", inventory.strip()])
    user_lines.extend(["", "Current prompt:", prompt])
    try:
        fixed = chat(system, "\n".join(user_lines), stage="fix_assets")
        if fixed and fixed.strip():
            return fixed.strip()
    except Exception:
        pass
    return prompt


def align_dialogue_with_llm(
    prompt: str,
    contract: Any,
    inventory: str | None = None,
    *,
    chat: Callable[..., str],
) -> str:
    """Stage 4 专用对话修复 API：输入 Stage 1 人物对话与素材匹配角色，对提示词中的对白进行精准对齐与修复。"""
    if chat is None:
        return prompt
    dialogues = list(getattr(contract, "dialogue", []) or [])
    system = load_prompt("align_dialogue")

    dialogue_info = []
    if dialogues:
        for idx, d in enumerate(dialogues, 1):
            dialogue_info.append(
                f"- #{idx} speaker_id={getattr(d, 'speaker_id', f'S{idx}')}; "
                f"speaker={getattr(d, 'speaker', 'character')}; "
                f"language={getattr(d, 'language', 'Chinese')}; "
                f"speech_type={getattr(d, 'speech_type', 'dialogue')}; "
                f"speaker_visible={str(getattr(d, 'speaker_visible', True)).lower()}; "
                f"lip_sync={getattr(d, 'lip_sync', 'required')}; "
                f"character_match={getattr(d, 'character_match', '') or 'NONE'}; "
                f"text={json.dumps(getattr(d, 'text', ''), ensure_ascii=False)}"
            )
        dialogue_block = "Target Dialogues & Matched Characters (Authoritative from Stage 1):\n" + "\n".join(dialogue_info)
    else:
        dialogue_block = "Target Dialogues: NONE (The user intent has no dialogue; remove any unrequested <d>...</d> tags)."

    user_msg = f"{dialogue_block}\n\n"
    if inventory:
        user_msg += f"Reference Material Context:\n{inventory[:1500]}\n\n"
    user_msg += f"Current Generated Prompt:\n{prompt}"

    try:
        fixed = chat(system, user_msg, stage="align_dialogue")
        if fixed and fixed.strip():
            return fixed.strip()
    except Exception:
        pass
    return prompt


def is_no_subtitles_requested(contract: Any) -> bool:
    """判断意图是否明确要求无字幕/无屏幕文字。"""
    if contract is None:
        return False
    forbidden = list(getattr(contract, "forbidden", []) or [])
    negatives = list(getattr(contract, "explicit_negatives", []) or [])
    all_terms = forbidden + negatives
    pattern = re.compile(r"不要字幕|无字幕|禁止字幕|不要屏幕文字|无屏幕文字|禁止屏幕文字|no\s+subtitles?|no\s+captions?|no\s+onscreen\s+text", re.I)
    return any(pattern.search(str(t)) for t in all_terms)


def check_forbidden_onscreen_quotes(prompt: str, contract: Any) -> list[VerifyIssue]:
    """若要求不要字幕/屏幕文字，画面描述中不应包含双引号文字贴字。"""
    if not is_no_subtitles_requested(contract):
        return []

    # 移除 <d>...</d> 标签后检查正文是否残留引号文字
    text_without_d = _DLANG_RE.sub("", prompt or "")

    quoted_hits = []
    for rx in _QUOTE_RES:
        for m in rx.finditer(text_without_d):
            val = (m.group(1) or "").strip()
            if len(val) >= 2:
                quoted_hits.append(val)

    issues: list[VerifyIssue] = []
    if quoted_hits:
        snippet = ", ".join(quoted_hits[:3])
        issues.append(
            VerifyIssue(
                "forbidden_onscreen_quotes",
                "error",
                f"已明确要求不要字幕/屏幕文字，但画面描述中残留双引号贴字: 「{snippet}」",
            )
        )
    return issues


def fix_nosubtitle_with_llm(
    prompt: str,
    contract: Any,
    inventory: str | None = None,
    *,
    chat: Callable[..., str],
) -> str:
    """Stage 4 专用无字幕修复 API：清除画面描述中的双引号贴字，并将其转换为自然画面动作或道具描述。"""
    if chat is None:
        return prompt
    system = load_prompt("fix_nosubtitle")
    user_msg = (
        "Negative Constraint: Forbidden Onscreen Text / No Subtitles (不要字幕、不要屏幕文字)\n\n"
        f"Contract Forbidden: {getattr(contract, 'forbidden', [])}\n\n"
    )
    if inventory:
        user_msg += f"Reference Material Context:\n{inventory[:1500]}\n\n"
    user_msg += f"Current Generated Prompt:\n{prompt}"

    try:
        fixed = chat(system, user_msg, stage="fix_nosubtitle")
        if fixed and fixed.strip():
            return fixed.strip()
    except Exception:
        pass
    return prompt


def verify_prompt(
    mode: str,
    prompt: str,
    *,
    duration: int,
    images: int = 0,
    videos: int = 0,
    audios: int = 0,
    intent: str = "",
    contract: Any = None,
    check_dialogue: bool = False,
) -> list[VerifyIssue]:
    """跑全部规则校验，返回问题列表。优先使用 contract 作为真理源。"""
    if contract is not None:
        dialogue_contract = list(getattr(contract, "dialogue", []) or [])
        locked_spoken = [d.text for d in dialogue_contract if getattr(d, "text", None)]
        locked_onscreen = list(getattr(contract, "onscreen_text", []) or [])
    else:
        from .contract import extract_dialogue_lines
        locked_spoken = extract_locked_dialogue(intent)
        locked_onscreen = extract_locked_onscreen(intent)
        dialogue_contract = extract_dialogue_lines(intent)

    issues: list[VerifyIssue] = []
    issues += check_field_structure(mode, prompt)
    issues += check_alignment_line(mode, prompt, duration)
    issues += check_timestamps(prompt, duration)
    issues += check_label_numbers(prompt, images=images, videos=videos, audios=audios)
    if mode == "r2va":
        issues += check_asset_citations(prompt, images=images, videos=videos, audios=audios)
        issues += check_label_usage(prompt)
    if check_dialogue:
        issues += check_dialogue_language(prompt)
        issues += check_dialogue_contract(prompt, dialogue_contract)
    issues += check_onscreen_verbatim(prompt, locked_onscreen)
    issues += check_forbidden_onscreen_quotes(prompt, contract)
    if check_dialogue:
        issues += check_speech_soundscape(prompt, locked_spoken)
    issues += check_canvas_residue(prompt)
    return issues


def check_intent_with_llm(
    chat: Callable[..., str],
    intent: str,
    prompt: str,
    inventory: str | None,
) -> list[VerifyIssue]:
    """LLM 判断最终提示词是否偏离原始意图（--verify-intent-llm 时启用）。"""
    system = load_prompt("verify_intent")
    user_lines = ["Original intent:", (intent or "").strip()]
    if inventory:
        user_lines.extend(["", "Reference inventory (excerpt):", inventory.strip()[:2000]])
    user_lines.extend(["", "Final prompt:", (prompt or "").strip()])
    raw = chat(system, "\n".join(user_lines), stage="verify_intent")
    try:
        payload = json.loads(raw.strip())
        if not isinstance(payload, dict):
            raise ValueError("非 JSON 对象")
        if not payload.get("consistent", True):
            problems = payload.get("problems") or []
            if not isinstance(problems, list):
                problems = []
            return [
                VerifyIssue("intent_drift", "error", f"意图偏差: {p}")
                for p in problems
                if isinstance(p, str)
            ]
        return []
    except (json.JSONDecodeError, ValueError):
        # 模型未按 JSON 回复时降级为警告，不阻断。
        return [
            VerifyIssue("intent_check_unparseable", "warning", "意图一致性检查返回无法解析的 JSON")
        ]


def apply_contract_hard_fixes(prompt: str, contract: Any) -> str:
    """对可用确定性修复的 contract 硬约束做就地修正（不整篇重写）。"""
    text = prompt or ""
    forbidden = list(getattr(contract, "forbidden", None) or [])
    dialogue = list(getattr(contract, "dialogue", None) or [])

    # 1. 只有当明确硬性禁止配乐时，才置为 N/A
    if any(re.search(r"不要配乐|禁止配乐|无配乐|不要音乐|禁止音乐|无BGM", f) for f in forbidden):
        text = re.sub(
            r"(non_diegetic_music\s*:\s*)(.*?)(?=\n\s*\n|\Z)",
            r"\1N/A",
            text,
            count=1,
            flags=re.I | re.S,
        )

    # 2. 仅在用户明确禁止对白/新增人声时剥离幻觉 <d>；没有锁定对白不等于禁止对白。
    no_dialogue_text = "\n".join([*forbidden, str(getattr(contract, "intent_raw", "") or "")])
    explicitly_silent = bool(
        re.search(
            r"无对白|没有对白|不开口|禁止[^，。；;\n]{0,12}(?:对白|台词|人声)|"
            r"不(?:新增|改写)[^，。；;\n]{0,8}(?:对白|台词|人声)",
            no_dialogue_text,
        )
    )
    if explicitly_silent and not dialogue and "<d>" in text:
        text = re.sub(
            r"(?:,\s*(?:and\s+)?(?:he|she|they|the\s+[A-Za-z0-9_-]+)?\s*(?:says?|speaks?|shouts?|whispers?|cries?|roars?|says in an off-screen voiceover)[\s:]*)*<d>\s*\[[^\]]+\]\s*.*?</d>",
            "",
            text,
            flags=re.S | re.I,
        )
        text = re.sub(r"<d>\s*\[[^\]]+\]\s*.*?</d>", "", text, flags=re.S | re.I)

    return text


def fidelity_issues_from_report(report: Any) -> list[VerifyIssue]:
    """把 FidelityReport 未通过项转为 VerifyIssue（阻断级 error）。"""
    issues: list[VerifyIssue] = []
    checks = getattr(report, "checks", None) or {}
    for code, item in checks.items():
        if getattr(item, "passed", True):
            continue
        detail = getattr(item, "detail", "") or ""
        viol = getattr(item, "violations", None) or []
        msg = f"{detail}: {', '.join(str(v) for v in viol)}" if viol else detail or code
        issues.append(VerifyIssue(f"fidelity_{code}", "error", msg))
    if not getattr(report, "passed", True) and not issues:
        issues.append(VerifyIssue("fidelity_fail", "error", "保真 gate 未通过"))
    return issues


def verify_and_fix(
    mode: str,
    prompt: str,
    *,
    duration: int,
    images: int = 0,
    videos: int = 0,
    audios: int = 0,
    chat: Callable[..., str] | None = None,
    asset_fix_chat: Callable[..., str] | None = None,
    intent: str = "",
    inventory: str | None = None,
    check_intent_llm: bool = False,
    max_fix_rounds: int = 1,
    contract: Any | None = None,
    max_fidelity_fix_rounds: int = 2,
    check_dialogue: bool = False,
) -> dict[str, Any]:
    """校验最终提示词；结构 error 与保真 gate 失败时做定向修复。

    Returns:
        prompt / fixed / status / rounds / fidelity_rounds / issues / fidelity
    """
    from .contract import IntentContract, extract_dialogue_lines, parse_intent_deterministic
    from .fidelity import evaluate_fidelity

    current = apply_contract_hard_fixes(prompt, contract) if contract is not None else prompt
    current = normalize_timestamps(current)
    issues = verify_prompt(
        mode,
        current,
        duration=duration,
        images=images,
        videos=videos,
        audios=audios,
        intent=intent,
        contract=contract,
        check_dialogue=check_dialogue,
    )
    if check_intent_llm and chat is not None and (intent or "").strip():
        issues.extend(check_intent_with_llm(chat, intent, current, inventory))

    rounds = 0

    # 1. 素材漏引使用独立修复通道；默认开启，不依赖昂贵的全量语义校验。
    asset_issues = [issue for issue in issues if issue.code == "asset_uncited"]
    if asset_fix_chat is not None and asset_issues:
        fixed_assets = fix_asset_citations_with_llm(
            current,
            asset_issues,
            inventory=inventory,
            contract=contract,
            chat=asset_fix_chat,
        )
        if fixed_assets.strip() != current.strip():
            current = normalize_timestamps(
                apply_contract_hard_fixes(fixed_assets, contract)
                if contract is not None
                else fixed_assets
            )
            rounds += 1
            issues = verify_prompt(
                mode,
                current,
                duration=duration,
                images=images,
                videos=videos,
                audios=audios,
                intent=intent,
                contract=contract,
                check_dialogue=check_dialogue,
            )

    # 2. 优先调用 Stage 4 专门的对白对齐修复 API
    dialogue_issues = [i for i in issues if i.code.startswith("dialogue_")]
    if chat is not None and dialogue_issues and contract is not None:
        fixed_dialogue = align_dialogue_with_llm(current, contract, inventory=inventory, chat=chat)
        if fixed_dialogue and fixed_dialogue.strip() != current.strip():
            current = fixed_dialogue
            rounds += 1
            issues = verify_prompt(
                mode,
                current,
                duration=duration,
                images=images,
                videos=videos,
                audios=audios,
                intent=intent,
                contract=contract,
                check_dialogue=check_dialogue,
            )

    # 3. 检查若存在不要字幕违背（画面描述中残留双引号贴字/文字），调用专门的 fix_nosubtitle API
    nosubtitle_issues = [i for i in issues if i.code == "forbidden_onscreen_quotes"]
    if chat is not None and nosubtitle_issues and contract is not None:
        fixed_nosub = fix_nosubtitle_with_llm(current, contract, inventory=inventory, chat=chat)
        if fixed_nosub and fixed_nosub.strip() != current.strip():
            current = fixed_nosub
            rounds += 1
            issues = verify_prompt(
                mode,
                current,
                duration=duration,
                images=images,
                videos=videos,
                audios=audios,
                intent=intent,
                contract=contract,
                check_dialogue=check_dialogue,
            )

    if chat is not None and max_fix_rounds > 0 and any(i.severity == "error" for i in issues):
        system = load_prompt("verify_fix")
        for _ in range(max_fix_rounds):
            user_lines = [
                "Issues:",
                *[f"- [{i.code}] {i.message}" for i in issues],
            ]
            locked = (
                list(getattr(contract, "dialogue", []) or [])
                if check_dialogue and contract is not None
                else extract_dialogue_lines(intent) if check_dialogue else []
            )
            if locked:
                user_lines.extend(
                    [
                        "",
                        "VOICE CONTRACT (one <d> per numbered entry; preserve order/count/S-ID/language/exact punctuation):",
                        *[
                            f"- #{idx} speaker_id={line.speaker_id or 'unknown'}; "
                            f"speaker={line.speaker or 'unknown'}; language={line.language}; "
                            f"speech_type={getattr(line, 'speech_type', 'dialogue')}; "
                            f"speaker_visible={str(getattr(line, 'speaker_visible', True)).lower()}; "
                            f"lip_sync={getattr(line, 'lip_sync', 'required')}; "
                            f"character_match={getattr(line, 'character_match', '') or 'NONE'}; "
                            f"exact_text={json.dumps(line.text, ensure_ascii=False)}"
                            for idx, line in enumerate(locked, start=1)
                        ],
                    ]
                )
            onscreen = list(getattr(contract, "onscreen_text", []) or []) if contract is not None else extract_locked_onscreen(intent)
            if onscreen:
                user_lines.extend(
                    [
                        "",
                        "Locked on-screen lines (keep verbatim as on-screen text in the original language):",
                        *[f"- {line}" for line in onscreen],
                    ]
                )
            if inventory:
                user_lines.extend(
                    [
                        "",
                        "Reference inventory (restore identity / coverage / spatial cues if missing):",
                        inventory.strip()[:2000],
                    ]
                )
            user_lines.extend(["", "Prompt:", current])
            fixed = chat(system, "\n".join(user_lines), stage="verify")
            if fixed.strip() == current.strip():
                break
            current = fixed
            rounds += 1
            issues = verify_prompt(
                mode,
                current,
                duration=duration,
                images=images,
                videos=videos,
                audios=audios,
                intent=intent,
                contract=contract,
                check_dialogue=check_dialogue,
            )
            if check_intent_llm and chat is not None and (intent or "").strip():
                issues.extend(check_intent_with_llm(chat, intent, current, inventory))
            if not any(i.severity == "error" for i in issues):
                break

    # ---- 保真 gate：结构通过后仍可能语义发明 / 动作链丢失 ----
    fidelity_rounds = 0
    fidelity_payload: dict[str, Any] | None = None
    if contract is None and (intent or "").strip():
        contract = parse_intent_deterministic(intent, mode=mode)
    run_fidelity = (
        isinstance(contract, IntentContract)
        and chat is not None
        and bool((intent or "").strip())
    )
    if run_fidelity:
        current = apply_contract_hard_fixes(current, contract)
        f_rep = evaluate_fidelity(contract, current, chat=chat, inventory=inventory or "")
        fidelity_payload = f_rep.to_dict()
        fid_issues = fidelity_issues_from_report(f_rep)
        if fid_issues and max_fidelity_fix_rounds > 0:
            system = load_prompt("verify_fidelity_fix")
            for _ in range(max_fidelity_fix_rounds):
                if not fid_issues:
                    break
                user_lines = [
                    f"MODE={mode}",
                    "",
                    contract.format_for_prompt(),
                    "",
                    "Fidelity violations (fix only these):",
                    *[f"- [{i.code}] {i.message}" for i in fid_issues],
                    "",
                    "Prompt:",
                    current,
                ]
                fixed = chat(system, "\n".join(user_lines), stage="verify")
                if fixed.strip() == current.strip():
                    break
                current = apply_contract_hard_fixes(fixed, contract)
                fidelity_rounds += 1
                # 结构复检 + 保真复检
                issues = verify_prompt(
                    mode,
                    current,
                    duration=duration,
                    images=images,
                    videos=videos,
                    audios=audios,
                    intent=intent,
                    check_dialogue=check_dialogue,
                )
                f_rep = evaluate_fidelity(contract, current, chat=chat, inventory=inventory or "")
                fidelity_payload = f_rep.to_dict()
                fid_issues = fidelity_issues_from_report(f_rep)
                if not fid_issues and not any(i.severity == "error" for i in issues):
                    break
            issues = [i for i in issues if i.severity == "error"] + fid_issues + [
                i for i in issues if i.severity != "error"
            ]

    status = "passed" if not any(i.severity == "error" for i in issues) else "failed"
    return {
        "prompt": current,
        "fixed": current != prompt,
        "status": status,
        "rounds": rounds,
        "fidelity_rounds": fidelity_rounds,
        "issues": [asdict(i) for i in issues],
        "errors": sum(1 for i in issues if i.severity == "error"),
        "warnings": sum(1 for i in issues if i.severity == "warning"),
        "intent_llm": bool(check_intent_llm),
        "dialogue_check": bool(check_dialogue),
        "fidelity": fidelity_payload,
    }
