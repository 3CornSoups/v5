"""Intent Contract：把短意图解析成机器可校验的保真事实来源。

设计约束（见 docs/优化方案-保真优先-v2.md §3.1）：
1. 抽取节点只抽取、不推断不补全；用户没说的保持空。
2. dialogue / onscreen_text 必须原文逐字（含标点）。
3. 未归属人物台词交给专用低温模型最佳匹配；调用失败时稳定兜底，不中断增强。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from .config import load_prompt
from .verify import (
    extract_locked_dialogue_spans,
    extract_locked_onscreen,
)

ChatFn = Callable[..., str]

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
DURATION_RE = re.compile(r"(?:约|大概|大约)?\s*(\d{1,2})\s*秒")
ACTION_CHAIN_RE = re.compile(
    r"([\u4e00-\u9fffA-Za-z0-9]{1,12})"
    r"(?:\s*[→\-–—]+\s*|\s*->\s*|\s*→\s*)"
    r"([\u4e00-\u9fffA-Za-z0-9]{1,12}"
    r"(?:\s*[→\-–—]+\s*|\s*->\s*|\s*→\s*"
    r"[\u4e00-\u9fffA-Za-z0-9]{1,12})*)"
)
ARROW_SPLIT_RE = re.compile(r"\s*(?:→|->|—|–|-)\s*")
SLOGAN_RE = re.compile(
    r"(?:口号|标语|花字|字幕|标题|屏上|屏幕文字|CTA|"
    r"写着|旁注|标注|显示|屏显|界面|高亮|胸牌|说明牌|广告牌|绣|大字|提示)"
    r"[^「」\"“”\n]{0,16}[「\"“]([^」\"”]+)[」\"”]"
)
FORBIDDEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"禁止切镜|不得切镜|不要切镜|单镜头|一镜到底|不要分镜"), "禁止切镜/单镜头"),
    (re.compile(r"禁止(?:人物|人脸|人)入镜|不得(?:人物|人脸|人)入镜|不要(?:人物|人脸|人)入镜|禁止人物"), "禁止人物/人脸入镜"),
    (re.compile(r"不要配乐|禁止配乐|无配乐|不要音乐|不要非叙境"), "不要配乐"),
    (re.compile(r"不得出现任何文字|不要出现文字|禁止文字|画面不得出现任何文字|无字幕"), "画面不得出现文字"),
    (re.compile(r"不要实拍|禁止实拍|非实拍"), "不要实拍"),
)
STYLE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"纯手绘速写|手绘速写|纯手绘"), "纯手绘速写"),
    (re.compile(r"动漫|二次元|日漫"), "动漫"),
    (re.compile(r"水墨|国风水墨"), "水墨"),
    (re.compile(r"赛博朋克"), "赛博朋克"),
)
NEGATIVE_STYLE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"不要实拍|禁止实拍"), "不要实拍"),
    (re.compile(r"不要动漫|禁止动漫"), "不要动漫"),
)


SPEECH_TYPES = ("dialogue", "voiceover", "narration")


def normalize_speech_metadata(
    speech_type: Any,
    *,
    speaker: str = "",
    context: str = "",
) -> tuple[str, bool, str]:
    """规范语音类型，并派生可见性与口型策略，避免下游自行猜测。"""
    raw = str(speech_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    ctx = f"{speaker} {context}".lower()
    if raw in {"voiceover", "voice_over", "inner_thought", "inner_monologue", "internal_monologue"}:
        kind = "voiceover"
    elif raw in {"narration", "narrator", "commentary", "explainer"}:
        kind = "narration"
    elif raw == "dialogue":
        kind = "dialogue"
    elif re.search(r"画外音|内心独白|内心旁白|voice[ -]?over|inner (?:thought|monologue)", ctx, re.I):
        kind = "voiceover"
    elif re.search(r"旁白|解说|解说员|narrat(?:or|ion)|commentary|explainer", ctx, re.I):
        kind = "narration"
    else:
        kind = "dialogue"

    if kind == "voiceover":
        return kind, True, "closed"
    if kind == "narration":
        return kind, False, "none"
    return kind, True, "required"


@dataclass
class DialogueLine:
    """一句发声事件：原文、语言、发声源，以及下游口型/可见性策略。"""

    text: str
    language: str = ""
    speaker: str = ""
    speaker_id: str = ""
    speech_type: str = "dialogue"
    speaker_visible: bool = True
    lip_sync: str = "required"
    character_match: str = ""
    attribution: str = ""  # explicit_before|explicit_after|single_character
    attribution_confidence: float | None = None


@dataclass
class ShotConstraint:
    """镜头约束：是否单镜头、最大镜头数。"""

    single_shot: bool = False
    max_shots: int | None = None


@dataclass
class RefAttr:
    """参考素材上的一条属性（perceive 之后回填）。"""

    ref_tag: str
    attr: str
    kind: str = "identity"  # identity|scene|motion|style


@dataclass
class IntentContract:
    """用户意图契约：保真度的唯一事实来源。"""

    must_elements: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    dialogue: list[DialogueLine] = field(default_factory=list)
    onscreen_text: list[str] = field(default_factory=list)
    shot_constraint: ShotConstraint = field(default_factory=ShotConstraint)
    duration_sec: float = 5.0
    action_chain: list[str] = field(default_factory=list)
    explicit_style: str | None = None
    explicit_negatives: list[str] = field(default_factory=list)
    reference_attrs: list[RefAttr] = field(default_factory=list)
    reference_roles: dict[str, str] = field(default_factory=dict)
    intent_raw: str = ""
    mode: str = "t2va"
    ambiguities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """序列化为可写入 run.json 的字典。"""
        return asdict(self)

    def is_nonempty(self) -> bool:
        """是否抽出了至少一项有意义约束（供 S3 全量抽检）。"""
        if self.must_elements or self.forbidden or self.dialogue or self.onscreen_text:
            return True
        if self.action_chain or self.explicit_style or self.explicit_negatives:
            return True
        if self.shot_constraint.single_shot or self.shot_constraint.max_shots is not None:
            return True
        if self.ambiguities:
            return True
        # 极简意图：至少应有 duration 与原始文本
        return bool((self.intent_raw or "").strip()) and self.duration_sec > 0

    def format_for_prompt(self) -> str:
        """把契约格式化为下游节点可粘贴的 CONTRACT 块。"""
        lines = [
            "=== INTENT CONTRACT (machine-verified; do not invent beyond this) ===",
            f"mode: {self.mode}",
            f"duration_sec: {self.duration_sec}",
            f"must_elements: {json.dumps(self.must_elements, ensure_ascii=False)}",
            f"forbidden: {json.dumps(self.forbidden, ensure_ascii=False)}",
            f"action_chain: {json.dumps(self.action_chain, ensure_ascii=False)}",
            f"explicit_style: {self.explicit_style or ''}",
            f"explicit_negatives: {json.dumps(self.explicit_negatives, ensure_ascii=False)}",
            f"shot_constraint: single_shot={self.shot_constraint.single_shot}, "
            f"max_shots={self.shot_constraint.max_shots}",
        ]
        if self.dialogue:
            lines.append("dialogue (verbatim):")
            for d in self.dialogue:
                meta = f" [{d.language}]" if d.language else ""
                sp = f" ({d.speaker_id}; speaker={d.speaker})" if d.speaker_id else ""
                delivery = (
                    f" type={d.speech_type}; speaker_visible={str(d.speaker_visible).lower()}; "
                    f"lip_sync={d.lip_sync}"
                )
                match = f"; character_match={d.character_match}" if d.character_match else ""
                lines.append(f'  - "{d.text}"{meta}{sp};{delivery}{match}')
        if self.onscreen_text:
            lines.append("onscreen_text (verbatim):")
            for t in self.onscreen_text:
                lines.append(f'  - "{t}"')
        if self.reference_attrs:
            lines.append("reference_attrs:")
            for a in self.reference_attrs:
                lines.append(f"  - {a.ref_tag} [{a.kind}]: {a.attr}")
        if self.ambiguities:
            lines.append(f"ambiguities: {json.dumps(self.ambiguities, ensure_ascii=False)}")
        lines.append("=== END CONTRACT ===")
        return "\n".join(lines)


def _unique_keep_order(items: list[str]) -> list[str]:
    """去空、去重并保序。"""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = (item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def extract_duration_sec(intent: str, fallback: float = 5.0) -> float:
    """从短意图抽取时长秒数，夹到 4–15；未写则用 fallback。"""
    m = DURATION_RE.search(intent or "")
    if not m:
        return float(max(4, min(15, fallback)))
    return float(max(4, min(15, int(m.group(1)))))


_ROLE_PREFIX_RE = re.compile(
    r"^(球员|厨师|舞者|骑手|店员|技术员|女孩|老人|她|他|运动员|主角)(.+)$"
)


def _peel_subject_prefix(parts: list[str]) -> list[str]:
    """若首段以常见角色主语开头，剥掉主语保留动作。"""
    if len(parts) < 2:
        return parts
    m = _ROLE_PREFIX_RE.match(parts[0])
    if not m:
        return parts
    action = m.group(2).strip()
    if not action:
        return parts
    return [action, *parts[1:]]


def extract_action_chain(intent: str) -> list[str]:
    """从「A→B→C」或「A->B->C」类写法抽取有序动作链；无箭头则空。"""
    text = intent or ""
    # 优先匹配显式箭头链
    for m in re.finditer(
        r"([\u4e00-\u9fffA-Za-z0-9]{1,16}(?:\s*(?:→|->|—|–)\s*[\u4e00-\u9fffA-Za-z0-9]{1,16}){1,8})",
        text,
    ):
        parts = [p for p in ARROW_SPLIT_RE.split(m.group(1)) if p.strip()]
        if len(parts) >= 2:
            return _unique_keep_order(_peel_subject_prefix(parts))
    return []


def extract_forbidden(intent: str) -> list[str]:
    """用确定性模式抽出否定约束原文片段（不发明）。"""
    text = intent or ""
    hits: list[str] = []
    # 整句级：带「禁止/不要/不得」的短片段
    for m in re.finditer(r"[^，。；;\n]{0,8}(?:禁止|不要|不得|勿)[^，。；;\n]{0,16}", text):
        frag = m.group(0).strip(" ，。；;")
        if frag:
            hits.append(frag)
    for rx, _label in FORBIDDEN_PATTERNS:
        if rx.search(text):
            # 用命中的原文片段，而非标签
            m = rx.search(text)
            if m:
                hits.append(m.group(0))
    return _unique_keep_order(hits)


def extract_shot_constraint(intent: str) -> ShotConstraint:
    """抽取镜头约束：单镜头 / 最大镜头数。"""
    text = intent or ""
    single = bool(
        re.search(r"禁止切镜|不得切镜|不要切镜|单镜头|一镜到底|不要分镜|不得分镜", text)
    )
    max_shots: int | None = 1 if single else None
    m = re.search(r"(?:最多|不超过)\s*(\d+)\s*(?:个)?(?:镜头|切镜)", text)
    if m:
        max_shots = int(m.group(1))
        if max_shots == 1:
            single = True
    return ShotConstraint(single_shot=single, max_shots=max_shots)


def extract_explicit_style(intent: str) -> str | None:
    """抽取用户显式风格；未写则 None。"""
    text = intent or ""
    for rx, label in STYLE_PATTERNS:
        if rx.search(text):
            m = rx.search(text)
            return (m.group(0) if m else label).strip()
    return None


def extract_explicit_negatives(intent: str) -> list[str]:
    """抽取用户显式排除的风格。"""
    text = intent or ""
    hits: list[str] = []
    for rx, _label in NEGATIVE_STYLE_PATTERNS:
        m = rx.search(text)
        if m:
            hits.append(m.group(0))
    return _unique_keep_order(hits)


def extract_onscreen_extended(intent: str) -> list[str]:
    """屏上文字：复用 verify 抽取，并补「口号」类引号。"""
    base = list(extract_locked_onscreen(intent))
    extra: list[str] = []
    for m in SLOGAN_RE.finditer(intent or ""):
        line = (m.group(1) or "").strip()
        if line:
            extra.append(line)
    return _unique_keep_order(base + extra)

def _dialogue_language(text: str) -> str:
    """按原文字符确定 H3 语言标签；中文优先，纯拉丁文本为英文。"""
    if re.search(r"[\u4e00-\u9fff]", text or ""):
        return "Chinese"
    if re.search(r"[A-Za-z]", text or ""):
        return "English"
    return ""


def _normalize_speaker_name(name: str) -> str:
    """去掉紧邻发言动词的时序副词，保留实际人物名称。"""
    value = (name or "").strip()
    if re.search(r"[\u4e00-\u9fff]", value):
        value = re.sub(
            r"^(?:然后|随后|接着|这时|而|但|最后|最终|起初|首先|紧接着)+", "", value
        )
        value = re.sub(r"(?:先|再次|再|又|继续|随即|立即)$", "", value)
    else:
        value = re.sub(r"^(?:then|next|finally|first)\s+", "", value, flags=re.I)
        value = re.sub(r"\s+(?:then|again|next)$", "", value, flags=re.I)
    return value.strip(" ，,:：")


_ZH_SPEECH_VERBS = (
    r"说道|低声说|大声说|回答|回应|喊道|问道|大喊|补充|怒吼|质问|反问|叫道|说"
)
_EN_SPEECH_VERBS = r"says?|asks?|replies?|shouts?|whispers?|answers?"
_ZH_CHARACTER_RE = re.compile(
    r"女孩|男孩|女人|男人|老人|孩子|妈妈|爸爸|母亲|父亲|"
    r"老师|学生|医生|护士|店员|顾客|主持人|记者|警察|和尚|唐僧|小妖|"
    r"(?<!其)她(?!们)|(?<!其)他(?!们)"
)
_EN_CHARACTER_RE = re.compile(
    r"\b(?:the|a|an)\s+(?:woman|man|girl|boy|child|mother|father|teacher|student|"
    r"doctor|nurse|clerk|customer|host|reporter|officer|monk)\b", re.I
)


def _speaker_before_quote(intent: str, quote_pos: int) -> str:
    """从引号前的局部上下文抽取明确说话人；不跨句猜测。"""
    prefix = (intent or "")[max(0, quote_pos - 64) : quote_pos]
    zh = re.search(
        rf"([\u4e00-\u9fffA-Za-z0-9·]{{1,16}}?)(?:{_ZH_SPEECH_VERBS})\s*[：:,，]?\s*\Z",
        prefix,
    )
    if zh:
        return _normalize_speaker_name(zh.group(1))
    en = re.search(
        rf"([A-Za-z][A-Za-z0-9 _-]{{0,30}}?)\s+(?:{_EN_SPEECH_VERBS})\s*[:,-]?\s*\Z",
        prefix,
        re.I,
    )
    return _normalize_speaker_name(en.group(1)) if en else ""


def _speaker_after_quote(intent: str, quote_end: int) -> str:
    """识别“台词”，女孩说 / “Dialogue,” says the woman 等后置归属。"""
    suffix = (intent or "")[quote_end : quote_end + 64]
    zh = re.match(
        rf"\s*[，,。.!！？?]?\s*([\u4e00-\u9fffA-Za-z0-9·]{{1,16}}?)(?:{_ZH_SPEECH_VERBS})(?=[，,。.!！？?；;\s]|\Z)",
        suffix,
    )
    if zh:
        return _normalize_speaker_name(zh.group(1))
    en_verb_first = re.match(
        rf"\s*[,.-]?\s*(?:{_EN_SPEECH_VERBS})\s+([A-Za-z][A-Za-z0-9 _-]{{0,30}}?)(?=[,.!?;]|\Z)",
        suffix,
        re.I,
    )
    if en_verb_first:
        return _normalize_speaker_name(en_verb_first.group(1))
    en_name_first = re.match(
        rf"\s*[,.-]?\s*([A-Za-z][A-Za-z0-9 _-]{{0,30}}?)\s+(?:{_EN_SPEECH_VERBS})(?=[,.!?;]|\Z)",
        suffix,
        re.I,
    )
    return _normalize_speaker_name(en_name_first.group(1)) if en_name_first else ""


def _single_character_candidate(intent: str) -> str:
    """仅当台词外叙述只有一个高置信人物时，允许自动绑定。"""
    chars = list(intent or "")
    for start, end, _line in extract_locked_dialogue_spans(intent):
        chars[start:end] = " " * (end - start)
    text = "".join(chars)
    zh = [m.group(0) for m in _ZH_CHARACTER_RE.finditer(text)]
    concrete = [x for x in zh if x not in {"她", "他"}]
    concrete.extend(m.group(0) for m in _EN_CHARACTER_RE.finditer(text))
    unique: dict[str, str] = {}
    for candidate in concrete:
        key = re.sub(r"^(?:the|a|an)\s+", "", candidate, flags=re.I).casefold()
        unique.setdefault(key, candidate)
    if len(unique) == 1:
        return next(iter(unique.values()))
    if unique:
        return ""
    pronouns = _unique_keep_order([x for x in zh if x in {"她", "他"}])
    return pronouns[0] if len(pronouns) == 1 else ""


STYLE_KEYWORD_RE = re.compile(
    r"^(?:国风3D(?:渲染)?|3D渲染|水墨风格?|赛博朋克|二次元|皮影戏|定格动画|粘土风|真人实拍|电影质感|动漫风格?|像素风|写实风格?)$",
    re.I,
)
STYLE_META_SECTION_RE = re.compile(
    r"(?:\n\s*(?:Visual style|美术规范|风格说明|美术提示词|Style description)\s*[:：].*)\Z",
    re.I | re.S,
)


def extract_dialogue_lines(intent: str) -> list[DialogueLine]:
    """抽取有序对白事件：逐字、语言、说话人、归属来源及稳定 S-ID。
    
    严格限制：仅当引号前后存在明确发言动词或台词标记时才识别为对白。
    画风描述、物理属性或强调词（如 “完整一体结构”、“completely bald head”）绝不识别为对白。
    """
    clean_intent = STYLE_META_SECTION_RE.sub("", intent or "")
    onscreen = set(extract_onscreen_extended(clean_intent))
    style = extract_explicit_style(intent)
    lines: list[DialogueLine] = []
    speaker_ids: dict[str, str] = {}
    for start, end, text in extract_locked_dialogue_spans(clean_intent):
        if text in onscreen:
            continue
        if STYLE_KEYWORD_RE.match(text) or text.endswith("风格") or text.endswith("画风"):
            continue
        if style and (text in style or style in text):
            continue
        speaker = _speaker_before_quote(clean_intent, start)
        attribution = "explicit_before" if speaker else ""
        if not speaker:
            speaker = _speaker_after_quote(clean_intent, end)
            attribution = "explicit_after" if speaker else ""
        
        # 必须有明确的前置或后置发言动词/显式台词标记，才视为对白
        if not speaker:
            prefix = clean_intent[max(0, start - 20) : start]
            if re.search(r"(?:对白|台词|台词是|台词为|念道|高喊|叫道|自语|旁白|解说|画外音|内心独白)[:：]\s*$", prefix):
                speaker = "旁白" if re.search(r"旁白|解说", prefix) else ""
                attribution = "explicit_cue"
            else:
                # 无任何发言动作，纯属描述/强调/关键词，跳过，不当做对白
                continue

        speaker_id = ""
        if speaker:
            speaker_key = re.sub(
                r"^(?:the|a|an)\s+", "", speaker, flags=re.I
            ).casefold()
            if speaker_key not in speaker_ids:
                speaker_ids[speaker_key] = f"S{len(speaker_ids) + 1}"
            speaker_id = speaker_ids[speaker_key]
        speech_type, speaker_visible, lip_sync = normalize_speech_metadata(
            "", speaker=speaker, context=clean_intent[max(0, start - 40) : start]
        )
        lines.append(
            DialogueLine(
                text=text,
                language=_dialogue_language(text),
                speaker=speaker,
                speaker_id=speaker_id,
                speech_type=speech_type,
                speaker_visible=speaker_visible,
                lip_sync=lip_sync,
                attribution=attribution,
            )
        )
    return lines


def unresolved_dialogue_lines(contract: IntentContract) -> list[DialogueLine]:
    """返回无法安全绑定人物的台词。"""
    return [line for line in contract.dialogue if not line.speaker_id]


def _dialogue_free_narration(intent: str) -> str:
    chars = list(intent or "")
    for start, end, _line in extract_locked_dialogue_spans(intent):
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def _speaker_is_grounded(speaker: str, narration: str) -> bool:
    value = (speaker or "").strip()
    if not value:
        return False
    if re.search(r"[A-Za-z]", value):
        return value.casefold() in narration.casefold()
    return value in narration


def resolve_dialogue_speakers(contract: IntentContract, chat: ChatFn) -> IntentContract:
    """用低温专用模型为未归属台词做最佳匹配，并固化到 Voice Contract。"""
    unresolved = [i for i, line in enumerate(contract.dialogue) if not line.speaker_id]
    if not unresolved:
        return contract
    system = load_prompt("resolve_speakers")
    user = json.dumps(
        {
            "intent": contract.intent_raw,
            "dialogue": [
                {
                    "index": i + 1,
                    "text": line.text,
                    "language": line.language,
                    "known_speaker": line.speaker or None,
                    "speech_type": line.speech_type,
                    "speaker_visible": line.speaker_visible,
                    "character_match": line.character_match or None,
                }
                for i, line in enumerate(contract.dialogue)
            ],
        },
        ensure_ascii=False,
        indent=2,
    )
    payload: dict[str, Any] = {}
    try:
        payload = _parse_json_obj(chat(system, user, stage="resolve_speakers"))
    except Exception:  # noqa: BLE001 - 失败时仍继续增强，使用稳定兜底
        payload = {}
    raw_assignments = payload.get("assignments") or []
    assignments: dict[int, dict[str, Any]] = {}
    if isinstance(raw_assignments, list):
        for item in raw_assignments:
            if not isinstance(item, dict):
                continue
            try:
                assignments[int(item.get("index")) - 1] = item
            except (TypeError, ValueError):
                continue

    narration = _dialogue_free_narration(contract.intent_raw)
    candidates = [m.group(0) for m in _ZH_CHARACTER_RE.finditer(narration)]
    candidates.extend(m.group(0) for m in _EN_CHARACTER_RE.finditer(narration))
    candidates = _unique_keep_order(candidates)
    speaker_ids: dict[str, str] = {}
    for line in contract.dialogue:
        if line.speaker and line.speaker_id:
            key = re.sub(r"^(?:the|a|an)\s+", "", line.speaker, flags=re.I).casefold()
            speaker_ids.setdefault(key, line.speaker_id)

    fallback_used = False
    for index in unresolved:
        line = contract.dialogue[index]
        item = assignments.get(index) or {}
        proposed = str(item.get("speaker") or "").strip()
        if line.speech_type == "narration":
            # 独立旁白是音频身份，绝不能为了归属而绑定或发明画面人物。
            speaker = line.speaker or ("旁白" if line.language == "Chinese" else "Narrator")
            attribution = "narrator_audio_identity"
            confidence = 1.0
        elif _speaker_is_grounded(proposed, narration):
            speaker = proposed
            attribution = "llm_best_match"
            try:
                confidence = max(0.0, min(1.0, float(item.get("confidence"))))
            except (TypeError, ValueError):
                confidence = None
        else:
            fallback_used = True
            speaker = candidates[0] if candidates else ("旁白" if line.language == "Chinese" else "Narrator")
            attribution = "fallback_first_character" if candidates else "fallback_narrator"
            confidence = None
        key = re.sub(r"^(?:the|a|an)\s+", "", speaker, flags=re.I).casefold()
        if key not in speaker_ids:
            speaker_ids[key] = f"S{len(speaker_ids) + 1}"
        line.speaker = speaker
        line.speaker_id = speaker_ids[key]
        line.attribution = attribution
        line.attribution_confidence = confidence

    contract.ambiguities = [
        note for note in contract.ambiguities if not note.startswith("台词说话人不明确:")
    ]
    if fallback_used:
        contract.ambiguities.append("说话人模型匹配不可用或无效，已使用稳定兜底归属")
    return contract


def detect_ambiguities(intent: str, shot: ShotConstraint) -> list[str]:
    """只记录不猜测：检测自相矛盾表述。"""
    text = intent or ""
    notes: list[str] = []
    multi_cut = bool(
        re.search(
            r"三个不同场景|多场景|场景切换|切到|跳切|多个?机位|快速切换|四个机位",
            text,
        )
    )
    if shot.single_shot and multi_cut:
        notes.append("单镜头/一镜到底/禁止切镜 与 多场景或跳切 冲突")
    if re.search(r"不要配乐", text) and re.search(r"(?:要|加|带)配乐", text):
        notes.append("不要配乐 与 要配乐 冲突")
    return notes


def extract_must_elements_heuristic(intent: str) -> list[str]:
    """极简启发式主体抽取：仅用于无 LLM 时的离线兜底，宁缺毋滥。"""
    text = (intent or "").strip()
    if not text:
        return []
    hits: list[str] = []
    for m in re.finditer(
        r"(?:一只|一个|一位|一名|一条|一辆|一架)([\u4e00-\u9fffA-Za-z]{1,12})",
        text,
    ):
        hits.append(m.group(0).strip())
    stop = {"禁止", "不要", "不得", "单镜头", "必须", "同时", "风格", "动作", "约秒", "小时", "分钟"}
    filtered = [h for h in hits if h not in stop and len(h) >= 2]
    return _unique_keep_order(filtered)[:8]


def parse_intent_deterministic(intent: str, mode: str = "t2va") -> IntentContract:
    """纯确定性抽取（无 LLM）：对白/屏上字/时长/否定/镜头/动作链/风格。

    用于单测与离线 gate；must_elements 仅弱启发式。
    """
    raw = (intent or "").strip()
    shot = extract_shot_constraint(raw)
    dialogue = extract_dialogue_lines(raw)
    onscreen = extract_onscreen_extended(raw)
    # 口号若已在 onscreen，从 dialogue 再滤一次
    onscreen_set = set(onscreen)
    style = extract_explicit_style(raw)
    if style:
        dialogue = [d for d in dialogue if d.text not in style and style not in d.text and not d.text.endswith("风格") and not d.text.endswith("画风")]
    return IntentContract(
        must_elements=extract_must_elements_heuristic(raw),
        forbidden=extract_forbidden(raw),
        dialogue=dialogue,
        onscreen_text=onscreen,
        shot_constraint=shot,
        duration_sec=extract_duration_sec(raw),
        action_chain=extract_action_chain(raw),
        explicit_style=style,
        explicit_negatives=extract_explicit_negatives(raw),
        intent_raw=raw,
        mode=mode,
        ambiguities=_unique_keep_order(
            detect_ambiguities(raw, shot)
            + [
                f"台词说话人不明确: {d.text}"
                for d in dialogue
                if not d.speaker_id
            ]
        ),
    )


def _parse_json_obj(raw: str) -> dict[str, Any]:
    """从模型输出中解析 JSON 对象。"""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("parse_intent 输出不是 JSON 对象")
    return payload


def _as_str_list(value: Any) -> list[str]:
    """把任意值规范成去空白字符串列表。"""
    if not value:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    out: list[str] = []
    for item in value:
        if isinstance(item, dict):
            text = str(item.get("text") or item.get("attr") or "").strip()
        else:
            text = str(item).strip()
        if text:
            out.append(text)
    return _unique_keep_order(out)


def contract_from_llm_payload(
    payload: dict[str, Any],
    *,
    intent: str,
    mode: str,
) -> IntentContract:
    """把 Agent(LLM) 结构化 JSON 转为 IntentContract，完全以 Agent 语义理解为准。"""
    must = _as_str_list(payload.get("must_elements"))
    forbidden = _as_str_list(payload.get("forbidden"))
    action = _as_str_list(payload.get("action_chain"))
    style = payload.get("explicit_style")
    if isinstance(style, str):
        style = style.strip() or None
    else:
        style = None
    negatives = _as_str_list(payload.get("explicit_negatives"))
    amb = _as_str_list(payload.get("ambiguities"))

    shot_raw = payload.get("shot_constraint") or {}
    if isinstance(shot_raw, dict):
        single = bool(shot_raw.get("single_shot", False))
        max_shots = shot_raw.get("max_shots", None)
        if max_shots is not None:
            try:
                max_shots = int(max_shots)
            except (TypeError, ValueError):
                max_shots = None
        shot = ShotConstraint(single_shot=single, max_shots=max_shots)
    else:
        shot = ShotConstraint(single_shot=False, max_shots=None)

    dur = payload.get("duration_sec")
    try:
        duration = float(dur) if dur is not None else 5.0
        duration = float(max(4, min(15, duration)))
    except (TypeError, ValueError):
        duration = 5.0

    # dialogue 由专用 Dialogue Extractor 注入 payload；这里只做子串反幻觉校验与标准化。
    llm_dialogue_raw = payload.get("dialogue")
    dialogue: list[DialogueLine] = []
    if isinstance(llm_dialogue_raw, list):
        for item in llm_dialogue_raw:
            if not isinstance(item, dict):
                continue
            txt = str(item.get("text") or "").strip()
            if not txt or txt not in intent:
                continue
            lang = str(item.get("language") or _dialogue_language(txt)).strip()
            spk = str(item.get("speaker") or "").strip()
            spk_id = str(item.get("speaker_id") or "").strip()
            attribution = str(item.get("attribution") or "dialogue_extractor_api").strip()
            speech_type, speaker_visible, lip_sync = normalize_speech_metadata(
                item.get("speech_type"), speaker=spk
            )
            character_match = str(item.get("character_match") or "").strip()
            dialogue.append(
                DialogueLine(
                    text=txt,
                    language="Chinese" if "chin" in lang.lower() or _CJK_RE.search(txt) else ("English" if "eng" in lang.lower() else lang),
                    speaker=spk,
                    speaker_id=spk_id,
                    speech_type=speech_type,
                    speaker_visible=speaker_visible,
                    lip_sync=lip_sync,
                    character_match=character_match,
                    attribution=attribution,
                )
            )

    # 解析 Agent 提取的 onscreen_text
    llm_onscreen = _as_str_list(payload.get("onscreen_text"))
    onscreen_text = [t for t in llm_onscreen if t in intent]

    return IntentContract(
        must_elements=_unique_keep_order(must),
        forbidden=_unique_keep_order(forbidden),
        dialogue=dialogue,
        onscreen_text=_unique_keep_order(onscreen_text),
        shot_constraint=shot,
        duration_sec=duration,
        action_chain=_unique_keep_order(action),
        explicit_style=style,
        explicit_negatives=_unique_keep_order(negatives),
        intent_raw=(intent or "").strip(),
        mode=mode,
        ambiguities=_unique_keep_order(amb),
    )


def extract_dialogue_specialized(
    intent: str,
    inventory: str | None = None,
    *,
    chat: ChatFn,
) -> list[DialogueLine]:
    """Stage 1 并发调用的专用对话抽取与角色-素材匹配 API。"""
    raw = (intent or "").strip()
    if not raw or chat is None:
        return []
    system = load_prompt("extract_dialogue")
    user_msg = f"User Intent:\n{raw}"
    if inventory:
        user_msg += f"\n\nReference Material Inventory:\n{inventory}"
    try:
        resp = chat(system, user_msg, stage="extract_dialogue")
        data = _parse_json_obj(resp)
        items = data.get("dialogues") or []
        out: list[DialogueLine] = []
        speaker_ids: dict[str, str] = {}
        for idx, it in enumerate(items, 1):
            if not isinstance(it, dict):
                continue
            txt = str(it.get("text") or "").strip()
            if not txt or txt not in raw:
                continue
            lang = str(it.get("language") or _dialogue_language(txt)).strip()
            spk = str(it.get("speaker") or "").strip()
            speech_type, speaker_visible, lip_sync = normalize_speech_metadata(
                it.get("speech_type"), speaker=spk
            )
            if speech_type == "narration" and not spk:
                spk = "旁白" if _CJK_RE.search(txt) else "Narrator"
            if speech_type == "voiceover" and not spk:
                # 没有可绑定角色的画外音按独立旁白处理，避免发明画面人物。
                speech_type, speaker_visible, lip_sync = "narration", False, "none"
                spk = "旁白" if _CJK_RE.search(txt) else "Narrator"
            character_match = str(it.get("character_match") or "").strip()
            spk_id = str(it.get("speaker_id") or "").strip()
            if not spk_id and spk:
                spk_key = re.sub(r"^(?:the|a|an)\s+", "", spk, flags=re.I).casefold()
                if spk_key not in speaker_ids:
                    speaker_ids[spk_key] = f"S{len(speaker_ids) + 1}"
                spk_id = speaker_ids[spk_key]
            out.append(
                DialogueLine(
                    text=txt,
                    language=lang,
                    speaker=spk,
                    speaker_id=spk_id or f"S{idx}",
                    speech_type=speech_type,
                    speaker_visible=speaker_visible,
                    lip_sync=lip_sync,
                    character_match=character_match,
                    attribution="dialogue_extractor_api",
                )
            )
        # 模型可能按说话人或 speech_type 分组；不增加 schema 字段，直接按台词在
        # 原始意图中的出现位置恢复为一条统一发声时间线。重复原句按各自出现次数消费。
        positions_by_text: dict[str, list[int]] = {}
        for line in out:
            if line.text in positions_by_text:
                continue
            positions: list[int] = []
            cursor = 0
            while True:
                pos = raw.find(line.text, cursor)
                if pos < 0:
                    break
                positions.append(pos)
                cursor = pos + max(1, len(line.text))
            positions_by_text[line.text] = positions
        used: dict[str, int] = {}
        decorated: list[tuple[int, int, DialogueLine]] = []
        for original_index, line in enumerate(out):
            occurrence = used.get(line.text, 0)
            used[line.text] = occurrence + 1
            positions = positions_by_text.get(line.text, [])
            position = positions[occurrence] if occurrence < len(positions) else len(raw) + original_index
            decorated.append((position, original_index, line))
        out = [line for _, _, line in sorted(decorated)]

        # S-ID 也以统一时间线中的首次发声顺序为准，不信任模型返回顺序。
        timeline_ids: dict[str, str] = {}
        for line in out:
            speaker_key = re.sub(r"^(?:the|a|an)\s+", "", line.speaker, flags=re.I).casefold()
            speaker_key = speaker_key or f"__anonymous_{len(timeline_ids) + 1}"
            if speaker_key not in timeline_ids:
                timeline_ids[speaker_key] = f"S{len(timeline_ids) + 1}"
            line.speaker_id = timeline_ids[speaker_key]
        return out
    except Exception:
        return []


def parse_intent(
    intent: str,
    mode: str = "t2va",
    *,
    inventory: str | None = None,
    chat: ChatFn | None = None,
    use_llm: bool = True,
) -> IntentContract:
    """解析短意图为 IntentContract。

    在 Stage 1 并发执行专门的对话抽取与角色匹配 API，精准锚定对白与语种。
    """
    raw = (intent or "").strip()
    if not use_llm or chat is None:
        return parse_intent_deterministic(raw, mode=mode)

    from concurrent.futures import ThreadPoolExecutor

    system = load_prompt("parse_intent")
    user = (
        f"Mode: {mode}\n"
        "Extract Intent Contract JSON only. Do not invent elements absent from the intent.\n\n"
        f"Short intent:\n{raw}"
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_intent = pool.submit(chat, system, user, stage="parse_intent")
        f_dialogue = pool.submit(extract_dialogue_specialized, raw, inventory=inventory, chat=chat)

        try:
            resp_intent = f_intent.result()
            payload = _parse_json_obj(resp_intent)
        except Exception:
            payload = {}

        try:
            specialized_dialogues = f_dialogue.result()
        except Exception:
            specialized_dialogues = []

    # 台词的唯一真值源是专用 Dialogue Extractor。空数组也必须覆盖，避免通用
    # Intent Contract 模型把 "lips moving as if speaking" 等视觉动作误判为台词。
    authoritative_dialogue = [
        {
            "text": d.text,
            "language": d.language,
            "speaker": d.speaker,
            "speaker_id": d.speaker_id,
            "speech_type": d.speech_type,
            "speaker_visible": d.speaker_visible,
            "lip_sync": d.lip_sync,
            "character_match": d.character_match,
            "attribution": d.attribution or "dialogue_extractor_api",
        }
        for d in specialized_dialogues
    ]

    if not payload:
        contract = parse_intent_deterministic(raw, mode=mode)
        contract.dialogue = specialized_dialogues
        return contract

    # 无条件覆盖任何越界返回的 dialogue 字段；parse_intent.txt 已不再承担台词任务。
    payload["dialogue"] = authoritative_dialogue

    return contract_from_llm_payload(payload, intent=raw, mode=mode)


def assert_verbatim_locks(contract: IntentContract, intent: str) -> None:
    """断言对白/屏上文字均是意图子串（逐字）；供单测与回归。"""
    text = intent or ""
    for d in contract.dialogue:
        if d.text not in text:
            raise AssertionError(f"对白被改写或不在原文中: {d.text!r}")
    for line in contract.onscreen_text:
        if line not in text:
            raise AssertionError(f"屏上文字被改写或不在原文中: {line!r}")
