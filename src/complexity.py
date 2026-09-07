"""由 Intent Contract 推导 elaborate/format 复杂度预算（目标词数区间）。"""

from __future__ import annotations

from typing import Any


def estimate_shot_count(contract: Any) -> int:
    """估计镜头数：单镜头=1，否则取 max_shots 或动作链长度。"""
    shot = getattr(contract, "shot_constraint", None)
    if shot is not None and getattr(shot, "single_shot", False):
        return 1
    max_shots = getattr(shot, "max_shots", None) if shot is not None else None
    if max_shots is not None:
        try:
            return max(1, int(max_shots))
        except (TypeError, ValueError):
            pass
    actions = list(getattr(contract, "action_chain", None) or [])
    return max(1, min(6, len(actions) or 1))


def complexity_word_budget(contract: Any) -> tuple[int, int]:
    """按 镜头数 × 主体数 × 动作链长度 × 时长 给出目标英文词数 [lo, hi]。

    简单单镜头短片压下限，避免灌水；多拍多主体抬上限以拉高 EN1/EN5。
    """
    shots = estimate_shot_count(contract)
    subjects = max(1, len(getattr(contract, "must_elements", None) or []))
    actions = max(1, len(getattr(contract, "action_chain", None) or []))
    dur = float(getattr(contract, "duration_sec", None) or 5.0)
    dur = max(4.0, min(15.0, dur))
    # 基准：每「镜头×主体×动作」单元约 35 词，再按时长微调
    units = shots * subjects * actions
    mid = int(35 * units * (0.55 + dur / 12.0))
    mid = max(90, min(520, mid))
    lo = max(80, int(mid * 0.85))
    hi = min(650, int(mid * 1.35))
    if hi < lo + 30:
        hi = lo + 30
    return lo, hi


def format_complexity_budget_block(contract: Any) -> str:
    """生成写入 elaborate USER 的复杂度预算块。"""
    lo, hi = complexity_word_budget(contract)
    shots = estimate_shot_count(contract)
    subjects = max(1, len(getattr(contract, "must_elements", None) or []))
    actions = max(1, len(getattr(contract, "action_chain", None) or []))
    dur = float(getattr(contract, "duration_sec", None) or 5.0)
    return (
        "COMPLEXITY BUDGET (from Intent Contract; target the final scene prose):\n"
        f"- estimated_shots={shots}, must_elements={subjects}, action_steps={actions}, "
        f"duration_sec≈{dur:.0f}\n"
        f"- target_word_count: {lo}-{hi} English words for the enriched scene note "
        "(integrated description body after formatting).\n"
        "- Fill the budget with Anchored + Inferred detail: materials, light direction/quality, "
        "textures, camera type+amplitude+speed, diegetic foley/ambience, spatial layers.\n"
        "- Do NOT pad with Invented significant entities (new people, logos, captions, brands, "
        "locations, dialogue)."
    )
