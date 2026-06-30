# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cesium
#
"""
filter_selector.py — 约束过滤与序列筛选模块 (改进版)

新增 (改进2+4):
  - 折叠评分过滤 (min_folding_score)
  - 知识约束评分 (文献先验)
"""
from typing import List, Dict, Tuple

import numpy as np

from .utils import validate_sequence


def filter_by_constraints(
    sequences: List[Dict],
    exclusion_list: List[str],
    min_len: int = 220,
    max_len: int = 250,
    brightness_cutoff: float = 0.3,
    folding_score_min: float = 0.30,
    brightness_floor: float = 0.70,
) -> Tuple[List[Dict], List[Dict]]:
    """
    硬约束过滤 (改进版)。

    新增步骤:
      4. 折叠评分检查 (改进2)
      5. 知识约束评分标记 (改进4 — 不淘汰，仅标记)
    """
    exclusion_set = set(exclusion_list)
    passed = []
    rejected = []

    for item in sequences:
        seq = item['sequence']
        reasons = []

        # 1. Sequence validation
        is_valid, err_msg = validate_sequence(seq, min_len, max_len)
        if not is_valid:
            reasons.append(f"序列校验失败: {err_msg}")

        # 2. Exclusion list
        if seq in exclusion_set:
            reasons.append("序列在 Exclusion_List 中")

        # 3. Brightness threshold (v4.4: also accept 'brightness' from deep search results)
        brightness = item.get('predicted_brightness',
                     item.get('brightness',
                     item.get('ensemble_brightness', 0)))
        if brightness < brightness_cutoff:
            reasons.append(f"亮度 {brightness:.4f} 低于淘汰阈值 {brightness_cutoff}")

        # v4.0: Brightness hard floor for product scoring
        # Under strict product scoring (B × S), low-brightness sequences are
        # essentially dead weight — no amount of stability can compensate.
        # Floor: brightness < 0.80 → demoted (cannot enter Top-6).
        if brightness < brightness_floor:
            reasons.append(
                f"亮度 {brightness:.4f} 低于产品制硬底线 {brightness_floor} "
                f"(乘积制下低亮度无法被热稳补偿)"
            )

        # 4. Folding score threshold (改进2)
        folding = item.get('folding_score')
        if folding is not None and folding < folding_score_min:
            reasons.append(
                f"折叠评分 {folding:.4f} 低于阈值 {folding_score_min} "
                f"(存在负上位效应风险)"
            )

        # 5. Knowledge constraints — warning only, no elimination (改进4)
        knowledge = item.get('knowledge_score')
        if knowledge is not None and knowledge < 0.4:
            if 'knowledge_warnings' not in item:
                item['knowledge_warnings'] = []
            item['knowledge_warnings'].append(f"知识约束评分偏低 ({knowledge:.4f})")

        if reasons:
            item_copy = dict(item)
            item_copy['rejection_reasons'] = reasons
            rejected.append(item_copy)
        else:
            passed.append(item)

    return passed, rejected


def select_portfolio_top6(
    sequences: List[Dict],
    score_key: str = 'composite_score',
    delta_b_key: str = 'delta_brightness',
    stability_key: str = 'predicted_stability',
) -> List[Dict]:
    """
    v4.4: 2+2+2 投资组合策略 — 三种阵容各 2 条, 对冲竞赛不确定性。

    阵容 A (防御型, 2条): 亮度损失小 + 稳定性增益大
      - dB >= -0.02 (亮度损失 < 2%), S >= 0.89
    阵容 B (全能型, 2条): 亮度与稳定性兼得
      - 按 composite_score 排序, 排除已选
    阵容 C (进攻型, 2条): 亮度高 + 稳定性可接受
      - 按 delta_brightness 排序, S >= 0.88
    """
    if len(sequences) < 6:
        return sorted(sequences, key=lambda x: x.get(score_key, 0), reverse=True)

    pool = sorted(sequences, key=lambda x: x.get(score_key, 0), reverse=True)
    selected = []
    used_labels = set()

    # ── A: 防御型 — 亮度无损 + 高稳定 ──
    a_candidates = [s for s in pool
                    if s.get(delta_b_key, -99) >= -0.02
                    and s.get(stability_key, 0) >= 0.89]
    for s in a_candidates[:2]:
        selected.append(s)

    # ── B: 全能型 — 综合分最高 ──
    for s in pool:
        if len(selected) >= 4: break
        if s in selected: continue
        if s.get(stability_key, 0) >= 0.885:
            selected.append(s)

    # ── C: 进攻型 — 亮度最高 ──
    c_candidates = sorted(
        [s for s in pool if s not in selected and s.get(stability_key, 0) >= 0.88],
        key=lambda x: x.get(delta_b_key, -99), reverse=True
    )
    for s in c_candidates[:2]:
        selected.append(s)

    # Fallback: fill any remaining slots by composite
    if len(selected) < 6:
        for s in pool:
            if s not in selected:
                selected.append(s)
            if len(selected) >= 6:
                break

    return selected[:6]


def select_top_n(
    sequences: List[Dict],
    n: int = 6,
    score_key: str = 'composite_score',
    diversity_threshold: float = 0.95,
    portfolio_diversify: bool = True,
) -> List[Dict]:
    """
    按得分选出 Top-N 序列，保障多样性 + 模板多样性 + 策略组合分散 (v3.3)。

    v3.3: 新增投资组合分散策略 — 确保 Top-6 覆盖至少 3 种不同的设计策略:
      - conservative: 突变少(≤3)、依赖 Golden Pairs → 稳健型
      - tgp_aggressive: 高净负电荷(≤-8)、依赖 TGP → 热稳定激进型
      - chromophore_focused: 生色团区域有突变 → 亮度特化型
      - balanced: 其他 → 平衡型
    通过多元化设计哲学对冲 72°C 湿实验的不确定性。
    """
    if len(sequences) <= n:
        return sorted(sequences, key=lambda x: x.get(score_key, 0), reverse=True)

    sorted_seqs = sorted(sequences, key=lambda x: x.get(score_key, 0), reverse=True)

    # v3.3: Classify candidates by design strategy
    def _classify(item: Dict) -> str:
        seq = item.get('sequence', '')
        n_mut = sum(1 for a, b in zip(seq, sorted_seqs[0].get('sequence', ''))
                    if a != b) if sorted_seqs and seq else 0
        if n_mut == 0:
            # Fallback: count from knowledge results
            n_mut = item.get('n_mutations', sum(1 for c in seq if c != 'M'))  # rough

        net_charge = (seq.count('D') + seq.count('E')) - (seq.count('K') + seq.count('R'))
        has_chromo_mut = item.get('has_chromophore_mutation', 0)
        n_risky = item.get('n_risky_pairs', 0)

        if n_mut <= 3 and n_risky == 0:
            return 'conservative'
        elif net_charge <= -8:
            return 'tgp_aggressive'
        elif has_chromo_mut:
            return 'chromophore_focused'
        else:
            return 'balanced'

    # First pass: diversity-filtered selection
    selected = []
    for item in sorted_seqs:
        seq = item['sequence']
        too_similar = False
        for sel in selected:
            if _pairwise_identity(seq, sel['sequence']) > diversity_threshold:
                too_similar = True
                break
        if not too_similar:
            item['_strategy'] = _classify(item)
            selected.append(item)
        if len(selected) >= n:
            break

    # v3.3: Portfolio diversification — ensure coverage of design strategies
    if portfolio_diversify and len(selected) >= n:
        strategies_present = set(item.get('_strategy', 'balanced') for item in selected)
        min_strategies = 2  # At minimum, want 2 different strategies

        if len(strategies_present) < min_strategies:
            # Try to replace the last similar-strategy sequence with a different one
            for i in range(len(selected) - 1, -1, -1):
                current_strat = selected[i].get('_strategy', 'balanced')
                # Find replacement with a different strategy
                for item in sorted_seqs:
                    item_strat = _classify(item)
                    if item_strat not in strategies_present:
                        # Check diversity against all other selections
                        ok = True
                        for j, sel in enumerate(selected):
                            if j != i:
                                if _pairwise_identity(item['sequence'], sel['sequence']) > diversity_threshold:
                                    ok = False
                                    break
                        if ok:
                            item['_strategy'] = item_strat
                            selected[i] = item
                            strategies_present.add(item_strat)
                            break
                if len(strategies_present) >= min_strategies:
                    break

    # v4.0: Template quota — sfGFP priority with avGFP dynamic backup
    # Competition uses Top-1 scoring (best single sequence determines rank).
    # Therefore: prioritize sfGFP (higher baseline brightness) for max ceiling.
    # avGFP sequences only enter if:
    #   - sfGFP candidate pool is exhausted, OR
    #   - An avGFP sequence has exceptionally high composite score (top 20%)
    if len(selected) >= n:
        sf_present = [s for s in selected if s.get('template_type') == 'sfGFP']
        av_present = [s for s in selected if s.get('template_type') == 'avGFP']

        # Compute dynamic threshold: only allow avGFP if its composite score
        # is competitive with sfGFP candidates
        sf_scores = [s.get('composite_score', 0) for s in selected
                     if s.get('template_type') == 'sfGFP']
        sf_median = np.median(sf_scores) if sf_scores else 0
        av_threshold = sf_median * 0.85  # avGFP must be within 85% of sfGFP median

        # Remove avGFP entries that don't meet threshold (unless pool depleted)
        sf_candidates_available = any(
            s.get('template_type') == 'sfGFP' and s not in selected
            for s in sorted_seqs
        )

        if sf_candidates_available and av_present:
            for i in range(len(selected) - 1, -1, -1):
                if selected[i].get('template_type') == 'avGFP':
                    av_score = selected[i].get('composite_score', 0)
                    if av_score < av_threshold:
                        # Replace with best available sfGFP
                        for item in sorted_seqs:
                            if (item.get('template_type') == 'sfGFP'
                                and item not in selected):
                                ok = True
                                for j, sel in enumerate(selected):
                                    if j != i:
                                        if _pairwise_identity(item['sequence'], sel['sequence']) > diversity_threshold:
                                            ok = False
                                            break
                                if ok:
                                    selected[i] = item
                                    break

    if len(selected) < n:
        for item in sorted_seqs:
            if item not in selected:
                item['_strategy'] = _classify(item)
                selected.append(item)
            if len(selected) >= n:
                break

    return selected[:n]


def _infer_template(seq: str) -> str:
    """从序列推断最可能的 GFP 模板 (基于序列一致性)。"""
    from .utils import GFP_TEMPLATES
    if not GFP_TEMPLATES:
        return ""
    best_tpl, best_id = "", 0.0
    for name, tpl_seq in GFP_TEMPLATES.items():
        min_len = min(len(seq), len(tpl_seq))
        if min_len == 0:
            continue
        identity = sum(1 for i in range(min_len) if seq[i] == tpl_seq[i]) / min_len
        if identity > best_id:
            best_id, best_tpl = identity, name
    return best_tpl if best_id > 0.7 else ""


def _pairwise_identity(seq1: str, seq2: str) -> float:
    if not seq1 or not seq2:
        return 0.0
    min_len = min(len(seq1), len(seq2))
    matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
    return matches / min_len


def generate_selection_report(
    selected: List[Dict],
    rejected_count: int,
    total_count: int,
) -> str:
    lines = [
        "=" * 70,
        "         GFP Variant Selection Report (Improved)",
        "=" * 70,
        "",
        f"  Total variants generated : {total_count}",
        f"  Passed all filters       : {total_count - rejected_count}",
        f"  Rejected                 : {rejected_count}",
        f"  Selected for submission  : {len(selected)}",
        "",
        "-" * 70,
        f"  {'Rank':<6} {'Len':<6} {'Brightness':<12} {'Stability':<12} {'Composite':<12} {'Knowledge':<12}",
        "-" * 70,
    ]

    for rank, item in enumerate(selected, 1):
        seq = item['sequence']
        lines.append(
            f"  {rank:<6} {len(seq):<6} "
            f"{item.get('predicted_brightness', item.get('ensemble_brightness', 0)):<12.4f} "
            f"{item.get('predicted_stability', 0):<12.4f} "
            f"{item.get('composite_score', 0):<12.4f} "
            f"{item.get('knowledge_score', 0):<12.4f}"
        )

    lines.append("-" * 70)

    # Show knowledge warnings for selected sequences
    lines.append("")
    lines.append("  Knowledge Constraint Warnings (if any):")
    has_warnings = False
    for rank, item in enumerate(selected, 1):
        if item.get('knowledge_warnings'):
            has_warnings = True
            for w in item['knowledge_warnings']:
                lines.append(f"    Rank {rank}: {w}")
        if item.get('knowledge_bonus'):
            for b in item.get('knowledge_bonus', []):
                lines.append(f"    Rank {rank} [+]: {b}")
    if not has_warnings:
        lines.append("    (none — all sequences pass knowledge checks)")

    return '\n'.join(lines)
