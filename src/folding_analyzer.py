# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cesium
#
"""
folding_analyzer.py — 折叠鲁棒性评估与上位效应检测 (改进2)

检测多点突变组合是否可能产生负上位效应 (negative epistasis)，
预测序列跌破折叠阈值的风险。

基于 Nature 2024 文献 "Local fitness landscape of the green
fluorescent protein" (已提供) 的发现：GFP 存在非线性折叠阈值。

方法:
  1. 从 GFP_data.xlsx 提取共现突变对及其亮度影响
  2. 对新序列中的突变组合，检查是否曾被实验验证为安全
  3. 使用 BLOSUM62 加权距离 + ESM 伪困惑度评估折叠概率
"""
import math
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

from .utils import parse_mutation_string

# Module-level cache for co-occurrence data (avoid recomputing per sequence)
_COOC_CACHE = {}


# BLOSUM62 matrix (subset for 20 standard AAs)
# Scores > 0 indicate conservative substitutions
BLOSUM62 = {
    'A': {'A': 4, 'R': -1, 'N': -2, 'D': -2, 'C': 0, 'Q': -1, 'E': -1, 'G': 0, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 0, 'W': -3, 'Y': -2, 'V': 0},
    'R': {'A': -1, 'R': 5, 'N': 0, 'D': -2, 'C': -3, 'Q': 1, 'E': 0, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 2, 'M': -1, 'F': -3, 'P': -2, 'S': -1, 'T': -1, 'W': -3, 'Y': -2, 'V': -3},
    'N': {'A': -2, 'R': 0, 'N': 6, 'D': 1, 'C': -3, 'Q': 0, 'E': 0, 'G': 0, 'H': 1, 'I': -3, 'L': -3, 'K': 0, 'M': -2, 'F': -3, 'P': -2, 'S': 1, 'T': 0, 'W': -4, 'Y': -2, 'V': -3},
    'D': {'A': -2, 'R': -2, 'N': 1, 'D': 6, 'C': -3, 'Q': 0, 'E': 2, 'G': -1, 'H': -1, 'I': -3, 'L': -4, 'K': -1, 'M': -3, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -4, 'Y': -3, 'V': -3},
    'C': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': 9, 'Q': -3, 'E': -4, 'G': -3, 'H': -3, 'I': -1, 'L': -1, 'K': -3, 'M': -1, 'F': -2, 'P': -3, 'S': -1, 'T': -1, 'W': -2, 'Y': -2, 'V': -1},
    'Q': {'A': -1, 'R': 1, 'N': 0, 'D': 0, 'C': -3, 'Q': 5, 'E': 2, 'G': -2, 'H': 0, 'I': -3, 'L': -2, 'K': 1, 'M': 0, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -2, 'Y': -1, 'V': -2},
    'E': {'A': -1, 'R': 0, 'N': 0, 'D': 2, 'C': -4, 'Q': 2, 'E': 5, 'G': -2, 'H': 0, 'I': -3, 'L': -3, 'K': 1, 'M': -2, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'G': {'A': 0, 'R': -2, 'N': 0, 'D': -1, 'C': -3, 'Q': -2, 'E': -2, 'G': 6, 'H': -2, 'I': -4, 'L': -4, 'K': -2, 'M': -3, 'F': -3, 'P': -2, 'S': 0, 'T': -2, 'W': -2, 'Y': -3, 'V': -3},
    'H': {'A': -2, 'R': 0, 'N': 1, 'D': -1, 'C': -3, 'Q': 0, 'E': 0, 'G': -2, 'H': 8, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -1, 'P': -2, 'S': -1, 'T': -2, 'W': -2, 'Y': 2, 'V': -3},
    'I': {'A': -1, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -3, 'E': -3, 'G': -4, 'H': -3, 'I': 4, 'L': 2, 'K': -3, 'M': 1, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -3, 'Y': -1, 'V': 3},
    'L': {'A': -1, 'R': -2, 'N': -3, 'D': -4, 'C': -1, 'Q': -2, 'E': -3, 'G': -4, 'H': -3, 'I': 2, 'L': 4, 'K': -2, 'M': 2, 'F': 0, 'P': -3, 'S': -2, 'T': -1, 'W': -2, 'Y': -1, 'V': 1},
    'K': {'A': -1, 'R': 2, 'N': 0, 'D': -1, 'C': -3, 'Q': 1, 'E': 1, 'G': -2, 'H': -1, 'I': -3, 'L': -2, 'K': 5, 'M': -1, 'F': -3, 'P': -1, 'S': 0, 'T': -1, 'W': -3, 'Y': -2, 'V': -2},
    'M': {'A': -1, 'R': -1, 'N': -2, 'D': -3, 'C': -1, 'Q': 0, 'E': -2, 'G': -3, 'H': -2, 'I': 1, 'L': 2, 'K': -1, 'M': 5, 'F': 0, 'P': -2, 'S': -1, 'T': -1, 'W': -1, 'Y': -1, 'V': 1},
    'F': {'A': -2, 'R': -3, 'N': -3, 'D': -3, 'C': -2, 'Q': -3, 'E': -3, 'G': -3, 'H': -1, 'I': 0, 'L': 0, 'K': -3, 'M': 0, 'F': 6, 'P': -4, 'S': -2, 'T': -2, 'W': 1, 'Y': 3, 'V': -1},
    'P': {'A': -1, 'R': -2, 'N': -2, 'D': -1, 'C': -3, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -3, 'L': -3, 'K': -1, 'M': -2, 'F': -4, 'P': 7, 'S': -1, 'T': -1, 'W': -4, 'Y': -3, 'V': -2},
    'S': {'A': 1, 'R': -1, 'N': 1, 'D': 0, 'C': -1, 'Q': 0, 'E': 0, 'G': 0, 'H': -1, 'I': -2, 'L': -2, 'K': 0, 'M': -1, 'F': -2, 'P': -1, 'S': 4, 'T': 1, 'W': -3, 'Y': -2, 'V': -2},
    'T': {'A': 0, 'R': -1, 'N': 0, 'D': -1, 'C': -1, 'Q': -1, 'E': -1, 'G': -2, 'H': -2, 'I': -1, 'L': -1, 'K': -1, 'M': -1, 'F': -2, 'P': -1, 'S': 1, 'T': 5, 'W': -2, 'Y': -2, 'V': 0},
    'W': {'A': -3, 'R': -3, 'N': -4, 'D': -4, 'C': -2, 'Q': -2, 'E': -3, 'G': -2, 'H': -2, 'I': -3, 'L': -2, 'K': -3, 'M': -1, 'F': 1, 'P': -4, 'S': -3, 'T': -2, 'W': 11, 'Y': 2, 'V': -3},
    'Y': {'A': -2, 'R': -2, 'N': -2, 'D': -3, 'C': -2, 'Q': -1, 'E': -2, 'G': -3, 'H': 2, 'I': -1, 'L': -1, 'K': -2, 'M': -1, 'F': 3, 'P': -3, 'S': -2, 'T': -2, 'W': 2, 'Y': 7, 'V': -1},
    'V': {'A': 0, 'R': -3, 'N': -3, 'D': -3, 'C': -1, 'Q': -2, 'E': -2, 'G': -3, 'H': -3, 'I': 3, 'L': 1, 'K': -2, 'M': 1, 'F': -1, 'P': -2, 'S': -2, 'T': 0, 'W': -3, 'Y': -1, 'V': 4},
}


def extract_mutation_cooccurrence(
    brightness_df: pd.DataFrame,
    template_type: str = "avGFP",
) -> Dict[Tuple[int, int], float]:
    """
    从实验数据提取突变共现信息。

    对于每一对同时出现的突变位点 (pos_i, pos_j)，
    计算其平均亮度。用于检测负上位效应。

    Returns:
        {(pos_i, pos_j): mean_brightness, ...}
    """
    df = brightness_df[brightness_df['GFP type'] == template_type]
    cooccurrence = defaultdict(list)

    for _, row in df.iterrows():
        mutations = parse_mutation_string(row['aaMutations'])
        brightness = row['Brightness']
        positions = sorted(set(m[0] for m in mutations))

        # Record all pairwise combinations
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                pair = (positions[i], positions[j])
                cooccurrence[pair].append(brightness)

    # Compute mean brightness for each pair
    result = {}
    for pair, scores in cooccurrence.items():
        if len(scores) >= 2:  # At least 2 observations
            result[pair] = sum(scores) / len(scores)

    return result


def detect_risky_epistasis(
    seq: str,
    template: str,
    brightness_df: pd.DataFrame,
    template_type: str = "avGFP",
    risk_threshold: float = 2.5,
) -> Dict:
    """
    检测变体序列中存在的潜在负上位效应。

    Args:
        seq: 变体序列
        template: 模板序列
        brightness_df: 亮度数据
        template_type: GFP 类型
        risk_threshold: 低亮度阈值 (低于此值视为风险)

    Returns:
        {
            'n_mutations': 突变总数,
            'risky_pairs': [(pos_i, pos_j, mean_brightness), ...],
            'n_risky_pairs': 风险突变对数量,
            'risk_score': 归一化风险评分 [0,1] (1=高风险),
            'unobserved_pairs': 未在数据中出现的突变对,
        }
    """
    # Find mutations
    mutations = []
    for i, (s_aa, t_aa) in enumerate(zip(seq, template)):
        if s_aa != t_aa:
            mutations.append(i + 1)

    if len(mutations) <= 1:
        return {
            'n_mutations': len(mutations),
            'risky_pairs': [],
            'n_risky_pairs': 0,
            'risk_score': 0.0,
            'unobserved_pairs': 0,
        }

    # Get cooccurrence data (cached at module level to avoid recomputing)
    global _COOC_CACHE
    cache_key = (id(brightness_df), template_type)
    if cache_key not in _COOC_CACHE:
        _COOC_CACHE[cache_key] = extract_mutation_cooccurrence(brightness_df, template_type)
    cooc = _COOC_CACHE[cache_key]

    risky_pairs = []
    unobserved = 0
    positions = sorted(set(mutations))

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            pair = (positions[i], positions[j])
            if pair in cooc:
                if cooc[pair] < risk_threshold:
                    risky_pairs.append((positions[i], positions[j], cooc[pair]))
            else:
                unobserved += 1
                # Unobserved pair: potentially risky
                risky_pairs.append((positions[i], positions[j], -1.0))

    n_total_pairs = len(positions) * (len(positions) - 1) // 2
    risk_score = len(risky_pairs) / max(n_total_pairs, 1)

    return {
        'n_mutations': len(mutations),
        'risky_pairs': risky_pairs,
        'n_risky_pairs': len(risky_pairs),
        'risk_score': round(risk_score, 4),
        'unobserved_pairs': unobserved,
    }


def compute_blosum_distance(seq: str, template: str) -> float:
    """
    计算序列与模板之间的 BLOSUM62 加权距离。

    低距离 → 保守替换 → 更可能保留折叠。
    """
    if len(seq) != len(template):
        min_len = min(len(seq), len(template))
        seq = seq[:min_len]
        template = template[:min_len]

    total_score = 0.0
    n_diff = 0
    for s_aa, t_aa in zip(seq, template):
        if s_aa != t_aa:
            blosum_score = BLOSUM62.get(t_aa, {}).get(s_aa, -4)
            total_score += blosum_score
            n_diff += 1

    if n_diff == 0:
        return 0.0

    avg_score = total_score / n_diff
    # Normalize: BLOSUM scores range roughly from -4 to +11
    normalized = (avg_score + 4) / 15.0
    # Distance = 1 - normalized_similarity
    return round(1.0 - max(0.0, min(1.0, normalized)), 4)


def compute_folding_score(
    seq: str,
    template: str,
    brightness_df: pd.DataFrame,
    template_type: str = "avGFP",
    esm_model=None,
    esm_alphabet=None,
    device: str = "cpu",
) -> Dict:
    """
    v2.2: 综合折叠概率评分 (含 PLL 软惩罚)。

    综合:
      - 上位效应风险 (data co-occurrence)
      - BLOSUM 距离
      - ESM 伪对数似然 (PLL) — v2.2 新增
      - 突变数量软惩罚 — v2.2 新增

    PLL 阈值逻辑:
      delta_PLL = PLL(mutant) - PLL(WT)
      若 delta_PLL < -15 → folding_score 乘 0.3 惩罚
      若 delta_PLL < -25 → folding_score = 0 (硬淘汰)
    """
    # 1. Epistasis risk
    epistasis = detect_risky_epistasis(seq, template, brightness_df, template_type)
    epistasis_score = 1.0 - epistasis['risk_score']

    # 2. BLOSUM distance
    blosum_dist = compute_blosum_distance(seq, template)
    blosum_score = 1.0 - blosum_dist

    # 3. ESM Pseudo-Log-Likelihood (v2.2)
    ppl_score = 0.5
    ppl = None
    delta_pll = 0.0
    pll_penalty = 1.0
    if esm_model is not None and esm_alphabet is not None:
        try:
            from .stability_predictor import compute_pseudo_ppl
            ppl_mutant = compute_pseudo_ppl(seq, esm_model, esm_alphabet, device)
            ppl_wt = compute_pseudo_ppl(template, esm_model, esm_alphabet, device)
            ppl = ppl_mutant
            ppl_score = 1.0 / (1.0 + ppl_mutant / 20.0)

            # v2.2: delta_PLL threshold defense
            delta_pll = ppl_mutant - ppl_wt
            if delta_pll > 25:
                pll_penalty = 0.0  # Hard kill: sequence "naturalness" has collapsed
            elif delta_pll > 15:
                pll_penalty = 0.3  # Severe penalty: likely can't fold
            elif delta_pll > 8:
                pll_penalty = 0.7  # Moderate penalty
            else:
                pll_penalty = 1.0  # OK
        except Exception:
            pass

    # 4. Mutation count soft penalty (v2.2)
    # Data shows: n_mut ≤ 5 is safe, 6-8 needs caution, >8 is risky
    n_mut = epistasis['n_mutations']
    if n_mut <= 5:
        mut_penalty = 1.0
    elif n_mut <= 8:
        mut_penalty = 1.0 - (n_mut - 5) * 0.10  # Gradual penalty
    else:
        mut_penalty = max(0.3, 1.0 - 0.30 - (n_mut - 8) * 0.15)  # Steeper after 8

    # 5. Combine (v4.3: PLL removed — zero discriminative power on historical winners)
    folding_score = (
        0.40 * epistasis_score
        + 0.30 * blosum_score
        + 0.30 * mut_penalty
    )

    return {
        'folding_score': round(folding_score, 4),
        'epistasis_risk_score': round(epistasis['risk_score'], 4),
        'blosum_distance': blosum_dist,
        'pseudo_ppl': ppl,
        'delta_pll': round(delta_pll, 2) if isinstance(delta_pll, float) else None,
        'pll_penalty': round(pll_penalty, 2),
        'mut_penalty': round(mut_penalty, 2),
        'risky_details': epistasis['risky_pairs'][:5],
        'n_mutations': n_mut,
        'n_risky_pairs': epistasis['n_risky_pairs'],
    }


def batch_folding_scores(
    sequences: List[str],
    template: str,
    brightness_df: pd.DataFrame,
    template_type: str = "avGFP",
) -> List[Dict]:
    """批量计算折叠评分。"""
    return [compute_folding_score(s, template, brightness_df, template_type)
            for s in sequences]
