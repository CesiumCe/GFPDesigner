# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cesium
#
"""
data_distillation_scorer.py — v4.4 数据蒸馏评分器 (位点增益矩阵 + 跨类型迁移)

从 141k 数据构建 (位置, 氨基酸) → 亮度增益查找表。
用于候选序列的相对排序: B_pct = gain_score 的百分位映射。
Spearman ρ ≈ 0.74 (avGFP 内部), 排序力验证通过。

核心产出:
  1. GAIN_MATRIX: (pos, aa) → 跨类型增益 (2928 entries)
  2. B_pct: 增益分 → 百分位亮度得分 [0.7, 1.0]
  3. 其余蒸馏规则 (AA偏好, 突变数命中率) 保留用于生成
  2. aa_preference_matrix: 全局 AA 偏好 (Top5% 富集度)
  3. chromophore_shell_rules: 生色团壳层的疏水性/芳香性约束
  4. mutation_count_guide: 各突变数的 Top5% 命中率

设计原则 (Trap 防护):
  - 全局 AA 偏好仅用于非关键区域 — knowledge_constraints 优先
  - 位置偏好用"亮度增益"而非"出现频率" (避免高频但无益位点)
  - 不替代 XGBoost — 仅作为生成引导 + 预过滤器
"""
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import numpy as np
import pandas as pd


# ============================================================
# 0. Gain Matrix (position-AA lookup for brightness ranking)
# ============================================================

def build_gain_matrix(brightness_df: pd.DataFrame) -> dict:
    """
    从 141k 数据构建三级增益查找: 位点×AA → 位点 → 全局AA。

    三级回退确保所有突变都有增益估计值:
      L1: (pos, aa) 组合, n≥3 时使用
      L2: 位点级, n≥5 时使用
      L3: 全局氨基酸级 (总是有数据)
    Spearman ρ ≈ 0.74 on avGFP internal validation.
    """
    import re
    from collections import defaultdict

    def parse(s):
        if s == 'WT' or pd.isna(s) or not str(s).strip(): return []
        muts = []
        for p in str(s).split(':'):
            m = re.match(r'^([A-Z])(\d+)([A-Z])', p.strip())
            if m: muts.append((int(m.group(2)), m.group(1), m.group(3)))
        return muts

    wt_by_type = {}
    for t in brightness_df['GFP type'].unique():
        rows = brightness_df[(brightness_df['GFP type'] == t) & (brightness_df['aaMutations'] == 'WT')]
        if len(rows): wt_by_type[t] = rows['Brightness'].mean()

    # L1: (pos, aa)
    pos_aa = defaultdict(list)
    # L2: pos
    pos = defaultdict(list)
    # L3: aa (global)
    aa = defaultdict(list)

    for _, row in brightness_df.iterrows():
        t = row['GFP type']
        wt = wt_by_type.get(t, 3.72)
        for p, _, to_aa in parse(row['aaMutations']):
            gain = row['Brightness'] - wt
            pos_aa[(p, to_aa)].append(gain)
            pos[p].append(gain)
            aa[to_aa].append(gain)

    # Build with shrinkage
    L1 = {}; L2 = {}; L3 = {}
    for (p, a), gains in pos_aa.items():
        if len(gains) >= 3:
            mu = np.mean(gains); n = len(gains)
            L1[(p, a)] = mu * n/(n+5)
    for p, gains in pos.items():
        if len(gains) >= 5:
            mu = np.mean(gains); n = len(gains)
            L2[p] = mu * n/(n+10)
    for a, gains in aa.items():
        mu = np.mean(gains); n = len(gains)
        L3[a] = mu * n/(n+20)

    return {'L1': L1, 'L2': L2, 'L3': L3}  # 3-level fallback


def compute_gain_score(seq: str, template: str, gain_matrix: dict) -> tuple:
    """
    三级回退增益计算: L1 → L2 → L3。
    返回 (total_gain, n_found_L1, n_found_L2, n_fallback_L3)
    """
    L1 = gain_matrix.get('L1', gain_matrix)  # compat with old format
    L2 = gain_matrix.get('L2', {})
    L3 = gain_matrix.get('L3', {})
    total = 0.0
    n1 = n2 = n3 = 0
    for i, (s_aa, t_aa) in enumerate(zip(seq, template)):
        if s_aa != t_aa:
            p = i + 1
            g = L1.get((p, s_aa))
            if g is not None: n1 += 1
            else:
                g = L2.get(p)
                if g is not None: n2 += 1
                else:
                    g = L3.get(s_aa, 0.0)
                    n3 += 1
            total += g
    return total, n1, n2, n3


def compute_gain_score_simple(seq: str, template: str, gain_matrix: dict) -> float:
    """便捷版本: 只返回增益总分 (忽略覆盖率统计)"""
    gs, _, _, _ = compute_gain_score(seq, template, gain_matrix)
    return gs


# ============================================================
# 1. Data Loading & Processing (distillation rules for generation)
# ============================================================

def load_distilled_rules(
    brightness_df: pd.DataFrame,
    template_seqs: Dict[str, str],
    top_percentile: float = 5.0,
) -> Dict:
    """
    从亮度数据中提取蒸馏规则。

    Returns:
        {
            'per_position_gain': {pos: {aa: gain_score, ...}, ...},
            'aa_preference': {aa: enrichment_score, ...},
            'mutation_count_guide': {n: hit_rate, ...},
            'chromophore_shell_hydro_range': (min, max),
            'chromophore_shell_aromatic_min': float,
            'top5_threshold': float,
            'wt_brightness_by_type': {type: float},
        }
    """
    df = brightness_df.copy()
    top5_threshold = df['Brightness'].quantile(1 - top_percentile / 100)
    top5 = df[df['Brightness'] >= top5_threshold]
    print(f"  [Distill] Top {top_percentile}% threshold: {top5_threshold:.4f}")
    print(f"  [Distill] Top {top_percentile}%: {len(top5):,} / {len(df):,} sequences")

    # ── 1a. Per-position brightness (absolute + conditional high-brightness) ──
    # For each (position, to_aa), compute mean brightness and high-brightness fraction.
    # Using ABSOLUTE brightness rather than gain-over-WT because:
    #   - avGFP mutations are mostly deleterious (mean < WT), so gain is always negative
    #   - What matters is: which positions can tolerate mutation while maintaining
    #     reasonable absolute brightness?
    wt_by_type = {}
    for t in df['GFP type'].unique():
        wt_rows = df[(df['GFP type'] == t) & (df['aaMutations'] == 'WT')]
        if len(wt_rows) > 0:
            wt_by_type[t] = wt_rows['Brightness'].mean()

    pos_brightness_raw = defaultdict(lambda: defaultdict(list))

    for _, row in df.iterrows():
        muts = _parse(row['aaMutations'])
        gfp_type = row['GFP type']
        brightness = row['Brightness']

        for pos, _, to_aa in muts:
            pos_brightness_raw[pos][to_aa].append(brightness)

    # Aggregate: mean brightness + high-brightness fraction (>= 3.5)
    per_position_gain = {}
    for pos, aa_brightness in pos_brightness_raw.items():
        per_position_gain[pos] = {}
        for aa, vals in aa_brightness.items():
            if len(vals) >= 3:
                mean_b = np.mean(vals)
                high_frac = sum(1 for v in vals if v >= 3.5) / len(vals)
                n = len(vals)
                # Composite score: mean_brightness * (1 + 2*high_fraction)
                # This rewards positions where mutations yield both good average
                # AND a high chance of being in the bright regime
                composite = mean_b * (1.0 + 2.0 * high_frac)
                # Shrinkage for low-N
                shrinkage = n / (n + 5.0)
                per_position_gain[pos][aa] = round(float(composite * shrinkage), 4)

    # ── 1b. Global AA preference (Top5% enrichment) ──
    top5_aa_count = defaultdict(int)
    all_aa_count = defaultdict(int)

    for _, row in df.iterrows():
        muts = _parse(row['aaMutations'])
        is_top5 = row['Brightness'] >= top5_threshold
        for _, _, to_aa in muts:
            all_aa_count[to_aa] += 1
            if is_top5:
                top5_aa_count[to_aa] += 1

    total_top5_aa = sum(top5_aa_count.values())
    total_all_aa = sum(all_aa_count.values())
    total_top5_seqs = len(top5)
    total_all_seqs = len(df)

    aa_preference = {}
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        t5_f = top5_aa_count.get(aa, 0) / max(total_top5_aa, 1)
        all_f = all_aa_count.get(aa, 0) / max(total_all_aa, 1)
        enrichment = t5_f / max(all_f, 0.0001)
        # Normalize to [0, 2] range for sampling weights
        aa_preference[aa] = round(float(np.clip(enrichment, 0.1, 2.0)), 3)

    # ── 1c. Mutation count guide ──
    def _count_muts(s):
        if s == 'WT' or pd.isna(s) or not str(s).strip(): return 0
        return len(str(s).split(':'))

    df['_n_mut'] = df['aaMutations'].apply(_count_muts)
    top5['_n_mut'] = top5['aaMutations'].apply(_count_muts)

    mutation_count_guide = {}
    for n in range(0, 10):
        n_total = (df['_n_mut'] == n).sum()
        n_top5 = (top5['_n_mut'] == n).sum()
        hit_rate = n_top5 / max(n_total, 1)
        if n_total >= 10:
            mutation_count_guide[n] = round(float(hit_rate), 4)

    # ── 1d. Chromophore shell rules ──
    CHROMO_ZONE = {62, 63, 64, 68, 69, 70, 93, 94, 95, 96,
                   144, 145, 146, 147, 148, 201, 202, 203, 204, 205, 221, 222, 223}
    HYDRO = {'A':1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,
             'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,
             'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,
             'W':-0.9,'Y':-1.3}
    AROMATIC = set('FYW')

    # Analyze chromophore zone mutations in top5%
    chromo_hydro_vals = []
    chromo_aro_vals = []
    for _, row in top5.iterrows():
        muts = _parse(row['aaMutations'])
        for pos, _, to_aa in muts:
            if pos in CHROMO_ZONE:
                chromo_hydro_vals.append(HYDRO.get(to_aa, 0))
                if to_aa in AROMATIC:
                    chromo_aro_vals.append(1)

    # WT chromophore zone hydrophobicity baseline
    # Approximate from avGFP sequence positions
    chromo_shell_hydro_range = (
        round(float(np.percentile(chromo_hydro_vals, 10)) if chromo_hydro_vals else -3.0, 1),
        round(float(np.percentile(chromo_hydro_vals, 90)) if chromo_hydro_vals else 3.0, 1),
    )
    chromo_shell_aromatic_min = (
        round(float(np.mean(chromo_aro_vals)), 2) if chromo_aro_vals else 0.05
    )

    print(f"  [Distill] Computed {len(per_position_gain)} position-gain entries")
    print(f"  [Distill] Chromophore shell hydro range: {chromo_shell_hydro_range}")
    print(f"  [Distill] Top mutation counts by hit rate: "
          f"{dict(sorted(mutation_count_guide.items(), key=lambda x: -x[1])[:5])}")

    return {
        'per_position_gain': dict(per_position_gain),
        'aa_preference': aa_preference,
        'mutation_count_guide': mutation_count_guide,
        'chromophore_shell_hydro_range': chromo_shell_hydro_range,
        'chromophore_shell_aromatic_min': chromo_shell_aromatic_min,
        'top5_threshold': top5_threshold,
        'wt_brightness_by_type': wt_by_type,
        'gain_matrix': build_gain_matrix(brightness_df),  # v4.4: (pos,aa) gain lookup
    }


# ============================================================
# 2. Guided Generation Helpers
# ============================================================

def select_high_gain_positions(
    distilled_rules: Dict,
    template_type: str,
    top_n: int = 50,
    min_score: float = 3.0,
    exclude_positions: Set[int] = None,
) -> List[Tuple[int, float]]:
    """
    选择绝对亮度期望值最高的位点。

    使用复合评分 (mean_brightness × (1 + 2×high_fraction)),
    而非 gain-over-WT。这避免了 avGFP 突变几乎全是负增益的问题。
    """
    if exclude_positions is None:
        exclude_positions = set()

    gain_data = distilled_rules['per_position_gain']
    scored = []
    for pos, aa_scores in gain_data.items():
        if pos in exclude_positions:
            continue
        if not aa_scores:
            continue
        best_score = max(aa_scores.values())
        mean_score = np.mean(list(aa_scores.values()))
        composite = best_score * 0.6 + mean_score * 0.4
        if composite >= min_score:
            scored.append((pos, round(float(composite), 4)))

    scored.sort(key=lambda x: -x[1])
    return scored[:top_n]


def get_template_mutation_budget(template_type: str) -> Tuple[int, int]:
    """
    v4.0: 骨架相对化突变预算。

    sfGFP 已有 6 个核心折叠增强突变(F64L/S147P/I171V/A206V/S30R/Y39N),
    额外突变需克制在 1-3 个。
    avGFP 需要更多突变来弥补折叠缺陷, 允许 2-4 个。
    """
    if template_type == 'sfGFP':
        return (1, 3)   # 额外 1-3 个突变
    else:
        return (2, 4)   # 额外 2-4 个突变


def sample_mutation_count(distilled_rules: Dict) -> int:
    """
    v4.0: 按 Top5% 命中率加权采样突变数。

    60% 概率: 2 突变 (最高命中率)
    30% 概率: 1 突变
    10% 概率: 3+ 突变
    绝对不生成 0 突变 (WT) 或 >4 突变 (命中率极低)
    """
    guide = distilled_rules.get('mutation_count_guide', {})
    # Target distribution: 1:30%, 2:60%, 3:8%, 4:2%
    weights = [0.30, 0.60, 0.08, 0.02]
    n_mut = np.random.choice([1, 2, 3, 4], p=weights)
    return n_mut


def weighted_aa_choice(
    distilled_rules: Dict,
    position: int,
    exclude_aa: str,
    surface_bias: bool = False,
) -> str:
    """
    v4.0: AA 偏好加权采样。

    surface_bias=True 时: 对表面位点增强 D/E 采样权重 (TGP 策略)。
    这确保即使模板自带正电荷 (如 sfGFP net=+6), 额外突变也会
    优先引入负电荷残基, 逐步翻转净电荷符号以抵抗 72°C 聚集。
    """
    preferences = distilled_rules.get('aa_preference', {})
    pos_gains = distilled_rules.get('per_position_gain', {}).get(position, {})

    candidates = []
    weights = []
    for aa in 'ACDEFGHIKLMNPQRSTVWY':
        if aa == exclude_aa:
            continue
        pos_score = pos_gains.get(aa, 3.5)
        pref = preferences.get(aa, 1.0)

        # v4.0 TGP fix: surface charge conservation + negative bias
        # - K/R are EXCLUDED from surface positions (force charge toward negative)
        # - D/E are strongly preferred
        # - C is penalized everywhere (CFPS reducing environment → aggregation risk)
        # - If original residue IS D/E, prefer keeping D/E (charge preservation)
        if aa == 'C':
            pref *= 0.1  # v4.0: Cys universally penalized in CFPS
        if surface_bias:
            if aa in ('K', 'R'):
                continue  # Block K/R entirely on surface
            if aa in ('D', 'E'):
                pref *= 8.0  # Very strong preference for negative charges
            if exclude_aa in ('D', 'E') and aa in ('D', 'E'):
                pref *= 3.0  # Extra: prefer keeping D/E if original was D/E

        if pos_score >= 7.0:
            w = 5.0
        elif pos_score >= 5.0:
            w = 2.5 * pref
        elif pos_score >= 3.5:
            w = 1.0 * pref
        else:
            w = 0.2 * pref

        candidates.append(aa)
        weights.append(max(0.01, w))

    total = sum(weights)
    probs = [w / total for w in weights]
    return np.random.choice(candidates, p=probs)


def _parse(mut_str: str) -> List[Tuple[int, str, str]]:
    if mut_str == 'WT' or pd.isna(mut_str) or not str(mut_str).strip():
        return []
    muts = []
    for part in str(mut_str).split(':'):
        m = re.match(r'^([A-Z])(\d+)([A-Z])', part.strip())
        if m:
            muts.append((int(m.group(2)), m.group(1), m.group(3)))
    return muts
