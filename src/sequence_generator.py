# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cesium
#
"""
sequence_generator.py — 序列生成模块

基于知识引导的策略生成 GFP 变体序列:

策略 A (主要): 基于官方 GFP_data.xlsx 中已知有益突变位点
  - 从 141k+ 条实验数据中提取高频高亮度突变位置
  - 在数据驱动的候选位点上进行组合突变

策略 B (补充): 基于文献/结构的候选位点
  - 使用教程推荐的位点池（靠近生色团、已知稳定性位点等）
  - 保守突变 + 随机突变组合

策略 C (探索): De novo 从头设计
  - 基于 GFP 氨基酸分布随机生成，作为多样性补充
"""
import random
from typing import List, Tuple, Dict, Set, Optional

import pandas as pd

from .utils import (
    STANDARD_AA,
    CHROMOPHORE_REGION_1IDX,
    parse_mutation_string,
    apply_mutations_to_template,
)


def extract_high_value_positions(
    brightness_df: pd.DataFrame,
    template_type: str = "avGFP",
    top_n_positions: int = 50,
    min_brightness: float = 2.0,
    min_occurrence: int = 3,
) -> List[int]:
    """
    从实验亮度数据中提取高频高亮度突变位置。

    Args:
        brightness_df: 亮度数据 DataFrame
        template_type: GFP 模板类型
        top_n_positions: 返回的 top 位置数量
        min_brightness: 最低亮度阈值
        min_occurrence: 该位置在数据中出现的最少次数

    Returns:
        候选位置列表 (1-indexed)
    """
    df = brightness_df[brightness_df['GFP type'] == template_type].copy()
    if df.empty:
        return []

    # Parse all mutations
    position_scores: Dict[int, List[float]] = {}
    for _, row in df.iterrows():
        mutations = parse_mutation_string(row['aaMutations'])
        brightness = row['Brightness']
        for pos, _, _ in mutations:
            if pos not in position_scores:
                position_scores[pos] = []
            position_scores[pos].append(brightness)

    # Score each position by: mean brightness × log(occurrence)
    scored_positions = []
    for pos, scores in position_scores.items():
        if len(scores) >= min_occurrence:
            mean_brightness = sum(scores) / len(scores)
            if mean_brightness >= min_brightness:
                score = mean_brightness * (1 + 0.1 * len(scores))
                scored_positions.append((pos, score, mean_brightness, len(scores)))

    scored_positions.sort(key=lambda x: x[1], reverse=True)

    # Protect chromophore region
    result = [p[0] for p in scored_positions if p[0] not in CHROMOPHORE_REGION_1IDX]
    return result[:top_n_positions]


def extract_beneficial_mutations(
    brightness_df: pd.DataFrame,
    template_type: str = "avGFP",
    min_brightness: float = 2.5,
) -> Dict[int, Dict[str, float]]:
    """
    从实验数据中提取每个位置上的有益突变及其平均亮度。

    Returns:
        {position: {to_aa: avg_brightness, ...}, ...}
        例如: {109: {'D': 2.3, 'G': 2.1}, 145: {'D': 2.5}, ...}
    """
    df = brightness_df[brightness_df['GFP type'] == template_type].copy()
    if df.empty:
        return {}

    mut_data: Dict[int, Dict[str, List[float]]] = {}
    for _, row in df.iterrows():
        mutations = parse_mutation_string(row['aaMutations'])
        brightness = row['Brightness']
        for pos, _, to_aa in mutations:
            if pos not in mut_data:
                mut_data[pos] = {}
            if to_aa not in mut_data[pos]:
                mut_data[pos][to_aa] = []
            mut_data[pos][to_aa].append(brightness)

    # Average and filter
    beneficial = {}
    for pos, aa_scores in mut_data.items():
        if pos in CHROMOPHORE_REGION_1IDX:
            continue
        beneficial[pos] = {}
        for aa, scores in aa_scores.items():
            avg = sum(scores) / len(scores)
            if avg >= min_brightness and len(scores) >= 2:
                beneficial[pos][aa] = round(avg, 4)

    return {p: d for p, d in beneficial.items() if d}


def generate_data_driven_variants(
    template: str,
    template_type: str,
    brightness_df: pd.DataFrame,
    n_variants: int,
    min_len: int = 220,
    max_len: int = 250,
) -> List[str]:
    """
    策略 A: 基于实验数据的有益突变组合生成变体。

    从官方 GFP_data.xlsx 中提取该模板类型的高价值突变，
    进行 1-5 个突变的组合。
    """
    beneficial = extract_beneficial_mutations(brightness_df, template_type)
    positions = list(beneficial.keys())

    if len(positions) < 5:
        # Fall back to high-value position extraction
        positions = extract_high_value_positions(brightness_df, template_type, top_n_positions=30)

    if len(positions) < 5:
        # No data for this template — fall back to knowledge-guided positions
        positions = [10, 30, 64, 68, 69, 70, 71, 72, 101, 105, 109,
                     145, 147, 153, 163, 167, 171, 187, 203, 205, 221, 231, 232, 235]

    n_positions = len(positions)
    variants = []
    seq_list = list(template)

    for _ in range(n_variants):
        n_mutations = random.randint(1, min(5, max(1, n_positions)))
        chosen_positions = random.sample(positions, min(n_mutations, len(positions)))

        new_seq = list(seq_list)
        for pos in chosen_positions:
            if pos in beneficial and beneficial[pos]:
                # Choose best known mutation at this position (with some randomness)
                if random.random() < 0.7:
                    # Pick top mutation
                    best_aa = max(beneficial[pos], key=beneficial[pos].get)
                else:
                    # Random from top-3
                    sorted_aas = sorted(beneficial[pos], key=beneficial[pos].get, reverse=True)
                    best_aa = random.choice(sorted_aas[:3])
            else:
                # Random mutation
                original = seq_list[pos - 1]
                candidates = list(STANDARD_AA - {original})
                best_aa = random.choice(candidates)

            idx = pos - 1
            if 0 <= idx < len(new_seq):
                new_seq[idx] = best_aa

        variant = ''.join(new_seq)
        variant = _ensure_length(variant, min_len, max_len)
        variants.append(variant)

    return variants


def generate_knowledge_guided_variants(
    template: str,
    candidate_positions: List[int],
    n_variants: int,
    min_len: int = 220,
    max_len: int = 250,
) -> List[str]:
    """
    策略 B: 基于知识引导的候选位点组合突变。

    候选位点来自:
      - 文献报道的稳定性/亮度相关位点
      - 靠近生色团的位点
      - 历届优胜序列中的突变位点

    突变类型:
      - 40% 保守突变 (理化性质相似)
      - 30% 随机突变
      - 30% 偏向有益特征 (增加芳香族堆积、优化电荷)
    """
    from .utils import HYDROPATHY, AROMATIC_AA

    variants = []
    seq_list = list(template)
    mutable = [p for p in candidate_positions if p not in CHROMOPHORE_REGION_1IDX]
    mutable = [p for p in mutable if 1 <= p <= len(template)]

    if len(mutable) < 3:
        mutable = list(range(2, len(template) + 1))

    for _ in range(n_variants):
        n_mutations = random.randint(1, min(8, len(mutable)))
        chosen = random.sample(mutable, min(n_mutations, len(mutable)))

        new_seq = list(seq_list)
        for pos in chosen:
            idx = pos - 1
            original = new_seq[idx]
            r = random.random()

            if r < 0.4:
                # Conservative: similar hydropathy
                orig_hydro = HYDROPATHY.get(original, 0)
                candidates = list(STANDARD_AA - {original})
                weights = [1.0 / (1.0 + abs(HYDROPATHY.get(aa, 0) - orig_hydro))
                           for aa in candidates]
                total = sum(weights)
                new_seq[idx] = random.choices(candidates, weights=[w/total for w in weights], k=1)[0]
            elif r < 0.7:
                # Random
                candidates = list(STANDARD_AA - {original})
                new_seq[idx] = random.choice(candidates)
            else:
                # Bias toward stabilizing features: prefer aromatics, charged, or Pro
                stabilizing = list((AROMATIC_AA | {'P', 'R', 'K', 'E', 'D'}) - {original})
                if stabilizing:
                    new_seq[idx] = random.choice(stabilizing)
                else:
                    new_seq[idx] = random.choice(list(STANDARD_AA - {original}))

        variant = ''.join(new_seq)
        variant = _ensure_length(variant, min_len, max_len)
        variants.append(variant)

    return variants


def generate_denovo_variants(
    n_variants: int,
    length_range: Tuple[int, int] = (220, 250),
) -> List[str]:
    """
    策略 C: 从头生成序列。
    基于 GFP 家族氨基酸频率分布随机生成。
    """
    gfp_aa_freq = {
        'A': 0.06, 'C': 0.01, 'D': 0.06, 'E': 0.07, 'F': 0.04,
        'G': 0.07, 'H': 0.02, 'I': 0.05, 'K': 0.07, 'L': 0.07,
        'M': 0.02, 'N': 0.04, 'P': 0.04, 'Q': 0.03, 'R': 0.05,
        'S': 0.06, 'T': 0.06, 'V': 0.06, 'W': 0.01, 'Y': 0.05,
    }
    aas = list(gfp_aa_freq.keys())
    weights = list(gfp_aa_freq.values())

    variants = []
    for _ in range(n_variants):
        length = random.randint(*length_range)
        seq = "M" + ''.join(random.choices(aas, weights=weights, k=length - 1))
        variants.append(seq)
    return variants


def generate_topseq_inspired_variants(
    template: str,
    before_top_seqs: pd.DataFrame,
    n_variants: int,
    min_len: int = 220,
    max_len: int = 250,
) -> List[str]:
    """
    策略 D: 基于历届优胜序列的突变模式生成变体。

    分析 beforetopseqs 中优胜序列相对于模板的突变位点，
    在高频突变位上引入类似突变模式。
    """
    variants = []

    # Extract mutation patterns from winning sequences
    all_positions: Dict[int, Dict[str, int]] = {}
    for _, row in before_top_seqs.iterrows():
        winner_seq = str(row['sequence']).strip().upper()
        # Find differences from template
        for i, (taa, waa) in enumerate(zip(template, winner_seq)):
            if taa != waa and i < len(template):
                pos = i + 1
                if pos not in all_positions:
                    all_positions[pos] = {}
                if waa not in all_positions[pos]:
                    all_positions[pos][waa] = 0
                all_positions[pos][waa] += 1

    # Rank positions by how often they're mutated in winning sequences
    hot_positions = sorted(all_positions.keys(),
                           key=lambda p: sum(all_positions[p].values()),
                           reverse=True)

    # Filter chromophore
    hot_positions = [p for p in hot_positions if p not in CHROMOPHORE_REGION_1IDX]

    seq_list = list(template)
    for _ in range(n_variants):
        n_mutations = random.randint(2, 8)
        chosen = random.sample(hot_positions[:min(40, len(hot_positions))],
                               min(n_mutations, len(hot_positions)))

        new_seq = list(seq_list)
        for pos in chosen:
            idx = pos - 1
            if idx >= len(new_seq):
                continue
            # Prefer mutations seen in winners
            if pos in all_positions and random.random() < 0.6:
                popular_aas = sorted(all_positions[pos], key=all_positions[pos].get, reverse=True)
                new_seq[idx] = random.choice(popular_aas[:3])
            else:
                original = new_seq[idx]
                new_seq[idx] = random.choice(list(STANDARD_AA - {original}))

        variant = ''.join(new_seq)
        variant = _ensure_length(variant, min_len, max_len)
        variants.append(variant)

    return variants


def _ensure_length(seq: str, min_len: int, max_len: int) -> str:
    """确保序列长度在允许范围内。"""
    if len(seq) < min_len:
        flexible = ['G', 'S', 'A', 'T', 'N']
        seq = seq + ''.join(random.choice(flexible) for _ in range(min_len - len(seq)))
    elif len(seq) > max_len:
        seq = seq[:max_len]
    return seq


def generate_variants_v4(
    template: str,
    template_type: str,
    distilled_rules: Dict,
    n_variants: int = 1000,
    min_len: int = 220,
    max_len: int = 250,
) -> List[str]:
    """
    v4.0: 数据蒸馏引导的变体生成。

    核心改进:
      - 模板相对化突变预算 (sfGFP:1-3, avGFP:2-4) (Trap 2 修正)
      - 位置亮度增益加权位点选择 (非高频位点) (Trap 3 修正)
      - AA 偏好加权采样 (非全局排除) (Trap 3 修正)
      - 60/30/10 突变数分布 (遵循 Top5% 数据)
      - 生色团 + N 端保护 (knowledge_constraints 优先)

    策略分配:
      - 65% 数据蒸馏引导 (位点增益 + AA偏好)
      - 20% Golden Pair 组合
      - 15% 保守突变探索 (邻近 AA, 小幅体积变化)
    """
    from .data_distillation_scorer import (
        select_high_gain_positions,
        get_template_mutation_budget,
        sample_mutation_count,
        weighted_aa_choice,
    )

    # Template-relative mutation budget
    min_mut, max_mut = get_template_mutation_budget(template_type)
    print(f"  [{template_type}] Mutation budget: {min_mut}-{max_mut} extra mutations")

    # v4.4: Surface positions for TGP charge biasing.
    # Curated from GFP beta-barrel topology (11 strands, outward-facing residues + loops).
    # Removed ~80 inward-facing beta-strand residues that participate in barrel
    # core closure or chromophore pocket packing (these are NOT solvent-exposed).
    # Loops (all solvent-exposed): 2-5, 22-23, 50-58, 76-82, 101-115, 129-145,
    #   155-159, 172-190, 207-229
    # Beta-strand outward-facing subset extracted from strand registers.
    SURFACE = {
        2,3,4,5, 8, 14,15, 18,19, 22,23,
        29,30, 33,34,
        45,46, 50,51,52,53,54,55,56,57,58,
        76,77,78,79,80,81,82,
        101,102,103,104,105,106,108,109,110,111,112,113,114,115,
        117,118,119,120, 122,123,124, 126,127,128,
        129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,
        148,149,150,151,152,
        154,155,156,157,158,159,
        161,162, 164,165,166, 168,169,170,
        172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,
        191,192,193,194,195, 197,198,199,200,
        201,202,203, 206,
        207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229
    }

    # High-gain positions (exclude chromophore region)
    exclude = CHROMOPHORE_REGION_1IDX.copy()
    exclude.add(1)  # Start M
    high_gain_positions = select_high_gain_positions(
        distilled_rules, template_type, top_n=60, exclude_positions=exclude
    )
    gain_positions = [p for p, _ in high_gain_positions]
    print(f"  [{template_type}] High-gain positions: {len(gain_positions)}")

    n_distilled = int(n_variants * 0.65)
    n_golden = int(n_variants * 0.20)
    n_conservative = n_variants - n_distilled - n_golden

    all_variants = []
    seq_list = list(template)

    # ── Strategy A (65%): Data-distilled generation ──
    for _ in range(n_distilled):
        n_mut = min(sample_mutation_count(distilled_rules), max_mut)
        n_mut = max(n_mut, min_mut)

        chosen = random.sample(
            gain_positions,
            min(n_mut, len(gain_positions))
        )

        new_seq = list(seq_list)
        for pos in chosen:
            idx = pos - 1
            if idx < 0 or idx >= len(new_seq):
                continue
            original = new_seq[idx]
            is_surface = pos in SURFACE
            new_aa = weighted_aa_choice(distilled_rules, pos, original, surface_bias=is_surface)
            new_seq[idx] = new_aa

        variant = ''.join(new_seq)
        variant = _ensure_length(variant, min_len, max_len)
        all_variants.append(variant)

    # ── Strategy B (20%): Top-performing Golden Pair seeds ──
    # Use the best position pairs from gain data
    if len(gain_positions) >= 2:
        gain_by_pos = {p: s for p, s in high_gain_positions}
        pairs = []
        for i in range(min(20, len(gain_positions))):
            for j in range(i + 1, min(20, len(gain_positions))):
                p1, p2 = gain_positions[i], gain_positions[j]
                # Prefer diverse positions (far apart in sequence)
                dist = abs(p1 - p2)
                score = gain_by_pos.get(p1, 0) + gain_by_pos.get(p2, 0) + dist * 0.001
                pairs.append((p1, p2, score))
        pairs.sort(key=lambda x: -x[2])

        for _ in range(n_golden):
            if not pairs:
                break
            p1, p2, _ = random.choice(pairs[:30])
            new_seq = list(seq_list)
            if 0 <= p1 - 1 < len(new_seq):
                new_seq[p1 - 1] = weighted_aa_choice(distilled_rules, p1, new_seq[p1 - 1])
            if 0 <= p2 - 1 < len(new_seq):
                new_seq[p2 - 1] = weighted_aa_choice(distilled_rules, p2, new_seq[p2 - 1])
            variant = ''.join(new_seq)
            variant = _ensure_length(variant, min_len, max_len)
            all_variants.append(variant)

    # ── Strategy C (15%): Conservative exploration ──
    from .utils import HYDROPATHY
    for _ in range(n_conservative):
        n_mut = min(2, max_mut)  # Conservative: only 1-2 mutations
        n_mut = max(n_mut, min_mut)
        chosen = random.sample(
            gain_positions[:40] if len(gain_positions) >= 40 else gain_positions,
            min(n_mut, max(1, len(gain_positions)))
        )

        new_seq = list(seq_list)
        for pos in chosen:
            idx = pos - 1
            if idx < 0 or idx >= len(new_seq):
                continue
            original = new_seq[idx]
            orig_hydro = HYDROPATHY.get(original, 0)
            # Conservative: prefer similar hydropathy + volume
            candidates = list(STANDARD_AA - {original})
            weights = [1.0 / (1.0 + abs(HYDROPATHY.get(aa, 0) - orig_hydro))
                       for aa in candidates]
            total = sum(weights)
            new_seq[idx] = random.choices(candidates, weights=[w/total for w in weights], k=1)[0]

        variant = ''.join(new_seq)
        variant = _ensure_length(variant, min_len, max_len)
        all_variants.append(variant)

    # ── v4.0 TGP charge optimization: MODERATE surface charge tuning ──
    # RISK: Extreme negative charge (net >= +14) → CFPS Mg2+ chelation →
    #       translation collapse + brightness crash (v4.0 audit: B dropped 0.84→0.55)
    # FIX: Target MODERATE TGP (net = D+E-K+R = 5~10).
    #       Literature shows net +5~+10 already provides sufficient electrostatic
    #       repulsion for 72°C anti-aggregation without Mg2+ toxicity.
    #       Flip only 2-3 surface K/R→D/E, preserving brightness-optimizing mutations.
    PROTECTED = {1, 30, 39, 64, 65, 66, 67, 68, 99, 153, 163}

    # Reserve 10% of variants for extreme TGP exploration (flip 5, net ~+14)
    n_extreme = max(1, len(all_variants) // 10)

    for idx in range(len(all_variants)):
        seq_list = list(all_variants[idx])
        is_extreme = (idx < n_extreme)  # First 10% = extreme TGP
        max_flips = 5 if is_extreme else random.randint(2, 3)

        flipped = 0
        candidates = [
            p for p in SURFACE
            if p not in PROTECTED
            and p-1 < len(seq_list)
            and seq_list[p-1] in ('K', 'R')
        ]
        for pos in candidates:
            if flipped >= max_flips:
                break
            seq_list[pos - 1] = 'E' if seq_list[pos - 1] == 'K' else 'D'
            flipped += 1

        all_variants[idx] = ''.join(seq_list)

    random.shuffle(all_variants)

    # ── v4.0 C-terminal tail de-aggregation engineering ──
    # TGP paper: C-terminal lattice contacts → aggregation at high T.
    # But eCGP123 C-term (MLPSQAK) ≠ sfGFP C-term (VLLEFVT).
    # Wholesale replacement to GGGSGGG risks disrupting sfGFP-specific
    # C-tail/barrel packing interactions not present in eCGP123.
    #
    # SAFER approach: retain C-term structure, only charge-optimize
    # hydrophobic residues → hydrophilic/charged at positions 219-225.
    # V219→E, L220→Q, L221→E, F223→D, V224→E, T225→E
    # This adds 4 negative charges (D/E) to the tail, enhancing
    # electrostatic repulsion without disrupting backbone geometry.
    CT_MUTATIONS = {219: 'E', 220: 'Q', 221: 'E', 223: 'D', 224: 'E', 225: 'E'}
    HYDROPHOBIC_AA = set('AILMFWV')
    for idx in range(len(all_variants)):
        if idx % 3 == 0:  # Apply to 33% (conservative — preserve WT in majority)
            seq_list = list(all_variants[idx])
            for pos, new_aa in CT_MUTATIONS.items():
                idx_p = pos - 1
                # v4.4: Only replace hydrophobic residues (maintain existing
                # hydrophilic/charged residues — they already provide charge benefit).
                # Avoids disrupting polar interaction networks unique to avGFP C-term.
                if (idx_p < len(seq_list)
                    and seq_list[idx_p] in HYDROPHOBIC_AA
                    and seq_list[idx_p] not in ('D', 'E')):
                    seq_list[idx_p] = new_aa
            all_variants[idx] = ''.join(seq_list)

    n_tgp = sum(1 for v in all_variants
                if (v.count('D')+v.count('E')) - (v.count('K')+v.count('R')) >= 3)
    print(f"  [{template_type}] Generated {len(all_variants)} variants, "
          f"{n_tgp}/{len(all_variants)} TGP-ready (D+E-K+R >= 3)")
    return all_variants


def generate_variants(
    template: str,
    template_type: str,
    brightness_df: pd.DataFrame = None,
    before_top_seqs: pd.DataFrame = None,
    n_variants: int = 2000,
    min_len: int = 220,
    max_len: int = 250,
    distilled_rules: Dict = None,
) -> List[str]:
    """
    综合生成变体池 (v4.0: 数据蒸馏引导)。

    若提供 distilled_rules 则使用 v4.0 数据蒸馏策略,
    否则回退至 v3.3 四策略混合。

    策略分配 (v4.0):
      - 65% 数据蒸馏引导 (位点增益 + AA偏好 + 骨架相对突变预算)
      - 20% Golden Pair 组合 (高增益位点对)
      - 15% 保守突变探索
    """
    if distilled_rules is not None:
        return generate_variants_v4(
            template, template_type, distilled_rules,
            n_variants, min_len, max_len,
        )

    # ── Fallback: v3.3 four-strategy generation ──
    n_data = int(n_variants * 0.50)
    n_knowledge = int(n_variants * 0.25)
    n_topseq = int(n_variants * 0.15)
    n_denovo = n_variants - n_data - n_knowledge - n_topseq

    literature_positions = [
        10, 30, 64, 68, 69, 70, 71, 72,
        101, 105, 109, 145, 147, 153,
        163, 167, 171, 187,
        203, 205, 221, 231, 232, 235,
    ]

    all_variants = []
    if brightness_df is not None and before_top_seqs is not None:
        print(f"  Generating {n_data} data-driven variants...")
        all_variants += generate_data_driven_variants(
            template, template_type, brightness_df, n_data, min_len, max_len
        )
        print(f"  Generating {n_topseq} topseq-inspired variants...")
        all_variants += generate_topseq_inspired_variants(
            template, before_top_seqs, n_topseq, min_len, max_len
        )

    print(f"  Generating {n_knowledge} knowledge-guided variants...")
    all_variants += generate_knowledge_guided_variants(
        template, literature_positions, n_knowledge, min_len, max_len
    )
    print(f"  Generating {n_denovo} de novo variants...")
    all_variants += generate_denovo_variants(n_denovo, (min_len, max_len))

    random.shuffle(all_variants)
    print(f"  Total variants generated: {len(all_variants)}")
    return all_variants
