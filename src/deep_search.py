# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cesium
#
"""
deep_search.py — v4.0 深度序列搜索算法

包含两个核心搜索策略:

1. Local Hill-Climbing (局部爬山)
   从 Top-N 候选序列出发，对每个突变位置进行 19 种氨基酸单点扫描，
   保留得分提升的突变。迭代 3-5 轮生成"局部最优"序列。

2. Golden Pairs Mining (正上位网络挖掘)
   从 141k 实验数据中找出"共现且高亮"的突变对。
   在生成新序列时优先组合这些已验证的正上位突变对。

复杂度:
  - 每轮: N_candidates × N_positions × 19_AAs × predict() ≈ 50 × 5 × 19 = 4750 次预测
  - XGBoost 单次预测 ~0.1ms → 每轮 < 1s
  - 3-5 轮总时间 < 10s
"""
import itertools
import random
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

from .utils import STANDARD_AA, CHROMOPHORE_REGION_1IDX, parse_mutation_string


# ============================================================
# 1. Golden Pairs Mining
# ============================================================

def mine_golden_pairs(
    brightness_df: pd.DataFrame,
    template_type: str = "avGFP",
    min_brightness: float = 3.5,
    min_cooccurrence: int = 3,
    top_k: int = 100,
) -> List[Dict]:
    """
    从实验数据挖掘 Golden Pairs — 共现且高亮度的突变对。

    筛选条件:
      - 两个突变必须在同一序列中共同出现 ≥ min_cooccurrence 次
      - 该突变对的平均亮度 ≥ min_brightness
      - 按 (平均亮度 × log(共现次数)) 排序

    Returns:
        [
            {
                'pos_i': int, 'pos_j': int,
                'aa_i': str, 'aa_j': str,
                'mean_brightness': float,
                'cooccurrence': int,
                'score': float,
            },
            ...
        ]
    """
    df = brightness_df[brightness_df['GFP type'] == template_type].copy()
    if df.empty:
        return []

    # Collect all pairwise mutations with brightness
    pair_data = defaultdict(list)

    for _, row in df.iterrows():
        mutations = parse_mutation_string(row['aaMutations'])
        brightness = row['Brightness']

        if len(mutations) < 2:
            continue

        # All pairwise combinations
        for i in range(len(mutations)):
            for j in range(i + 1, len(mutations)):
                pos_i, _, aa_i = mutations[i]
                pos_j, _, aa_j = mutations[j]

                # Canonical ordering: smaller position first
                if pos_i < pos_j:
                    key = (pos_i, aa_i, pos_j, aa_j)
                else:
                    key = (pos_j, aa_j, pos_i, aa_i)

                pair_data[key].append(brightness)

    # Filter and score
    golden_pairs = []
    for (pos_i, aa_i, pos_j, aa_j), scores in pair_data.items():
        if len(scores) < min_cooccurrence:
            continue

        mean_brightness = sum(scores) / len(scores)
        if mean_brightness < min_brightness:
            continue

        # Score: brightness × log(co-occurrence) → rewards both brightness and evidence strength
        score = mean_brightness * (1.0 + 0.2 * np.log(len(scores)))

        golden_pairs.append({
            'pos_i': pos_i, 'aa_i': aa_i,
            'pos_j': pos_j, 'aa_j': aa_j,
            'mean_brightness': round(mean_brightness, 3),
            'cooccurrence': len(scores),
            'score': round(score, 3),
        })

    golden_pairs.sort(key=lambda x: -x['score'])
    return golden_pairs[:top_k]


def apply_golden_pairs(
    template: str,
    golden_pairs: List[Dict],
    n_pairs: int = 2,
) -> str:
    """
    将 Golden Pairs 应用到模板序列上。

    随机选择 n_pairs 个 Golden Pair，将其突变写入序列。
    若两个 pair 使用相同位置，则后者覆盖前者。
    """
    if not golden_pairs:
        return template

    seq_list = list(template)
    selected = random.sample(golden_pairs, min(n_pairs, len(golden_pairs)))

    for pair in selected:
        idx_i = pair['pos_i'] - 1
        idx_j = pair['pos_j'] - 1
        if 0 <= idx_i < len(seq_list):
            seq_list[idx_i] = pair['aa_i']
        if 0 <= idx_j < len(seq_list):
            seq_list[idx_j] = pair['aa_j']

    return ''.join(seq_list)


# ============================================================
# 1b. Random Perturbation (escape local optima)
# ============================================================

def perturb_seeds(
    seed_sequences: List[str],
    template: str,
    n_perturbations: int = 2,
    perturb_fraction: float = 0.3,
) -> List[str]:
    """
    v3.1: 随机扰动种子序列以跳出局部最优。

    对 30% 的种子序列，随机回退 1-2 个突变（恢复为 WT 残基）。
    这增加了种子的多样性，使爬山算法有机会探索不同的局部区域。
    """
    perturbed = list(seed_sequences)
    n_to_perturb = max(1, int(len(seed_sequences) * perturb_fraction))

    indices = random.sample(range(len(seed_sequences)), min(n_to_perturb, len(seed_sequences)))

    for idx in indices:
        seq = seed_sequences[idx]
        # Find positions that differ from template
        diff_positions = []
        for i, (s_aa, t_aa) in enumerate(zip(seq, template)):
            if s_aa != t_aa:
                diff_positions.append(i)

        if len(diff_positions) == 0:
            continue

        # Revert 1-2 random mutations back to WT
        n_revert = min(random.randint(1, n_perturbations), len(diff_positions))
        revert_positions = random.sample(diff_positions, n_revert)

        seq_list = list(seq)
        for pos in revert_positions:
            seq_list[pos] = template[pos]

        perturbed[idx] = ''.join(seq_list)

    return perturbed


# ============================================================
# 2. Local Hill-Climbing Search
# ============================================================

def evaluate_sequence(
    seq: str,
    template: str,
    template_type: str,
    templates: Dict[str, str],
    xgb_predictor: Dict,
    brightness_df: pd.DataFrame,
    exclusion_set: Set[str],
    min_len: int = 220,
    max_len: int = 250,
    esm_model=None,
    esm_alphabet=None,
    device: str = "cpu",
    brightness_floor: float = 0.0,
    distilled_rules: Dict = None,
    gain_g_min: float = -10.0,
    gain_g_max: float = 5.0,
) -> Dict:
    """
    快速评估单条序列的综合得分 (v4.4: gain matrix scoring 与主管线一致)。

    v4.4: 亮度使用 gain_matrix → B_pct (与 main.py 主管线一致),
    替代旧版 XGBoost 亮度预测。这确保 Deep Search 优化的目标函数
    与最终排序的 objective 完全相同。

    Returns:
        {..., 'valid': bool, 'composite': float}
    """
    from .utils import validate_sequence
    from .stability_predictor import _estimate_stability_fallback
    from .folding_analyzer import compute_folding_score
    from .knowledge_constraints import evaluate_knowledge_constraints

    # Validation
    is_valid, _ = validate_sequence(seq, min_len, max_len)
    if not is_valid:
        return {'valid': False, 'composite': -999}

    # Exclusion check
    if seq in exclusion_set:
        return {'valid': False, 'composite': -999, 'reason': 'exclusion_list'}

    # v4.4: Brightness via gain matrix (same as main pipeline final scoring)
    if distilled_rules is not None and 'gain_matrix' in distilled_rules:
        from .data_distillation_scorer import compute_gain_score_simple
        gain_score = compute_gain_score_simple(seq, template, distilled_rules['gain_matrix'])
        if gain_g_max > gain_g_min:
            brightness = round(0.60 + 0.40 * (gain_score - gain_g_min) / (gain_g_max - gain_g_min), 4)
        else:
            brightness = 0.80
    else:
        # Fallback: XGBoost prediction (legacy)
        from .ensemble_predictor import predict_with_xgboost
        raw = predict_with_xgboost([seq], [template_type], templates, xgb_predictor)[0]
        brightness = raw / 3.72

    # v4.0: Brightness floor early exit — skip expensive stability/folding checks
    if brightness_floor > 0 and brightness < brightness_floor:
        return {'valid': False, 'composite': -999, 'brightness': round(brightness, 4),
                'reason': f'brightness {brightness:.3f} < floor {brightness_floor}'}

    # Stability — fast fallback (has TGP charge awareness since v4.0)
    stability = _estimate_stability_fallback(seq, template)

    # Folding
    fold = compute_folding_score(seq, template, brightness_df, template_type)
    folding = fold.get('folding_score', 0.5)

    # Knowledge
    know = evaluate_knowledge_constraints(seq, template_type)
    knowledge = know.get('knowledge_score', 0.5)

    # Aggregation risk
    agg_risk = compute_aggregation_risk(seq)

    # Composite (v4.4: B_pct × S × (1 - 0.3 × agg_risk), same as main pipeline)
    composite = brightness * stability * (1.0 - 0.3 * agg_risk)

    return {
        'sequence': seq,
        'brightness': round(brightness, 4),
        'stability': round(stability, 4),
        'composite': round(composite, 4),
        'folding_score': round(folding, 4),
        'knowledge_score': round(knowledge, 4),
        'aggregation_risk': round(agg_risk, 4),
        'valid': True,
    }


def local_hill_climbing(
    seed_sequences: List[str],
    template: str,
    template_type: str,
    templates: Dict[str, str],
    xgb_predictor: Dict,
    brightness_df: pd.DataFrame,
    exclusion_set: Set[str],
    n_rounds: int = 4,
    explore_positions: Optional[List[int]] = None,
    protect_positions: Optional[Set[int]] = None,
    esm_model=None,
    esm_alphabet=None,
    device: str = "cpu",
    brightness_floor: float = 0.0,
    distilled_rules: Dict = None,
    gain_g_min: float = -10.0,
    gain_g_max: float = 5.0,
) -> List[Dict]:
    """
    局部爬山搜索算法 (v4.4: gain matrix 评分)。

    从种子序列出发，每轮对每个可突变位置尝试 19 种替代氨基酸，
    保留得分提升的突变。迭代 n_rounds 轮。

    Args:
        seed_sequences: Top-N 种子序列列表
        template: 模板序列
        template_type: GFP 类型
        templates: 所有 GFP 模板
        xgb_predictor: XGBoost 预测器
        brightness_df: 亮度实验数据
        exclusion_set: 排除列表 (set)
        n_rounds: 迭代轮数 (3-5)
        explore_positions: 限定搜索位置 (None=所有突变位置)
        protect_positions: 保护位置 (不突变的位点)

    Returns:
        优化后的序列评估结果列表 (按 composite 降序)
    """
    print(f"\n  [DeepSearch] Local Hill-Climbing: {len(seed_sequences)} seeds × {n_rounds} rounds")

    # Determine mutable positions from seed sequences (positions that differ from template)
    if explore_positions is None:
        mutable_positions = set()
        for seq in seed_sequences:
            for i, (s_aa, t_aa) in enumerate(zip(seq, template)):
                if s_aa != t_aa:
                    mutable_positions.add(i + 1)  # 1-indexed
        mutable_positions = sorted(mutable_positions)[:30]  # Cap at 30 hot positions
    else:
        mutable_positions = explore_positions

    # Remove protected positions
    if protect_positions:
        mutable_positions = [p for p in mutable_positions if p not in protect_positions]

    print(f"  [DeepSearch] Mutable positions: {len(mutable_positions)}")
    print(f"  [DeepSearch] Protected: chromophore region ({sorted(protect_positions or [])})")

    current_best = None
    best_score = -999.0
    all_improvements = []
    n_evaluated = 0

    for seq in seed_sequences:
        # Evaluate baseline
        baseline = evaluate_sequence(
            seq, template, template_type, templates,
            xgb_predictor, brightness_df, exclusion_set,
            esm_model=esm_model, esm_alphabet=esm_alphabet, device=device,
            brightness_floor=brightness_floor,
            distilled_rules=distilled_rules, gain_g_min=gain_g_min, gain_g_max=gain_g_max,
        )
        if not baseline['valid']:
            continue

        current_seq = seq
        current_eval = baseline

        for round_idx in range(n_rounds):
            improved = False
            round_best_score = current_eval['composite']

            for pos in mutable_positions:
                idx = pos - 1
                if idx >= len(current_seq):
                    continue
                original_aa = current_seq[idx]

                for new_aa in STANDARD_AA - {original_aa}:
                    # Mutate
                    seq_list = list(current_seq)
                    seq_list[idx] = new_aa
                    candidate = ''.join(seq_list)

                    n_evaluated += 1
                    result = evaluate_sequence(
                        candidate, template, template_type, templates,
                        xgb_predictor, brightness_df, exclusion_set,
                        esm_model=esm_model, esm_alphabet=esm_alphabet, device=device,
                        distilled_rules=distilled_rules, gain_g_min=gain_g_min, gain_g_max=gain_g_max,
                    )

                    if result['valid'] and result['composite'] > round_best_score:
                        round_best_score = result['composite']
                        current_seq = candidate
                        current_eval = result
                        improved = True

                        all_improvements.append({
                            'round': round_idx + 1,
                            'position': pos,
                            'from_aa': original_aa,
                            'to_aa': new_aa,
                            'score_delta': round(result['composite'] - baseline['composite'], 4),
                            'new_composite': result['composite'],
                        })

                        # Greedy: accept first improvement at this position,
                        # move to next position
                        break

            if not improved:
                # Converged for this seed
                break

        if current_eval['composite'] > best_score:
            best_score = current_eval['composite']
            current_best = current_eval

    # Sort improvements for reporting
    all_improvements.sort(key=lambda x: -x['score_delta'])

    print(f"  [DeepSearch] Evaluated {n_evaluated} variants")
    print(f"  [DeepSearch] Found {len(all_improvements)} improvements")
    if all_improvements:
        top_imp = all_improvements[:5]
        for imp in top_imp:
            print(f"    Pos{imp['position']}: {imp['from_aa']}→{imp['to_aa']} "
                  f"(Δ={imp['score_delta']:+.4f}, round={imp['round']})")

    return [current_best] if current_best else []


# ============================================================
# 2b. Genetic Algorithm (v4.0: 替代局部爬山, 全局搜索)
# ============================================================

def genetic_algorithm(
    seed_sequences: List[str],
    template: str,
    template_type: str,
    templates: Dict[str, str],
    xgb_predictor: Dict,
    brightness_df: pd.DataFrame,
    exclusion_set: Set[str],
    n_generations: int = 50,
    population_size: int = 50,
    mutation_rate: float = 0.3,
    elite_size: int = 5,
    protect_positions: Optional[Set[int]] = None,
    esm_model=None,
    esm_alphabet=None,
    device: str = "cpu",
    brightness_floor: float = 0.0,
    distilled_rules: Dict = None,
    gain_g_min: float = -10.0,
    gain_g_max: float = 5.0,
) -> List[Dict]:
    """
    v4.4: 遗传算法 + 局部爬山 (Memetic Algorithm), gain matrix 评分。

    GA 负责全局探索 (crossover 组合不同父本的有利突变),
    局部爬山负责局部精炼 (对每个子代做 1-pass 单点扫描)。

    Args:
        seed_sequences: 初始种群
        n_generations: 迭代代数 (默认 30)
        population_size: 种群大小 (默认 50)
        mutation_rate: 变异概率
        elite_size: 精英保留数
        distilled_rules: 数据蒸馏规则 (含 gain_matrix, 用于亮度评分)
        gain_g_min/max: 全局增益范围 (用于 B_pct 归一化, 来自主管线 2000-variant pool)
    """
    print(f"\n  [GA+HC] Memetic Algorithm: pop={population_size}, gen={n_generations}, "
          f"mut_rate={mutation_rate}")

    # Build mutation position pool
    mutable_positions = set()
    for seq in seed_sequences[:30]:
        for i, (s_aa, t_aa) in enumerate(zip(seq, template)):
            if s_aa != t_aa:
                mutable_positions.add(i + 1)
    if protect_positions:
        mutable_positions -= protect_positions
    mutable_positions = sorted(mutable_positions)[:40]
    print(f"  [GA+HC] Mutable positions: {len(mutable_positions)}")

    # ── Evaluation helper (v4.4: gain matrix scoring) ──
    def eval_seq(seq):
        return evaluate_sequence(
            seq, template, template_type, templates,
            xgb_predictor, brightness_df, exclusion_set,
            esm_model=esm_model, esm_alphabet=esm_alphabet, device=device,
            brightness_floor=brightness_floor,
            distilled_rules=distilled_rules,
            gain_g_min=gain_g_min, gain_g_max=gain_g_max,
        )

    # ── Local hill-climbing (1-pass, 仅扫描该个体的突变位点) ──
    def local_refine(seq):
        """对单条序列做 1-pass 爬山: 在每个突变位点尝试替代 AA, 接受任何提升。"""
        current = seq
        current_score = fitness.get(current, -999)
        if current_score < -900:
            # Not yet evaluated
            r = eval_seq(current)
            current_score = r['composite'] if r['valid'] else -999
            fitness[current] = current_score

        # Find positions where this seq differs from template
        diff_positions = [i + 1 for i in range(len(current))
                          if current[i] != template[i] and (i + 1) in set(mutable_positions)]

        improved = False
        for pos in diff_positions:
            idx = pos - 1
            original_aa = current[idx]
            for new_aa in STANDARD_AA - {original_aa}:
                seq_list = list(current)
                seq_list[idx] = new_aa
                candidate = ''.join(seq_list)
                if candidate in fitness:
                    continue
                r = eval_seq(candidate)
                fitness[candidate] = r['composite'] if r['valid'] else -999
                if r['valid'] and r['composite'] > current_score:
                    current = candidate
                    current_score = r['composite']
                    improved = True
                    break  # First improvement accepted, move to next position
        return current, improved

    # ── Initialize population ──
    population = list(seed_sequences[:population_size])
    while len(population) < population_size:
        base = random.choice(seed_sequences[:min(30, len(seed_sequences))])
        seq_list = list(base)
        for _ in range(random.randint(1, 2)):
            pos = random.choice(mutable_positions)
            if pos - 1 < len(seq_list):
                seq_list[pos - 1] = random.choice(list(STANDARD_AA - {seq_list[pos - 1]}))
        population.append(''.join(seq_list))

    fitness = {}
    for seq in population:
        r = eval_seq(seq)
        fitness[seq] = r['composite'] if r['valid'] else -999

    best_overall = None
    best_score = -999
    n_evaluated = len(population)
    n_refined = 0

    for gen in range(n_generations):
        ranked = sorted(population, key=lambda s: fitness.get(s, -999), reverse=True)
        gen_best = fitness.get(ranked[0], -999)

        if gen_best > best_score:
            best_score = gen_best
            best_overall = eval_seq(ranked[0])

        # Elite
        new_population = ranked[:elite_size]

        # Breed
        while len(new_population) < population_size:
            p1 = max(random.sample(ranked[:min(30, len(ranked))], min(3, len(ranked))),
                     key=lambda s: fitness.get(s, -999))
            p2 = max(random.sample(ranked[:min(30, len(ranked))], min(3, len(ranked))),
                     key=lambda s: fitness.get(s, -999))

            # Crossover
            if len(p1) == len(p2) and random.random() < 0.7:
                cross_point = random.randint(1, len(p1) - 2)
                child = p1[:cross_point] + p2[cross_point:]
            else:
                child = p1 if fitness.get(p1, -999) >= fitness.get(p2, -999) else p2

            # Mutation
            if random.random() < mutation_rate:
                seq_list = list(child)
                for _ in range(random.randint(1, 2)):
                    pos = random.choice(mutable_positions)
                    if pos - 1 < len(seq_list):
                        seq_list[pos - 1] = random.choice(
                            list(STANDARD_AA - {seq_list[pos - 1]}))
                child = ''.join(seq_list)

            # ── v4.0: 局部爬山精炼 ──
            # 50% 概率对新子代执行 1-pass 爬山
            if random.random() < 0.5:
                refined, improved = local_refine(child)
                if improved:
                    n_refined += 1
                    child = refined

            new_population.append(child)

        # Evaluate unevaluated
        for seq in new_population[elite_size:]:
            if seq not in fitness:
                r = eval_seq(seq)
                fitness[seq] = r['composite'] if r['valid'] else -999
                n_evaluated += 1

        population = new_population

        if gen % 5 == 0 or gen == n_generations - 1:
            valid_scores = [fitness[s] for s in ranked if fitness.get(s, -999) > -900]
            if valid_scores:
                print(f"  [GA+HC] Gen {gen:2d}: best={max(valid_scores):.4f}, "
                      f"mean={sum(valid_scores)/len(valid_scores):.4f}, "
                      f"valid={len(valid_scores)}/{len(population)}, refined={n_refined}")

    # Return best unique sequences
    seen = set()
    results = []
    ranked_final = sorted(population, key=lambda s: fitness.get(s, -999), reverse=True)
    for seq in ranked_final:
        if seq not in seen and fitness.get(seq, -999) > -900:
            seen.add(seq)
            r = eval_seq(seq)
            if r['valid']:
                results.append(r)
        if len(results) >= population_size:
            break

    print(f"  [GA] Evaluated {n_evaluated} variants, "
          f"returning top {len(results)} unique")
    return results


# ============================================================
# 3. Deep Search Pipeline (Golden Pairs + GA)
# ============================================================

def deep_search_pipeline(
    top_candidates: List[Dict],
    template: str,
    template_type: str,
    templates: Dict[str, str],
    xgb_predictor: Dict,
    brightness_df: pd.DataFrame,
    exclusion_list: List[str],
    n_rounds: int = 4,
    n_seeds: int = 50,
    esm_model=None,
    esm_alphabet=None,
    device: str = "cpu",
    brightness_floor: float = 0.0,
    distilled_rules: Dict = None,
    gain_g_min: float = -10.0,
    gain_g_max: float = 5.0,
) -> List[Dict]:
    """
    v4.4 深度搜索管线 (gain matrix 评分, 与主管线一致)。

    流程:
      1. 挖掘 Golden Pairs (正上位网络)
      2. 将 Golden Pairs 应用到 Top 种子序列 → 生成增强种子
      3. GA+HC 全局搜索 (从增强种子出发, gain matrix 评分)
      4. 去重 + 排除列表过滤
      5. 返回优化后的候选池

    Args:
        top_candidates: v2.2 管线产出的 Top-N 候选 (含 sequence, scores)
        template: 模板序列
        template_type: GFP 类型
        templates: 所有 GFP 模板
        xgb_predictor: XGBoost 预测器
        brightness_df: 亮度实验数据
        exclusion_list: 排除列表
        n_rounds: 爬山迭代轮数
        n_seeds: 使用的种子序列数

    Returns:
        深度优化后的候选序列评估列表
    """
    exclusion_set = set(exclusion_list)
    protect = CHROMOPHORE_REGION_1IDX.copy()
    protect.add(1)  # Protect M start

    # Step 1: Mine Golden Pairs
    print(f"\n{'='*60}")
    print(f"  v3.0 Deep Search Pipeline")
    print(f"{'='*60}")

    print(f"\n  [Step 1/4] Mining Golden Pairs from 141k data...")
    golden_pairs = mine_golden_pairs(
        brightness_df, template_type,
        min_brightness=3.5, min_cooccurrence=3, top_k=100
    )
    print(f"  Found {len(golden_pairs)} Golden Pairs")
    if golden_pairs:
        top5 = golden_pairs[:5]
        for gp in top5:
            print(f"    {gp['aa_i']}{gp['pos_i']}+{gp['aa_j']}{gp['pos_j']}: "
                  f"brightness={gp['mean_brightness']:.2f}, n={gp['cooccurrence']}")

    # Step 2: Create enhanced seeds (Golden Pairs + original tops)
    print(f"\n  [Step 2/4] Creating enhanced seed sequences...")
    seed_sequences = []
    for item in top_candidates[:n_seeds]:
        seed_sequences.append(item['sequence'])

    # Apply Golden Pairs to create additional seeds
    if golden_pairs:
        for item in top_candidates[:min(20, len(top_candidates))]:
            for n_pairs in [1, 2]:
                gp_seq = apply_golden_pairs(template, golden_pairs, n_pairs=n_pairs)
                # Combine with candidate mutations
                combined = list(gp_seq)
                candidate_seq = item['sequence']
                for i, (gp_aa, cand_aa) in enumerate(zip(gp_seq, candidate_seq)):
                    if cand_aa != template[i] and i < len(combined):
                        combined[i] = cand_aa
                combined_seq = ''.join(combined)
                if len(combined_seq) >= 220:
                    seed_sequences.append(combined_seq)

    # Deduplicate seeds
    seed_sequences = list(dict.fromkeys(seed_sequences))  # preserve order
    print(f"  {len(seed_sequences)} unique seeds (from {len(top_candidates[:n_seeds])} candidates + Golden Pairs)")

    # Step 3: Genetic Algorithm (v4.0: 替代爬山, 全局搜索)
    print(f"\n  [Step 3/4] Genetic Algorithm ({n_rounds} → {n_rounds * 5} generations)...")
    optimized = genetic_algorithm(
        seed_sequences=seed_sequences[:n_seeds],
        template=template,
        template_type=template_type,
        templates=templates,
        xgb_predictor=xgb_predictor,
        brightness_df=brightness_df,
        exclusion_set=exclusion_set,
        n_generations=max(50, n_rounds * 5),
        population_size=min(300, len(seed_sequences)),
        mutation_rate=0.3,
        elite_size=5,
        protect_positions=protect,
        esm_model=esm_model,
        esm_alphabet=esm_alphabet,
        device=device,
        brightness_floor=brightness_floor,
        distilled_rules=distilled_rules,
        gain_g_min=gain_g_min,
        gain_g_max=gain_g_max,
    )

    # Step 4: Merge with original top candidates, deduplicate, sort
    print(f"\n  [Step 4/4] Merging and ranking...")
    all_candidates = {}

    # Add optimized sequences
    for item in optimized:
        if item['valid']:
            all_candidates[item['sequence']] = item

    # Add original top candidates (keep their scores if not improved)
    for item in top_candidates:
        seq = item['sequence']
        if seq not in all_candidates:
            all_candidates[seq] = {
                'sequence': seq,
                'brightness': item.get('predicted_brightness', 0),
                'stability': item.get('predicted_stability', 0.5),
                'composite': item.get('composite_score', 0),
                'folding_score': item.get('folding_score', 0.5),
                'knowledge_score': item.get('knowledge_score', 0.5),
                'aggregation_risk': 0.0,
                'valid': True,
            }

    # Sort by composite score
    ranked = sorted(all_candidates.values(), key=lambda x: -x.get('composite', 0))

    # Filter by exclusion list
    ranked = [r for r in ranked if r['sequence'] not in exclusion_set]

    print(f"  Final pool: {len(ranked)} candidates")
    if ranked:
        print(f"  Top composite score: {ranked[0]['composite']:.4f}")
        print(f"  Best brightness: {ranked[0]['brightness']:.4f}")

    return ranked


# ============================================================
# 4. Aggregation Risk (Task 2a)
# ============================================================

def compute_aggregation_risk(seq: str) -> float:
    """
    v3.0: 基于氨基酸组成的聚集倾向评估。

    模拟 Aggrescan/TANGO 的核心思想:
      1. 疏水斑块: 连续 5+ 个疏水残基 → 高聚集风险
      2. β-折叠倾向: V, I, L, F, Y, W 富集区域 → 淀粉样聚集风险
      3. 低净电荷 + 高疏水 → 疏水聚集风险
      4. 芳香族堆积: 连续 F/W/Y → π-π 聚集

    Returns:
        聚集风险评分 [0, 1], 0=安全, 1=极高风险
    """
    n = len(seq)
    if n == 0:
        return 0.5

    hydrophobic = set('AILMFVWY')
    beta_prone = set('VILFYW')
    aromatic = set('FWY')
    charged = set('RKDE')

    # 1. Hydrophobic patch detection
    max_hydro_run = 0
    current_run = 0
    for aa in seq:
        if aa in hydrophobic:
            current_run += 1
            max_hydro_run = max(max_hydro_run, current_run)
        else:
            current_run = 0
    patch_risk = min(1.0, max(0.0, (max_hydro_run - 4) / 6.0))  # >4 → risk starts, >10 → max risk

    # 2. Beta-sheet aggregation propensity (Chiti-Dobson scale, simplified)
    beta_scores = {'V': 0.8, 'I': 0.9, 'L': 0.7, 'F': 1.0, 'Y': 0.9, 'W': 1.0,
                   'A': 0.3, 'M': 0.4, 'C': 0.2, 'T': 0.3}
    beta_sum = sum(beta_scores.get(aa, 0.1) for aa in seq)
    beta_risk = min(1.0, beta_sum / (n * 0.15))  # normalize

    # 3. Charge-hydrophobicity balance
    n_charged = sum(1 for aa in seq if aa in charged)
    n_hydro = sum(1 for aa in seq if aa in hydrophobic)
    charge_frac = n_charged / n
    hydro_frac = n_hydro / n
    # High hydro + low charge → aggregation-prone
    balance_risk = max(0.0, (hydro_frac - 0.35) * 2.0 - charge_frac * 1.5)
    balance_risk = min(1.0, max(0.0, balance_risk))

    # 4. Aromatic stacking risk
    max_aro_run = 0
    current_aro = 0
    for aa in seq:
        if aa in aromatic:
            current_aro += 1
            max_aro_run = max(max_aro_run, current_aro)
        else:
            current_aro = 0
    aro_risk = min(1.0, max(0.0, (max_aro_run - 2) / 4.0))  # >2 adjacent aromatics → risk

    # Composite aggregation risk
    risk = 0.30 * patch_risk + 0.25 * beta_risk + 0.25 * balance_risk + 0.20 * aro_risk
    return round(min(1.0, max(0.0, risk)), 4)
