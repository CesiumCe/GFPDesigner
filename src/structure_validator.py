# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cesium
#
"""
structure_validator.py — ESM-2 结构级验证 (替代 ESMFold)

使用 ESM-2 自带的接触预测头 + 表示分析进行结构质量评估:

1. Contact Confidence (接触置信度):
   ESM-2 预训练时包含接触预测任务。高置信度的接触图
   意味着模型对残基间空间关系有清晰的预测 → 序列很可能折叠良好。

2. pLDDT Proxy (pLDDT 代理):
   ESM-2 的 per-residue embedding confidence 与 AlphaFold pLDDT
   有线性相关性 (Lin et al., 2023)。通过 embedding 范数估算。

3. Chromophore Region Integrity (生色团区域完整性):
   专门检查生色团附近 (64-70) 的结构置信度，
   该区域接触概率暴跌 → 生色团不能正确形成 → 无荧光。

优势 vs ESMFold:
  - 无需 OpenFold (不支持 Windows)
  - GPU 推理 ~1-2s/序列 (vs ESMFold ~60s)
  - 使用已有的 ESM-2 t30_150M 模型
  - 接触概率与折叠可靠性高度相关

Reference:
  Lin et al., "Language models of protein sequences at the scale
  of evolution enable accurate structure prediction", Science 2023.
"""
import math
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch


def predict_contact_confidence(
    seq: str,
    model,
    alphabet,
    device: str = "cuda",
) -> Dict:
    """
    使用 ESM-2 接触预测头评估结构置信度。

    ESM-2 接触概率分布极稀疏 (mean~0.018 for 238aa protein)。
    有效信号在 top-k 高置信度接触对中。

    校准数据 (avGFP WT, 238aa):
      mean=0.018, median=0.006, 99th_pct=0.38
      >0.05: 982 pairs, >0.1: 569 pairs

    Returns:
        {
            'top1pct_mean': 前1%高置信接触的平均概率 (核心信号),
            'top5pct_mean': 前5%高置信接触的平均概率,
            'n_strong_contacts': 接触概率>0.1的残基对数,
            'n_very_strong': 接触概率>0.3的残基对数,
            'long_range_top1pct': 长程(|i-j|>12)前1%平均,
        }
    """
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("seq", seq)])
    tokens = tokens.to(device)

    with torch.no_grad():
        contacts = model.predict_contacts(tokens)
        contact_prob = contacts[0].cpu().numpy()

    L = contact_prob.shape[0]
    if L <= 1:
        return {'top1pct_mean': 0.0, 'top5pct_mean': 0.0,
                'n_strong_contacts': 0, 'n_very_strong': 0,
                'long_range_top1pct': 0.0}

    # Extract non-trivial contacts (|i-j| > 2)
    pairs = []
    lr_pairs = []
    for i in range(L):
        for j in range(i + 3, L):
            p = contact_prob[i, j]
            pairs.append(p)
            if j - i > 12:
                lr_pairs.append(p)

    if not pairs:
        return {'top1pct_mean': 0.0, 'top5pct_mean': 0.0,
                'n_strong_contacts': 0, 'n_very_strong': 0,
                'long_range_top1pct': 0.0}

    pairs = np.array(pairs)
    lr_pairs = np.array(lr_pairs) if lr_pairs else np.array([0.0])

    # Top percentile means (the meaningful structure signal)
    k1 = max(1, int(len(pairs) * 0.01))
    k5 = max(1, int(len(pairs) * 0.05))
    lr_k1 = max(1, int(len(lr_pairs) * 0.01))

    return {
        'top1pct_mean': round(float(np.mean(np.sort(pairs)[-k1:])), 4),
        'top5pct_mean': round(float(np.mean(np.sort(pairs)[-k5:])), 4),
        'n_strong_contacts': int(np.sum(pairs > 0.1)),
        'n_very_strong': int(np.sum(pairs > 0.3)),
        'long_range_top1pct': round(float(np.mean(np.sort(lr_pairs)[-lr_k1:])), 4),
    }


def compute_embedding_confidence(
    seq: str,
    model,
    alphabet,
    device: str = "cuda",
    output_layer: int = -1,
) -> Dict:
    """
    基于 ESM-2 per-residue embedding 估算结构置信度。

    pLDDT ≈ sigmoid(α × ||embedding|| + β)
    其中 embedding 是最后一层的 per-residue representation。

    Returns:
        {
            'mean_confidence': 平均 per-residue 置信度 (pLDDT proxy),
            'min_confidence': 最低 residue 置信度,
            'chromophore_confidence': 生色团区域 (64-70) 平均置信度,
            'confidence_cv': 置信度变异系数 (高CV→结构不均匀),
        }
    """
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("seq", seq)])
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[model.num_layers + output_layer + 1])
        embeddings = results["representations"][model.num_layers + output_layer + 1]
        # embeddings: [1, L, D]
        # Compute per-residue L2 norm as confidence proxy
        norms = torch.norm(embeddings[0, 1:-1], dim=-1)  # exclude BOS/EOS
        norms = norms.cpu().numpy()

    # Normalize to [0, 1] via sigmoid-like scaling
    # Typical ESM-2 norm range: 10-50
    confidence = 1.0 / (1.0 + np.exp(-(norms - 25.0) / 8.0))

    # Chromophore region (positions 64-70, 1-indexed → 0-indexed in norms)
    chromo_start = max(0, 63)
    chromo_end = min(len(confidence), 70)
    chromo_conf = np.mean(confidence[chromo_start:chromo_end]) if chromo_end > chromo_start else 0.5

    return {
        'mean_confidence': round(float(np.mean(confidence)), 4),
        'min_confidence': round(float(np.min(confidence)), 4),
        'chromophore_confidence': round(float(chromo_conf), 4),
        'confidence_cv': round(float(np.std(confidence) / max(np.mean(confidence), 0.01)), 4),
    }


def validate_structure(
    seq: str,
    template: str,
    esm_model,
    esm_alphabet,
    device: str = "cuda",
    wt_contact: Optional[Dict] = None,
    wt_chromo_conf: Optional[float] = None,
) -> Dict:
    """
    v3.1: 综合结构验证 (Contact + Embedding + Chromophore)。

    Chromophore confidence 使用 WT-relative 评分:
      生色团区域 (SYG) 天然是柔性 loop，绝对置信度偏低是正常的。
      只有相对于 WT 的显著下降才意味着结构问题。

    Args:
        wt_contact: WT contact confidence (若提供则做相对比较)
        wt_chromo_conf: WT chromophore confidence baseline

    Returns:
        {
            'structure_score': 综合结构评分 [0,1],
            'contact_confidence': {...},
            'embedding_confidence': {...},
            'flags': ['good'|'warning'|'danger', ...],
            'passed': bool,
        }
    """
    # 1. Contact confidence
    contact = predict_contact_confidence(seq, esm_model, esm_alphabet, device)

    # 2. Embedding confidence
    embed = compute_embedding_confidence(seq, esm_model, esm_alphabet, device)

    # 3. Chromophore region check (WT-relative)
    chromo_conf = embed['chromophore_confidence']
    chromo_drop = 0.0
    if wt_chromo_conf is not None and wt_chromo_conf > 0:
        chromo_drop = max(0.0, (wt_chromo_conf - chromo_conf) / wt_chromo_conf)

    # 4. Score aggregation (recalibrated on WT avGFP baseline)
    # WT avGFP: top1pct≈0.54, n_strong≈569, lr_top1pct≈0.52
    # Well-folded GFP should exceed ~70% of WT values

    # Contact score: how strong are the top contacts relative to WT expectation
    top1 = contact['top1pct_mean']
    n_strong = contact['n_strong_contacts']
    lr_top1 = contact['long_range_top1pct']

    # Normalize: WT ~0.5 → score ~0.7
    contact_score = min(1.0, (top1 / 0.5) * 0.7)
    lr_score = min(1.0, (lr_top1 / 0.5) * 0.7)

    # Embedding score
    embed_score = embed['mean_confidence']

    # Structure score (v3.2): long-range contacts + geometric Contact Order
    # GFP is an 11-strand β-barrel — stability at 72°C depends on cross-barrel contacts
    # ProtSSN (eLife 2025): Contact Order is a key geometric predictor of thermostability
    structure_score = 0.30 * contact_score + 0.30 * lr_score + 0.20 * embed_score + 0.20 * min(1.0, chromo_conf * 1.5)

    # v3.2: Geometric Contact Order bonus (ProtSSN-inspired)
    contact_order_info = compute_contact_order(seq, esm_model, esm_alphabet, device)
    geo_bonus = contact_order_info['geometric_stability_bonus']
    structure_score *= geo_bonus
    structure_score = min(1.0, structure_score)  # Clamp

    # 5. Flagging (calibrated on WT baseline)
    flags = []
    if top1 > 0.40 and lr_top1 > 0.35:
        flags.append('good_contacts')
    elif top1 < 0.20:
        flags.append('danger_low_contacts')

    if n_strong > 400:
        flags.append('good_dense_contacts')
    elif n_strong < 150:
        flags.append('danger_sparse_contacts')

    # v3.1: Chromophore uses WT-relative drop, not absolute threshold
    if chromo_drop < 0.02:
        flags.append('good_chromophore')  # <2% drop from WT → fine
    elif chromo_drop > 0.30:
        flags.append('danger_chromophore')  # >30% drop → real problem
    elif chromo_drop > 0.15:
        flags.append('warning_chromophore')  # 15-30% drop → monitor

    if embed['confidence_cv'] > 0.5:
        flags.append('warning_uneven_structure')

    if top1 < 0.10:
        flags.append('danger_no_structure')

    # Pass/fail
    passed = ('danger_no_structure' not in flags
              and 'danger_chromophore' not in flags
              and 'danger_sparse_contacts' not in flags)

    return {
        'structure_score': round(structure_score, 4),
        'contact_confidence': contact,
        'embedding_confidence': embed,
        'chromophore_confidence': chromo_conf,
        'chromophore_drop': round(chromo_drop, 4),
        'contact_order': contact_order_info,      # v3.2
        'geometric_bonus': geo_bonus,             # v3.2
        'flags': flags,
        'passed': passed,
    }


def validate_batch(
    sequences: List[str],
    template: str,
    esm_model,
    esm_alphabet,
    device: str = "cuda",
) -> List[Dict]:
    """批量结构验证 (v3.1: WT-relative scoring)。"""
    # First, compute WT baseline
    print(f"  [Structure] Computing WT baseline...")
    wt_result = validate_structure(template, template, esm_model, esm_alphabet, device)
    wt_contact = wt_result['contact_confidence']
    wt_chromo = wt_result['chromophore_confidence']

    print(f"  [Structure] WT baseline: struct={wt_result['structure_score']:.4f}, "
          f"contacts top1%={wt_contact['top1pct_mean']:.4f}, chromo={wt_chromo:.4f}")

    # Now validate each variant with WT baseline
    results = []
    for i, seq in enumerate(sequences):
        if i % 10 == 0:
            print(f"  [Structure] Validating {i+1}/{len(sequences)}...")
        results.append(validate_structure(
            seq, template, esm_model, esm_alphabet, device,
            wt_contact=wt_contact, wt_chromo_conf=wt_chromo
        ))
    return results


def compute_contact_order(
    seq: str,
    model,
    alphabet,
    device: str = "cuda",
    top_fraction: float = 0.20,
) -> Dict:
    """
    v3.2: 从 ESM-2 接触概率矩阵计算 Contact Order (几何特征)。

    ProtSSN (eLife 2025) 证明接触顺序等几何特征对热稳定性预测极有效。
    GFP β-桶的稳定性依赖于长程跨桶接触 (β1-β11, β2-β10 等)。

    方法:
      1. 提取 top-20% 最高置信度的接触对
      2. 计算这些接触对的平均序列距离 (|i-j|)
      3. 长程接触越多 → β-桶越稳定 → geometric_stability_bonus

    Returns:
        {
            'contact_order': 平均序列距离 (top-20% contacts),
            'n_top_contacts': 高置信接触对数量,
            'long_range_fraction': 长程 (|i-j|>24) 接触占比,
            'geometric_stability_bonus': 几何稳定性加成 [0.9, 1.1],
        }
    """
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("seq", seq)])
    tokens = tokens.to(device)

    with torch.no_grad():
        contacts = model.predict_contacts(tokens)
        contact_prob = contacts[0].cpu().numpy()

    L = contact_prob.shape[0]
    if L <= 1:
        return {'contact_order': 0, 'n_top_contacts': 0,
                'long_range_fraction': 0, 'geometric_stability_bonus': 1.0}

    # Extract all non-trivial contact pairs (|i-j| > 3)
    pairs = []
    for i in range(L):
        for j in range(i + 4, L):
            pairs.append((i, j, contact_prob[i, j]))

    if not pairs:
        return {'contact_order': 0, 'n_top_contacts': 0,
                'long_range_fraction': 0, 'geometric_stability_bonus': 1.0}

    # Sort by contact probability, take top fraction
    pairs.sort(key=lambda x: -x[2])
    k = max(1, int(len(pairs) * top_fraction))
    top_pairs = pairs[:k]

    # Average sequence separation (Contact Order)
    separations = [abs(j - i) for i, j, _ in top_pairs]
    avg_separation = np.mean(separations)

    # Long-range fraction: |i-j| > 24 (cross-barrel in GFP)
    n_long_range = sum(1 for s in separations if s > 24)
    long_range_frac = n_long_range / len(separations)

    # Geometric stability bonus (calibrated on avGFP WT)
    # ESM-2 contact predictions favor local contacts → avg_separation ~20 for WT 238aa
    # Use WT as baseline (bonus=1.0 at WT level)
    # Higher separation → stronger long-range barrel → bonus > 1.0
    # Lower separation → weakening barrel → bonus < 1.0
    if avg_separation > 28:
        bonus = 1.08  # Exceptional cross-barrel packing
    elif avg_separation > 24:
        bonus = 1.04  # Good barrel integrity
    elif avg_separation > 18:
        bonus = 1.00  # Normal (WT-like)
    elif avg_separation > 14:
        bonus = 0.93  # Weakened barrel
    else:
        bonus = 0.85  # Barrel collapse risk

    return {
        'contact_order': round(float(avg_separation), 1),
        'n_top_contacts': k,
        'long_range_fraction': round(float(long_range_frac), 4),
        'geometric_stability_bonus': round(bonus, 4),
    }


def generate_structure_report(
    validation_results: List[Dict],
    top6_indices: List[int],
) -> str:
    """生成结构验证报告。"""
    lines = [
        "=" * 65,
        "  ESM-2 Structure Validation Report",
        "=" * 65,
        "",
    ]

    all_scores = [r['structure_score'] for r in validation_results]
    lines.append(f"  Total validated: {len(validation_results)}")
    lines.append(f"  Structure score: [{min(all_scores):.3f}, {max(all_scores):.3f}], "
                 f"mean={sum(all_scores)/len(all_scores):.3f}")
    lines.append(f"  Passed: {sum(1 for r in validation_results if r['passed'])}")

    # Top-6 detail
    if top6_indices:
        lines.append("")
        lines.append(f"  Top-6 Structure Validation:")
        lines.append(f"  {'Rank':<6} {'Struct':<8} {'Contact':<10} {'Chromo':<10} {'Flags'}")
        lines.append(f"  {'-'*60}")
        for rank, idx in enumerate(top6_indices, 1):
            if idx < len(validation_results):
                r = validation_results[idx]
                contact_val = r['contact_confidence'].get('top1pct_mean',
                            r['contact_confidence'].get('mean_contact_prob', 0))
                lines.append(
                    f"  {rank:<6} {r['structure_score']:<8.4f} "
                    f"{contact_val:<10.4f} "
                    f"{r['chromophore_confidence']:<10.4f} "
                    f"{','.join(r['flags'][:3])}"
                )

    # Flags summary
    flag_counts = {}
    for r in validation_results:
        for f in r['flags']:
            flag_counts[f] = flag_counts.get(f, 0) + 1
    lines.append("")
    lines.append("  Flag Distribution:")
    for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1]):
        lines.append(f"    {flag}: {count}")

    lines.append("")
    lines.append("=" * 65)
    return '\n'.join(lines)
