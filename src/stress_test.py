# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cesium
#
"""
stress_test.py — 最终 Top-6 压力测试与可视化

1. 热稳定性散点图: X=预测亮度, Y=综合稳定性得分
2. 排除列表 MD5 终验: 确保零命中
3. Agent 深搜叙事: 记录深搜发现 + 拦截日志
"""
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Set

import numpy as np


# ============================================================
# 1. Stability-Brightness Scatter Plot
# ============================================================

def generate_stress_test_plot(
    candidates: List[Dict],
    top6: List[Dict],
    output_dir: str = "output",
) -> str:
    """
    生成热稳定性-亮度散点图。

    X轴: 预测亮度
    Y轴: 综合稳定性得分 (ESM + TGP + 抗聚集)
    标注: Top-6 高亮显示，其余灰色背景
    象限线: 亮度=0.6, 稳定性=0.5 作为"安全区"边界
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [StressTest] matplotlib not available, skipping plot.")
        return ""

    # Extract data
    all_brightness = [c.get('predicted_brightness', c.get('brightness', 0)) for c in candidates]
    all_stability = [c.get('predicted_stability', c.get('stability', 0.5)) for c in candidates]
    all_agg = [c.get('aggregation_risk', 0) for c in candidates]

    # Composite stability = stability * (1 - aggregation_risk)
    all_comp_stability = [s * (1 - a) for s, a in zip(all_stability, all_agg)]

    top6_brightness = [c.get('predicted_brightness', c.get('brightness', 0)) for c in top6]
    top6_stability = [c.get('predicted_stability', c.get('stability', 0.5)) for c in top6]
    top6_agg = [c.get('aggregation_risk', 0) for c in top6]
    top6_comp_stability = [s * (1 - a) for s, a in zip(top6_stability, top6_agg)]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Background: all candidates
    ax.scatter(all_brightness, all_comp_stability, c='lightgray', alpha=0.4,
               s=30, label=f'All candidates ({len(candidates)})')

    # Highlight: Top-6
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']
    for i in range(len(top6)):
        ax.scatter(top6_brightness[i], top6_comp_stability[i],
                   c=colors[i % 6], s=200, edgecolors='black', linewidth=1.5,
                   zorder=5, label=f'Top-{i+1}')

    # Quadrant lines
    ax.axvline(x=0.6, color='gray', linestyle='--', alpha=0.5, label='Brightness=0.6')
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Stability=0.5')

    # Quadrant labels
    ax.text(0.85, 0.85, 'HIGH BRIGHTNESS\nHIGH STABILITY\n(TARGET ZONE)',
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    ax.text(0.20, 0.20, 'LOW BRIGHTNESS\nLOW STABILITY\n(AVOID)',
            transform=ax.transAxes, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))

    ax.set_xlabel('Predicted Brightness (relative to WT)', fontsize=12)
    ax.set_ylabel('Composite Stability (ESM + TGP + Anti-Aggregation)', fontsize=12)
    ax.set_title('GFP Variant Stress Test: Stability vs Brightness', fontsize=14)
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.2)

    # Top-6 annotation
    for i in range(len(top6)):
        ax.annotate(f'T{i+1}',
                    (top6_brightness[i], top6_comp_stability[i]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, fontweight='bold')

    fig.tight_layout()
    output_path = str(Path(output_dir) / "stress_test_scatter.png")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  [StressTest] Scatter plot saved: {output_path}")
    return output_path


# ============================================================
# 2. Exclusion List MD5 Final Verification
# ============================================================

def md5_final_verification(
    sequences: List[str],
    exclusion_list: List[str],
) -> Dict:
    """
    v3.0: 排除列表 MD5 终验。

    对每条提交序列做 MD5 哈希，与排除列表所有序列的 MD5 比对。
    双重保险: 字符串精确比对 + MD5 碰撞检测。

    Returns:
        {
            'all_clear': bool,
            'string_matches': [(seq_idx, matched_seq), ...],
            'md5_collisions': [(seq_idx, exclusion_idx), ...],
            'n_checked': int,
        }
    """
    string_matches = []
    md5_collisions = []

    # Build MD5 set for exclusion list (fast lookup)
    exclusion_md5 = {}
    for i, excl_seq in enumerate(exclusion_list):
        h = hashlib.md5(excl_seq.encode()).hexdigest()
        exclusion_md5[h] = (i, excl_seq)

    for seq_idx, seq in enumerate(sequences):
        # 1. String exact match (O(n) but n=135k, should be fast with set)
        # Already done in filter_selector; this is the final double-check

        # 2. MD5 check
        h = hashlib.md5(seq.encode()).hexdigest()
        if h in exclusion_md5:
            excl_idx, excl_seq = exclusion_md5[h]
            md5_collisions.append((seq_idx, excl_idx))
            # Verify it's not just a hash collision
            if seq == excl_seq:
                string_matches.append((seq_idx, excl_seq))

    return {
        'all_clear': len(string_matches) == 0 and len(md5_collisions) == 0,
        'string_matches': string_matches,
        'md5_collisions': md5_collisions,
        'n_checked': len(sequences),
    }


def generate_stress_report(
    top6: List[Dict],
    candidates: List[Dict],
    md5_result: Dict,
    deep_search_improvements: List[Dict],
    output_dir: str = "output",
) -> str:
    """
    生成完整的压力测试报告 (文本 + 关键指标)。
    """
    from ._version import __version__
    lines = [
        "=" * 65,
        f"  GFP Designer v{__version__} — Final Stress Test Report",
        "=" * 65,
        "",
        "---",
        "1. Brightness-Stability Analysis",
        "---",
    ]

    if top6:
        all_b = [c.get('predicted_brightness', c.get('brightness', 0)) for c in top6]
        all_s = [c.get('predicted_stability', c.get('stability', 0.5)) for c in top6]
        all_agg = [c.get('aggregation_risk', 0) for c in top6]

        lines.append(f"  Top-6 brightness range: [{min(all_b):.4f}, {max(all_b):.4f}]")
        lines.append(f"  Top-6 stability range:  [{min(all_s):.4f}, {max(all_s):.4f}]")
        lines.append(f"  Top-6 aggregation risk: [{min(all_agg):.4f}, {max(all_agg):.4f}]")

        n_high_quadrant = sum(1 for b, s, a in zip(all_b, all_s, all_agg)
                              if b > 0.6 and s * (1 - a) > 0.5)
        lines.append(f"  Sequences in TARGET ZONE (B>0.6, S>0.5): {n_high_quadrant}/{len(top6)}")
        lines.append("")

    lines.extend([
        "---",
        "2. Exclusion List MD5 Verification",
        "---",
        f"  Sequences checked: {md5_result['n_checked']}",
        f"  String matches:   {len(md5_result['string_matches'])}",
        f"  MD5 collisions:   {len(md5_result['md5_collisions'])}",
        f"  Status: {'ALL CLEAR' if md5_result['all_clear'] else 'FAILED — DO NOT SUBMIT'}",
        "",
        "---",
        "3. Deep Search Improvements",
        "---",
        f"  Total improvements found: {len(deep_search_improvements)}",
    ])

    if deep_search_improvements:
        for i, imp in enumerate(deep_search_improvements[:10]):
            lines.append(f"  {i+1}. Pos{imp['position']}: {imp['from_aa']}→{imp['to_aa']} "
                         f"(Δ={imp['score_delta']:+.4f}, round={imp['round']})")

    lines.extend([
        "",
        "---",
        "4. Aggregation Risk Assessment",
        "---",
    ])

    if top6:
        high_agg = [(i+1, c) for i, c in enumerate(top6)
                    if c.get('aggregation_risk', 0) > 0.4]
        if high_agg:
            lines.append("  WARNING: High aggregation risk detected:")
            for rank, c in high_agg:
                lines.append(f"    Rank {rank}: agg_risk={c.get('aggregation_risk', 0):.4f}")
        else:
            lines.append("  All Top-6 sequences: aggregation risk < 0.4 (safe)")

    lines.extend([
        "",
        "=" * 65,
    ])

    report = '\n'.join(lines)
    output_path = str(Path(output_dir) / "stress_test_report.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"  [StressTest] Report saved: {output_path}")
    return output_path
