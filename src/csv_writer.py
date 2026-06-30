# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cesium
#
"""
csv_writer.py — 输出格式化模块

按竞赛规范生成提交用的 CSV 文件。
"""
import csv
from pathlib import Path
from typing import List, Dict


def write_submission(
    team_name: str,
    sequences: List[Dict],
    output_path: str,
) -> str:
    """
    按竞赛要求格式写出 submission CSV。

    规范:
      - 三列: Team_Name, Seq_ID, Sequence
      - Seq_ID 为 1, 2, 3, ... 编号
      - 每行一条氨基酸序列

    Args:
        team_name: 队伍名称
        sequences: 入选序列列表 (每个 dict 需含 'sequence' 键)
        output_path: 输出 CSV 文件路径

    Returns:
        输出文件的绝对路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['Team_Name', 'Seq_ID', 'Sequence'])

        # Data rows
        for i, item in enumerate(sequences, 1):
            seq = item['sequence']
            writer.writerow([team_name, str(i), seq])

    return str(output_path.absolute())


def write_submission_with_scores(
    team_name: str,
    sequences: List[Dict],
    output_path: str,
) -> str:
    """
    写出包含评分信息的扩展 CSV（用于内部记录，不用于提交）。

    额外列: Length, Predicted_Brightness, Predicted_Stability, Composite_Score
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Team_Name', 'Seq_ID', 'Sequence',
            'Length', 'Predicted_Brightness', 'Predicted_Stability',
            'Composite_Score',
        ])
        for i, item in enumerate(sequences, 1):
            seq = item['sequence']
            writer.writerow([
                team_name, str(i), seq,
                len(seq),
                item.get('predicted_brightness', ''),
                item.get('predicted_stability', ''),
                item.get('composite_score', ''),
            ])

    return str(output_path.absolute())
