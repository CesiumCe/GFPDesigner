# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cesium
#
"""
property_predictor.py — 性质预测模块

预测 GFP 变体的亮度值。

主要策略 (ESM + Random Forest):
  1. 从官方 GFP_data.xlsx 加载 141k 条实验亮度数据
  2. 使用 ESM 模型将序列转换为嵌入向量
  3. 训练随机森林回归器预测亮度
  4. 对新序列进行预测

回退策略 (启发式):
  当 ESM 模型不可用时，使用基于理化特征的启发式评分。
"""
import math
import random
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from .utils import (
    compute_sequence_features,
    reconstruct_sequence_from_mutations,
    get_esm_embeddings_fast,
)

warnings.filterwarnings('ignore')


# ============================================================
# ESM + Random Forest Pipeline
# ============================================================

def train_brightness_predictor(
    brightness_df: pd.DataFrame,
    template_seq: str,
    template_type: str,
    model_name: str = "esm2_t6_8M_UR50D",
    max_train_samples: int = 3000,
    n_estimators: int = 100,
    random_state: int = 42,
) -> dict:
    """
    使用 ESM 嵌入 + 随机森林训练亮度预测器。

    流程:
      1. 从 brightness_df 中提取指定 GFP 类型的突变数据
      2. 重构完整突变序列
      3. 用 ESM 计算序列嵌入
      4. 训练 RandomForestRegressor

    Args:
        brightness_df: 亮度数据
        template_seq: 模板序列（用于重构突变序列）
        template_type: GFP 类型名 (如 'avGFP')
        model_name: ESM 模型名
        max_train_samples: 最大训练样本数 (ESM 嵌入较慢)
        n_estimators: 随机森林树的数量
        random_state: 随机种子

    Returns:
        {
            'rf_model': 训练好的 RandomForestRegressor,
            'esm_model_name': 使用的 ESM 模型名,
            'training_score': R² 得分,
            'n_train': 训练样本数,
            'X_val': 验证集特征,
            'y_val': 验证集标签,
            'val_score': 验证集 R² 得分,
        }
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

    print(f"\n  Loading brightness data for {template_type}...")
    df = brightness_df[brightness_df['GFP type'] == template_type].copy()

    if len(df) > max_train_samples:
        # Stratified sampling: ensure coverage of brightness range
        df['brightness_bin'] = pd.cut(df['Brightness'], bins=20, labels=False)
        df = df.groupby('brightness_bin', group_keys=False).apply(
            lambda x: x.sample(
                n=max(1, int(max_train_samples * len(x) / len(df))),
                random_state=random_state,
            )
        ).reset_index(drop=True)
        df = df.sample(n=min(max_train_samples, len(df)), random_state=random_state)

    print(f"  Using {len(df)} samples for training")

    # Reconstruct full sequences from mutations
    print(f"  Reconstructing mutant sequences from {template_type} template...")
    sequences = []
    brightnesses = []
    for _, row in df.iterrows():
        try:
            seq = reconstruct_sequence_from_mutations(template_seq, row['aaMutations'])
            sequences.append(seq)
            brightnesses.append(row['Brightness'])
        except Exception:
            continue

    print(f"  Successfully reconstructed {len(sequences)} sequences")

    # Compute ESM embeddings
    print(f"  Computing ESM embeddings (model: {model_name})...")
    print(f"  This may take a few minutes...")
    try:
        X = get_esm_embeddings_fast(sequences, model_name=model_name)
    except Exception as e:
        print(f"  [WARN] ESM embedding failed: {e}")
        print(f"  Falling back to physicochemical features...")
        X = np.array([list(compute_sequence_features(s).values()) for s in sequences])
        model_name = "physicochemical_fallback"

    y = np.array(brightnesses)

    # Train/test split
    if len(X) > 10:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=random_state
        )
    else:
        X_train, y_train = X, y
        X_val, y_val = None, None

    # Train Random Forest
    print(f"  Training RandomForestRegressor (n_estimators={n_estimators})...")
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(X_train, y_train)

    train_score = rf.score(X_train, y_train)
    val_score = rf.score(X_val, y_val) if X_val is not None and len(X_val) > 0 else None

    print(f"  Training R^2: {train_score:.4f}")
    if val_score is not None:
        print(f"  Validation R^2: {val_score:.4f}")

    return {
        'rf_model': rf,
        'esm_model_name': model_name,
        'training_score': train_score,
        'val_score': val_score,
        'n_train': len(X_train),
        'X_val_shape': X.shape[1],
    }


def predict_brightness_batch(
    sequences: List[str],
    predictor: dict,
    batch_size: int = 16,
) -> np.ndarray:
    """
    使用训练好的预测器批量预测亮度。

    Args:
        sequences: 待预测序列列表
        predictor: train_brightness_predictor 返回的字典
        batch_size: ESM 嵌入批大小

    Returns:
        predicted_brightness 数组
    """
    model_name = predictor['esm_model_name']
    rf_model = predictor['rf_model']

    if model_name == "physicochemical_fallback":
        X = np.array([list(compute_sequence_features(s).values()) for s in sequences])
    else:
        print(f"  Computing ESM embeddings for {len(sequences)} candidate sequences...")
        X = get_esm_embeddings_fast(sequences, model_name=model_name, batch_size=batch_size)

    predictions = rf_model.predict(X)
    return predictions


# ============================================================
# Heuristic Predictor (Fallback)
# ============================================================

def _sequence_similarity(seq: str, template: str) -> float:
    """计算与模板的序列一致性。"""
    if not seq or not template:
        return 0.0
    min_len = min(len(seq), len(template))
    matches = sum(1 for i in range(min_len) if seq[i] == template[i])
    return matches / min_len


def _estimate_folding_efficiency(features: dict) -> float:
    """基于理化特征粗略估算折叠效率。"""
    hydro = features['hydrophobicity_mean']
    aromatic = features['aromatic_fraction']
    charge = abs(features['charge_pH7'])
    gly = features['glycine_fraction']
    pro = features['proline_fraction']

    hydro_score = math.exp(-((hydro - 0.0) ** 2) / 2.0)
    aromatic_penalty = max(0, aromatic - 0.15) * 3.0
    charge_penalty = max(0, charge - 0.15) * 2.0
    gly_ok = 1.0 - max(0, gly - 0.12) * 4.0
    pro_ok = 1.0 - max(0, pro - 0.08) * 5.0

    score = (
        0.30 * hydro_score - 0.15 * aromatic_penalty
        - 0.20 * charge_penalty + 0.20 * gly_ok + 0.15 * pro_ok
    )
    return 1.0 / (1.0 + math.exp(-5.0 * (score - 0.3)))


def predict_heuristic_batch(
    sequences: List[str],
    template: str,
    brightness_cutoff: float = 0.3,
) -> List[Dict]:
    """
    启发式批量预测（回退方案）。
    """
    results = []
    for seq in sequences:
        features = compute_sequence_features(seq)
        similarity = _sequence_similarity(seq, template)
        base_brightness = _estimate_folding_efficiency(features)
        brightness = base_brightness * (0.4 + 0.6 * similarity)

        if similarity < 0.3:
            brightness *= 0.5

        brightness = max(0.0, min(5.0, brightness))
        disqualified = brightness < brightness_cutoff

        results.append({
            'sequence': seq,
            'predicted_brightness': round(brightness, 4),
            'predicted_stability': 0.5,  # Simplified in fallback
            'composite_score': round(brightness * 0.5, 4),
            'disqualified': disqualified,
            'similarity': round(similarity, 4),
        })
    return results


# ============================================================
# Main prediction interface
# ============================================================

def predict_properties(
    sequences: List[str],
    template: str,
    brightness_cutoff: float = 0.3,
    predictor: Optional[dict] = None,
) -> List[Dict]:
    """
    统一的序列性质预测接口。

    优先使用 ESM + RF 预测器，回退到启发式。

    Args:
        sequences: 待预测序列列表
        template: 参考模板序列
        brightness_cutoff: 亮度淘汰阈值（用于标记淘汰）
        predictor: 可选的训练好的预测器

    Returns:
        [{sequence, predicted_brightness, composite_score, disqualified, ...}, ...]
    """
    if predictor is not None and predictor.get('rf_model') is not None:
        print(f"  Using ESM+RF predictor for {len(sequences)} sequences...")
        brightness_preds = predict_brightness_batch(sequences, predictor)

        results = []
        for seq, brightness in zip(sequences, brightness_preds):
            brightness = float(brightness)
            # Convert from raw brightness to relative brightness score
            # Raw brightness in data: WT avGFP ~3.72
            # Normalize to approximate relative brightness
            relative_brightness = brightness / 3.72
            relative_brightness = max(0.0, min(1.5, relative_brightness))

            # Stability estimated from brightness (higher brightness → better folding → better stability)
            stability = 0.4 + 0.3 * relative_brightness
            stability = max(0.0, min(1.0, stability))

            composite = relative_brightness * stability
            disqualified = relative_brightness < brightness_cutoff

            results.append({
                'sequence': seq,
                'predicted_brightness': round(relative_brightness, 4),
                'predicted_stability': round(stability, 4),
                'composite_score': round(composite, 4),
                'disqualified': disqualified,
                'raw_brightness_prediction': round(brightness, 4),
            })
        return results
    else:
        print(f"  Using heuristic predictor (ESM not available)...")
        return predict_heuristic_batch(sequences, template, brightness_cutoff)
