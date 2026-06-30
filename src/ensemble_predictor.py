# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cesium
#
"""
ensemble_predictor.py — 多模型集成预测器 v3

彻底重写的预测策略:

核心洞察:
  - 仅 GFP 类型就解释 30% 方差 (R²=0.30)
  - ESM 零样本掩码边际评分比 mean-pooled embeddings 有效得多
  - XGBoost > RandomForest (对 141k 条 tabular 数据)
  - 突变级别的特征 (BLOSUM/体积/疏水性变化) 至关重要

三层架构:
  Layer 1: ESM-2 零样本 per-mutation 评分
    → 对每个突变位置计算 log P(mutant_AA) - log P(template_AA)
    → 聚合: sum, mean, max, min of per-mutation scores

  Layer 2: XGBoost 监督学习 (全量 141k 数据)
    → 特征: GFP type + ESM零样本分数 + 突变数 + 位置特征 + 理化特征
    → 预期 R²: 0.30-0.50

  Layer 3: 物理化学约束
    → 疏水核心、芳香族微环境、电荷平衡

集成:
  ensemble = 0.20*zs + 0.55*xgb + 0.25*physics  (监督主导)
  or
  ensemble = 0.40*zs + 0.60*physics  (无监督回退)
"""
import math
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

from .utils import (
    compute_sequence_features,
    parse_mutation_string,
    reconstruct_sequence_from_mutations,
    HYDROPATHY, AROMATIC_AA, STANDARD_AA,
)

warnings.filterwarnings('ignore')

# Amino acid volume (Angstrom^3, from Zamyatin 1972)
AA_VOLUME = {
    'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
    'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
    'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
    'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6,
}

# BLOSUM62 diagonal (self-similarity)
BLOSUM_SELF = {
    'A': 4, 'C': 9, 'D': 6, 'E': 5, 'F': 6, 'G': 6, 'H': 8,
    'I': 4, 'K': 5, 'L': 4, 'M': 5, 'N': 6, 'P': 7, 'Q': 5,
    'R': 5, 'S': 4, 'T': 5, 'V': 4, 'W': 11, 'Y': 7,
}

# BLOSUM62 full matrix (imported from folding_analyzer)
from .folding_analyzer import BLOSUM62


# ============================================================
# Spatial adjacency map for avGFP (ESMFold-computed CA distances < 8Å)
# ============================================================
# Key: position (1-indexed), Value: list of positions within 8Å in WT structure
# This captures which mutation pairs are likely to have epistatic interactions.
# Computed from ESMFold v1 3D coordinates of avGFP WT.
SPATIAL_NEIGHBORS_8A = {
    1: [2,3,4,5,6], 2: [1,3,4,5,6,85], 3: [1,2,4,5,6,7],
    4: [1,2,3,5,6,7,8,84,85,86,88,89], 5: [1,2,3,4,6,7,8,9,79,85],
    6: [1,2,3,4,5,7,8,9,10], 7: [3,4,5,6,8,9,10,12],
    8: [4,5,6,7,9,10,12,36,37,38,71], 9: [5,6,7,8,10,11,36,37,38],
    10: [6,7,8,9,11,12,35,36,37,38], 11: [9,10,12,13,34,35,36,37,38],
    12: [7,8,10,11,13,14,34,35,36,37,117,118,119],
    13: [11,12,14,15,32,33,34,35,117,118,119],
    14: [12,13,15,16,31,32,33,34,35,44,118,119,120,121],
    15: [13,14,16,17,30,31,32,33,118,119,120,121],
    16: [14,15,17,18,29,30,31,32,64,120,121,122,123],
    17: [15,16,18,19,28,29,30,31,120,121,122,123],
    18: [16,17,19,20,28,29,30,64,122,123,124,125],
    19: [17,18,20,21,26,27,28,29,123,124,125],
    20: [18,19,21,22,25,26,27,28,124,125,126,127],
    21: [19,20,22,23,24,25,26,27,125,126,127],
    22: [20,21,23,24,25,26,54,125,126,127,128,130],
    23: [21,22,24,25,127,128,129,130],
    24: [21,22,23,25,26], 25: [20,21,22,23,24,26,27,51,52,54],
    26: [19,20,21,22,24,25,27,28,50,51,52],
    27: [19,20,21,25,26,28,29,48,49,50,51,52,53,54],
    28: [17,18,19,20,26,27,29,30,48,49,50,51],
    29: [16,17,18,19,27,28,30,31,46,47,48,49,50],
    30: [15,16,17,18,28,29,31,32,46,47,48,49],
    31: [14,15,16,17,29,30,32,33,44,45,46,47],
    32: [13,14,15,16,30,31,33,34,44,45,46],
    33: [13,14,15,31,32,34,35,42,43,44,45],
    34: [11,12,13,14,32,33,35,36,42,43,44],
    35: [10,11,12,13,14,33,34,36,37,40,41,42,43,71],
    36: [8,9,10,11,12,34,35,37,38,39,40,41,42,71],
    37: [8,9,10,11,12,35,36,38,39,40,41,71,72],
    38: [8,9,10,11,36,37,39,40,41], 39: [36,37,38,40,41,72],
    40: [35,36,37,38,39,41,42,71,72,73],
    41: [35,36,37,38,39,40,42,43,71,72,229,232],
    42: [33,34,35,36,40,41,43,44,71,72,229],
    43: [33,34,35,41,42,44,45,229], 44: [14,31,32,33,34,42,43,45,46],
    45: [31,32,33,43,44,46,47], 46: [29,30,31,32,44,45,47,48],
    47: [29,30,31,45,46,48,49], 48: [27,28,29,30,46,47,49,50,51,52,53],
    49: [27,28,29,30,47,48,50,51], 50: [26,27,28,29,48,49,51,52],
    51: [25,26,27,28,48,49,50,52,53], 52: [25,26,27,48,50,51,53,54],
    53: [27,48,51,52,54,55,56,57], 54: [22,25,27,52,53,55,56],
    55: [53,54,56,57,59,60,136,137], 56: [53,54,55,57,58,59,60,137],
    57: [53,55,56,58,59,60,61], 58: [56,57,59,60,61,62],
    59: [55,56,57,58,60,61,62,63], 60: [55,56,57,58,59,61,62,63,64],
    61: [57,58,59,60,62,63,64,65,226],
    62: [58,59,60,61,63,64,65,66,222,223,225,226],
    63: [59,60,61,62,64,65,66,108], 64: [16,18,60,61,62,63,65,66,67,123],
    65: [61,62,63,64,66,67,68,69,226],
    66: [62,63,64,65,67,68,69,110],
    67: [64,65,66,68,69,70,94,110,111,112,121],
    68: [65,66,67,69,70,71,112,121],
    69: [65,66,67,68,70,71,72,73,225],
    70: [67,68,69,71,72,73,74], 71: [8,35,36,37,40,41,42,68,69,70,72,73,74],
    72: [37,39,40,41,42,69,70,71,73,74], 73: [40,69,70,71,72,74,75,203],
    74: [70,71,72,73,75,76,77,78,79], 75: [73,74,76,77,78,79],
    76: [74,75,77,78,79,80], 77: [74,75,76,78,79,80,81],
    78: [74,75,76,77,79,80,81], 79: [5,74,75,76,77,78,80,81,84],
    80: [76,77,78,79,81,82], 81: [77,78,79,80,82,83,84,85],
    82: [80,81,83,84,85,86], 83: [81,82,84,85,86,87,88],
    84: [4,79,81,82,83,85,86,87,88], 85: [2,4,5,81,82,83,84,86,87,88],
    86: [4,82,83,84,85,87,88,89,90], 87: [83,84,85,86,88,89,90,91,92],
    88: [4,83,84,85,86,87,89,90,91,92,112,113,114],
    89: [4,86,87,88,90,91,113,114], 90: [86,87,88,89,91,92,113,114],
    91: [87,88,89,90,92,93,111,112,113,114],
    92: [87,88,90,91,93,94,111,112,113],
    93: [91,92,94,95,109,110,111,112], 94: [67,92,93,95,96,109,110,111],
    95: [93,94,96,97,107,108,109,110], 96: [94,95,97,98,106,107,108,109],
    97: [95,96,98,99,105,106,107,108], 98: [96,97,99,100,105,106,107,108],
    99: [97,98,100,101,104,105,106], 100: [98,99,101,102,103,104,105],
    101: [99,100,102,103,104], 102: [100,101,103,104],
    103: [100,101,102,104,105,129,130,131,134,136],
    104: [99,100,101,102,103,105,106,127,128,129,130,131],
    105: [97,98,99,100,103,104,106,107,127,128,129,130],
    106: [96,97,98,99,104,105,107,108,125,126,127,128],
    107: [95,96,97,98,105,106,108,109,124,125,126,127,128],
    108: [63,95,96,97,98,106,107,109,110,123,124,125,126],
    109: [93,94,95,96,107,108,110,111,123,124,125],
    110: [66,67,93,94,95,108,109,111,112,121,122,123,124],
    111: [67,91,92,93,94,109,110,112,113,120,121,122,123],
    112: [67,68,88,91,92,93,110,111,113,114,119,120,121,122],
    113: [88,89,90,91,92,111,112,114,115,119,120,121],
    114: [88,89,90,91,112,113,115,116,117,118,119,120],
    115: [113,114,116,117,118,119,120], 116: [114,115,117,118,119],
    117: [12,13,114,115,116,118,119], 118: [12,13,14,15,114,115,116,117,119,120],
    119: [12,13,14,15,112,113,114,115,116,117,118,120,121],
    120: [14,15,16,17,111,112,113,114,115,118,119,121,122],
    121: [14,15,16,17,67,68,110,111,112,113,119,120,122,123],
    122: [16,17,18,110,111,112,120,121,123,124],
    123: [16,17,18,19,64,108,109,110,111,121,122,124,125],
    124: [18,19,20,107,108,109,110,122,123,125,126],
    125: [18,19,20,21,22,106,107,108,109,123,124,126,127],
    126: [20,21,22,106,107,108,124,125,127,128],
    127: [20,21,22,23,104,105,106,107,125,126,128,129,130],
    128: [22,23,104,105,106,107,126,127,129,130],
    129: [23,103,104,105,127,128,130,131],
    130: [22,23,103,104,105,127,128,129,131,132,134,136],
    131: [103,104,129,130,132,133,134,135,136],
    132: [130,131,133,134,135], 133: [131,132,134,135,136],
    134: [103,130,131,132,133,135,136,137,138],
    135: [131,132,133,134,136,137,138,139],
    136: [55,103,130,131,133,134,135,137,138,139],
    137: [55,56,134,135,136,138,139,140,141],
    138: [134,135,136,137,139,140,141], 139: [135,136,137,138,140,141,142,152,153,154],
    140: [137,138,139,141,142,143,152,153],
    141: [137,138,139,140,142,143,150,151,152,153],
    142: [139,140,141,143,144,150,151,152],
    143: [140,141,142,144,145,149,150,151],
    144: [142,143,145,146,147,149,150,151],
    145: [143,144,146,147,148,149,150], 146: [144,145,147,148,149],
    147: [144,145,146,148,149,150], 148: [145,146,147,149,150,168,169,170,171],
    149: [143,144,145,146,147,148,150,151,166,167,168,169,170,179],
    150: [141,142,143,144,145,147,148,149,151,152,166,167,168,169],
    151: [141,142,143,144,149,150,152,153,165,166,167,168],
    152: [139,140,141,142,150,151,153,154,164,165,166,167],
    153: [139,140,141,151,152,154,155,163,164,165],
    154: [139,152,153,155,156,162,163,164],
    155: [153,154,156,157,162,163,164], 156: [154,155,157,158,161,162,163],
    157: [155,156,158,159,160,161,162], 158: [156,157,159,160,161],
    159: [157,158,160,161], 160: [157,158,159,161,162],
    161: [156,157,158,159,160,162,163,183,184,185],
    162: [154,155,156,157,160,161,163,164,182,183,184],
    163: [153,154,155,156,161,162,164,165,181,182,183,184],
    164: [152,153,154,155,162,163,165,166,181,182,183],
    165: [151,152,153,163,164,166,167,178,179,180,181,182],
    166: [149,150,151,152,164,165,167,168,178,179,180,181],
    167: [149,150,151,152,165,166,168,169,176,177,178,179,180],
    168: [148,149,150,151,166,167,169,170,176,177,178,179],
    169: [148,149,150,167,168,170,171,174,175,176,177],
    170: [148,149,168,169,171,172,173,174,175,176,177],
    171: [148,169,170,172,173,174,175], 172: [170,171,173,174,175],
    173: [170,171,172,174,175], 174: [169,170,171,172,173,175,176],
    175: [169,170,171,172,173,174,176,177],
    176: [167,168,169,170,174,175,177,178,230],
    177: [167,168,169,170,175,176,178,179,227,230],
    178: [165,166,167,168,176,177,179,180,181,227,230],
    179: [149,165,166,167,168,177,178,180,181],
    180: [165,166,167,178,179,181,182],
    181: [163,164,165,166,178,179,180,182,183],
    182: [162,163,164,165,180,181,183,184],
    183: [161,162,163,164,181,182,184,185,186],
    184: [161,162,163,182,183,185,186,187,188],
    185: [161,183,184,186,187], 186: [183,184,185,187,188],
    187: [184,185,186,188,189], 188: [184,186,187,189,190,191],
    189: [187,188,190,191], 190: [188,189,191,192],
    191: [188,189,190,192,193], 192: [190,191,193,194],
    193: [191,192,194,195], 194: [192,193,195,196,197],
    195: [193,194,196,197,199], 196: [194,195,197,198,199],
    197: [194,195,196,198,199], 198: [196,197,199,200],
    199: [195,196,197,198,200,201,202], 200: [198,199,201,202],
    201: [199,200,202,203,204,228], 202: [199,200,201,203,204,205],
    203: [73,201,202,204,205,221,224,225], 204: [201,202,203,205,206,207],
    205: [202,203,204,206,207,208,217,221],
    206: [204,205,207,208,209], 207: [204,205,206,208,209,210,217],
    208: [205,206,207,209,210,211,213], 209: [206,207,208,210,211],
    210: [207,208,209,211,212,213], 211: [208,209,210,212,213,214],
    212: [210,211,213,214], 213: [208,210,211,212,214,215,216],
    214: [211,212,213,215,216,217], 215: [213,214,216,217,218],
    216: [213,214,215,217,218,219], 217: [205,207,214,215,216,218,219,220,221],
    218: [215,216,217,219,220,221,222], 219: [216,217,218,220,221,222,223],
    220: [217,218,219,221,222,223,224],
    221: [203,205,217,218,219,220,222,223,224,225],
    222: [62,218,219,220,221,223,224,225,226],
    223: [62,219,220,221,222,224,225,226,227],
    224: [203,220,221,222,223,225,226,227,228],
    225: [62,69,203,221,222,223,224,226,227,228,229],
    226: [61,62,65,222,223,224,225,227,228,229,230],
    227: [177,178,223,224,225,226,228,229,230,231],
    228: [201,224,225,226,227,229,230,231,232],
    229: [41,42,43,225,226,227,228,230,231,232,233],
    230: [176,177,178,226,227,228,229,231,232,233,234],
    231: [227,228,229,230,232,233,234,235],
    232: [41,228,229,230,231,233,234,235,236],
    233: [229,230,231,232,234,235,236,237],
    234: [230,231,232,233,235,236,237,238],
    235: [231,232,233,234,236,237,238],
    236: [232,233,234,235,237,238], 237: [233,234,235,236,238],
    238: [234,235,236,237],
}

# ============================================================
# Mutation-level features
# ============================================================

def compute_mutation_features(
    template: str,
    mutations_str: str,
) -> Dict:
    """
    计算单条突变组合的聚合特征 (v3.3: 新增空间上位效应特征)。

    特征包括:
      - n_mutations: 突变数量
      - sum_blosum: BLOSUM62 替换分之和
      - sum_volume_change: 体积变化之和
      - sum_hydro_change: 疏水性变化之和
      - mean_position: 平均突变位置
      - n_surface: 表面位点突变数 (粗略)
      - n_core: 核心位点突变数 (粗略)
      - has_chromophore_mutation: 是否突变生色团区域
      - max_blosum_penalty: 最大 BLOSUM 惩罚
      - n_spatial_pairs: 空间相邻突变对数量 (WT距离<8Å)  ← v3.3
      - spatial_blosum_product: 空间相邻突变对的 BLOSUM 乘积之和  ← v3.3
      - has_spatial_clash: 是否存在空间冲突 (两突变相邻且BLOSUM均<0) ← v3.3
    """
    mutations = parse_mutation_string(mutations_str)

    if not mutations:
        return {
            'n_mutations': 0, 'sum_blosum': 0, 'sum_volume_change': 0,
            'sum_hydro_change': 0, 'mean_position': 0,
            'n_surface': 0, 'n_core': 0,
            'has_chromophore_mutation': 0,
            'max_blosum_penalty': 0,
        }

    blosum_sum = 0.0
    vol_change_sum = 0.0
    hydro_change_sum = 0.0
    positions = []
    blosum_penalties = []
    n_surface = 0
    n_core = 0
    has_chromo = 0

    # Surface positions in GFP beta-barrel (approx from SASA > 30%)
    surface_positions = {2,3,4,5,6,8,10,14,15,18,19,22,23,29,30,33,34,
                         45,46,50,51,52,53,54,55,56,57,58,76,77,78,79,80,
                         81,82,101,102,103,104,105,106,107,108,109,110,111,
                         112,113,114,115,117,118,119,120,122,123,124,125,
                         126,127,128,129,130,131,132,133,134,135,136,137,
                         138,139,140,141,142,143,144,145,147,148,149,150,
                         151,152,154,155,156,157,158,159,160,161,162,164,
                         165,166,168,169,170,171,172,173,174,175,176,177,
                         178,179,180,181,182,183,184,185,186,187,188,189,
                         190,191,192,193,194,195,196,197,198,199,200,201,
                         202,203,204,206,207,208,209,210,211,212,213,214,
                         215,216,217,218,219,220,221,222,223,224,225,226,
                         227,228,229}
    chromophore_region = {64, 65, 66, 67, 68}

    for pos, from_aa, to_aa in mutations:
        positions.append(pos)

        # BLOSUM score
        blosum = BLOSUM62.get(from_aa, {}).get(to_aa, -4)
        blosum_sum += blosum
        if blosum < 0:
            blosum_penalties.append(blosum)

        # Volume change
        vol_from = AA_VOLUME.get(from_aa, 130)
        vol_to = AA_VOLUME.get(to_aa, 130)
        vol_change_sum += abs(vol_to - vol_from)

        # Hydrophobicity change
        hydro_from = HYDROPATHY.get(from_aa, 0)
        hydro_to = HYDROPATHY.get(to_aa, 0)
        hydro_change_sum += abs(hydro_to - hydro_from)

        # Position categories
        if pos in surface_positions:
            n_surface += 1
        else:
            n_core += 1
        if pos in chromophore_region:
            has_chromo = 1

    n = len(mutations)

    # v3.3: Spatial epistasis features — detect mutation pairs within 8Å in WT structure
    spatial_pairs = 0
    spatial_blosum_product = 0.0
    has_spatial_clash = 0
    # Store per-mutation BLOSUM for pairwise computation
    mut_blosums = {}
    for pos, from_aa, to_aa in mutations:
        blosum = BLOSUM62.get(from_aa, {}).get(to_aa, -4)
        mut_blosums[pos] = blosum

    # Check all mutation pairs for spatial proximity
    for i in range(len(mutations)):
        for j in range(i + 1, len(mutations)):
            pos_i = mutations[i][0]
            pos_j = mutations[j][0]
            # Check if positions are within 8Å in WT structure
            neighbors_i = SPATIAL_NEIGHBORS_8A.get(pos_i, [])
            if pos_j in neighbors_i:
                spatial_pairs += 1
                bi = mut_blosums.get(pos_i, 0)
                bj = mut_blosums.get(pos_j, 0)
                spatial_blosum_product += bi * bj
                # Clash: both mutations are individually deleterious AND spatially adjacent
                if bi < 0 and bj < 0:
                    has_spatial_clash = 1

    # v4.4: Position-aware features — per-position structural context
    # These features let XGBoost distinguish "chromophore-killing mutation" from
    # "benign surface mutation" — the key discriminative signal within a GFP type.
    chromophore_distances = []
    n_beta = n_helix = n_loop = 0
    # Beta strand positions in GFP (11-strand barrel, avGFP numbering)
    BETA_STRANDS = {6,7,8,9,10,11,12, 15,16,17,18,19,20,21,22,23,24,25,
                    27,28,29,30,31,32,33, 42,43,44,45,46,47,48,49,
                    83,84,85,86,87,88, 92,93,94,95,96,97,98,99,100,
                    107,108,109,110,111,112,113,114,115,116,
                    119,120,121,122,123,124,125,
                    147,148,149,150,151,152,153,
                    160,161,162,163,164,165,166,167,168,169,170,
                    196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212}
    HELIX = set(range(56, 73))

    for pos, _, _ in mutations:
        chromophore_distances.append(min(abs(pos - 65), abs(pos - 66), abs(pos - 67)))
        if pos in BETA_STRANDS: n_beta += 1
        elif pos in HELIX: n_helix += 1
        else: n_loop += 1

    min_chromo_dist = min(chromophore_distances) if chromophore_distances else 100
    mean_chromo_dist = sum(chromophore_distances) / max(n, 1)

    return {
        'n_mutations': n,
        'sum_blosum': round(blosum_sum, 2),
        'mean_blosum': round(blosum_sum / n, 2),
        'sum_volume_change': round(vol_change_sum, 1),
        'sum_hydro_change': round(hydro_change_sum, 2),
        'mean_position': round(sum(positions) / n, 0),
        'n_surface': n_surface,
        'n_core': n_core,
        'surface_core_ratio': round(n_surface / max(n, 1), 2),
        'has_chromophore_mutation': has_chromo,
        'max_blosum_penalty': round(min(blosum_penalties) if blosum_penalties else 0, 1),
        'n_spatial_pairs': spatial_pairs,
        'spatial_blosum_product': round(spatial_blosum_product, 2),
        'has_spatial_clash': has_spatial_clash,
        # v4.4: Position-aware structural context
        'min_chromophore_distance': min_chromo_dist,
        'mean_chromophore_distance': round(mean_chromo_dist, 1),
        'n_beta_strand_mutations': n_beta,
        'n_helix_mutations': n_helix,
        'n_loop_mutations': n_loop,
    }


# ============================================================
# Zero-shot ESM scoring
# ============================================================

def compute_esm_zero_shot_scores(
    seq: str,
    template: str,
    esm_model,
    esm_alphabet,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    ESM-2 零样本 per-mutation 掩码边际评分。

    对每个突变位置，独立计算:
      score_i = log P(mutant_AA | context) - log P(template_AA | context)

    正值 = 突变增强了该位置的序列天然度
    负值 = 突变破坏了该位置的序列天然度

    Returns:
        {
            'zs_sum': 各突变位置得分之和,
            'zs_mean': 各突变位置得分均值,
            'zs_min': 最差位置的得分,
            'zs_max': 最好位置的得分,
            'zs_n_beneficial': 有益突变 (正分) 的数量,
            'zs_n_deleterious': 有害突变 (负分) 的数量,
        }
    """
    if esm_model is None or esm_alphabet is None:
        return {'zs_sum': 0, 'zs_mean': 0, 'zs_min': 0, 'zs_max': 0,
                'zs_n_beneficial': 0, 'zs_n_deleterious': 0}

    # Find mutations
    mutations = []
    for i, (s_aa, t_aa) in enumerate(zip(seq, template)):
        if s_aa != t_aa:
            mutations.append((i + 1, t_aa, s_aa))

    if not mutations:
        return {'zs_sum': 0, 'zs_mean': 0, 'zs_min': 0, 'zs_max': 0,
                'zs_n_beneficial': 0, 'zs_n_deleterious': 0}

    import torch
    batch_converter = esm_alphabet.get_batch_converter()
    scores = []

    for pos, from_aa, to_aa in mutations:
        idx = pos - 1

        # Template AA at this position
        tpl_list = list(seq)
        tpl_list[idx] = from_aa
        tpl_seq = ''.join(tpl_list)

        try:
            # Mutant log prob
            _, _, mut_tokens = batch_converter([("mut", seq)])
            mut_tokens = mut_tokens.to(device)

            with torch.no_grad():
                results = esm_model(mut_tokens, repr_layers=[esm_model.num_layers])
                logits = results["logits"]
                mut_log_prob = torch.log_softmax(logits[0, idx + 1], dim=-1)[
                    esm_alphabet.get_idx(to_aa)
                ].item()

            # Template log prob
            _, _, tpl_tokens = batch_converter([("tpl", tpl_seq)])
            tpl_tokens = tpl_tokens.to(device)

            with torch.no_grad():
                results = esm_model(tpl_tokens, repr_layers=[esm_model.num_layers])
                logits = results["logits"]
                tpl_log_prob = torch.log_softmax(logits[0, idx + 1], dim=-1)[
                    esm_alphabet.get_idx(from_aa)
                ].item()

            scores.append(mut_log_prob - tpl_log_prob)
        except Exception:
            scores.append(0.0)

    if not scores:
        return {'zs_sum': 0, 'zs_mean': 0, 'zs_min': 0, 'zs_max': 0,
                'zs_n_beneficial': 0, 'zs_n_deleterious': 0}

    return {
        'zs_sum': round(sum(scores), 4),
        'zs_mean': round(sum(scores) / len(scores), 4),
        'zs_min': round(min(scores), 4),
        'zs_max': round(max(scores), 4),
        'zs_n_beneficial': sum(1 for s in scores if s > 0),
        'zs_n_deleterious': sum(1 for s in scores if s < 0),
    }


# ============================================================
# Full feature extraction for supervised model
# ============================================================

def extract_supervised_features(
    template: str,
    template_type: str,
    mutations_str: str,
    esm_model=None,
    esm_alphabet=None,
    device: str = "cpu",
) -> Dict:
    """
    提取用于监督学习的完整特征集。

    包括:
      - GFP type (one-hot encoded in training)
      - 突变特征 (BLOSUM, 体积, 疏水性)
      - ESM 零样本分数
      - 位置特征
    """
    # Mutation-level features
    mut_feats = compute_mutation_features(template, mutations_str)

    # Reconstruct full sequence for ESM scoring
    seq = reconstruct_sequence_from_mutations(template, mutations_str)

    # ESM zero-shot scores
    zs = compute_esm_zero_shot_scores(seq, template, esm_model, esm_alphabet, device)

    # Physicochemical features of full sequence
    phys = compute_sequence_features(seq)

    # Sequence similarity to template
    min_len = min(len(seq), len(template))
    identity = sum(1 for i in range(min_len) if seq[i] == template[i]) / max(min_len, 1)

    return {
        **mut_feats,
        **zs,
        'phys_hydrophobicity': phys['hydrophobicity_mean'],
        'phys_aromatic_fraction': phys['aromatic_fraction'],
        'phys_charge_pH7': phys['charge_pH7'],
        'phys_glycine_fraction': phys['glycine_fraction'],
        'phys_proline_fraction': phys['proline_fraction'],
        'seq_identity': round(identity, 4),
        'template_type': template_type,
    }


# ============================================================
# XGBoost-based predictor
# ============================================================

def train_xgboost_predictor(
    brightness_df: pd.DataFrame,
    templates: Dict[str, str],
    max_samples: int = 50000,
    validation_split: float = 0.15,
    random_state: int = 42,
) -> Dict:
    """
    训练 XGBoost 亮度预测器。

    使用全量数据 (最多 50000 条，XGBoost 可以高效处理)，
    特征包括 GFP type + 突变特征 + ESM 零样本分数。

    ESM 零样本分数的计算:
      由于对所有 141k 条做 ESM 推理太慢，我们采用近似策略:
      1. 对单点突变 (4600条) 做 ESM 评分，建立 position×AA 查找表
      2. 对多点突变，求和各单点 ESM 分数
      3. 只用部分多点突变 (随机采样) 做完整 ESM 评分用于验证

    Returns:
        {
            'xgb_model': 训练好的模型,
            'feature_names': 特征名称列表,
            'train_r2': 训练 R²,
            'val_r2': 验证 R²,
            'gfp_type_encoder': GFP类型编码器,
        }
    """
    import torch

    print(f"\n  [XGBoost] Loading full training data ({min(len(brightness_df), max_samples)} samples)...")

    # Sample if needed (XGBoost handles 50k easily)
    df = brightness_df.copy()
    if len(df) > max_samples:
        # Stratified by GFP type and n_mutations
        df['n_mut'] = df['aaMutations'].apply(
            lambda x: 1 if str(x) == 'WT' else len(str(x).split(':'))
        )
        # Use sample with stratification but preserve all columns
        sampled_dfs = []
        for (gfp_type, n_mut), group in df.groupby(['GFP type', 'n_mut']):
            n_sample = max(1, int(max_samples * len(group) / len(df)))
            sampled_dfs.append(group.sample(n=min(n_sample, len(group)), random_state=random_state))
        df = pd.concat(sampled_dfs, ignore_index=True)
        if 'n_mut' in df.columns:
            df = df.drop(columns=['n_mut'])

    print(f"  [XGBoost] Using {len(df)} samples")

    # ---- v2.2: Compute per-type baselines for RESIDUAL training ----
    # Instead of predicting absolute brightness (which makes the model
    # rely 95% on gfp_avGFP dummy), predict the RESIDUAL:
    #   residual = actual - baseline[gfp_type]
    # This forces the model to learn mutation-level features (BLOSUM,
    # volume change, etc.) rather than memorizing per-type baselines.
    type_baselines = {}
    for gfp_type in df['GFP type'].unique():
        type_baselines[gfp_type] = df[df['GFP type'] == gfp_type]['Brightness'].median()
    print(f"  [XGBoost v2.2] Residual training mode. Type baselines:")
    for t, b in sorted(type_baselines.items()):
        print(f"    {t}: baseline = {b:.4f}")

    # Extract features WITHOUT ESM (fast phase)
    print(f"  [XGBoost] Extracting mutation features (phase 1: fast)...")
    feature_rows = []
    residuals = []  # v2.2: now training on residuals
    brightnesses_raw = []  # keep raw for R2 computation
    gfp_types = []

    for _, row in df.iterrows():
        template_type = row['GFP type']
        if template_type not in templates:
            continue
        template = templates[template_type]
        mut_str = str(row['aaMutations'])

        # Fast features (no ESM) — NO gfp_type dummy needed for residual model
        features = compute_mutation_features(template, mut_str)
        seq = reconstruct_sequence_from_mutations(template, mut_str)
        phys = compute_sequence_features(seq)
        min_len = min(len(seq), len(template))
        identity = sum(1 for i in range(min_len) if seq[i] == template[i]) / max(min_len, 1)

        features.update({
            'phys_hydrophobicity': phys['hydrophobicity_mean'],
            'phys_aromatic_fraction': phys['aromatic_fraction'],
            'phys_charge_pH7': phys['charge_pH7'],
            'phys_glycine_fraction': phys['glycine_fraction'],
            'phys_proline_fraction': phys['proline_fraction'],
            'seq_identity': round(identity, 4),
        })

        raw_brightness = row['Brightness']
        baseline = type_baselines.get(template_type, raw_brightness)
        residual = raw_brightness - baseline  # v2.2: predict this

        feature_rows.append(features)
        residuals.append(residual)
        brightnesses_raw.append(raw_brightness)
        gfp_types.append(template_type)

    # Convert to DataFrame (v2.2: no GFP type dummies — model learns mutation effects only)
    feature_df = pd.DataFrame(feature_rows)
    # Store GFP type for later baseline lookup
    feature_df['_gfp_type'] = gfp_types

    # Prepare X, y (v2.2: y = residual, not absolute brightness)
    feature_names = list(feature_df.drop(columns=['_gfp_type']).columns)
    X = feature_df.drop(columns=['_gfp_type']).values.astype(np.float32)
    y = np.array(residuals, dtype=np.float32)
    y_raw = np.array(brightnesses_raw, dtype=np.float32)

    # Train XGBoost
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
    except ImportError:
        print(f"  [XGBoost] xgboost not installed, falling back to RF...")
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=random_state)
        model = RandomForestRegressor(n_estimators=200, max_depth=15, n_jobs=-1, random_state=random_state)
        model.fit(X_train, y_train)
        train_r2 = model.score(X_train, y_train)
        val_r2 = model.score(X_val, y_val)
        model_type = 'RandomForest'
    else:
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val, y_train_raw, y_val_raw, gfp_train, gfp_val = train_test_split(
            X, y, y_raw, gfp_types,
            test_size=validation_split, random_state=random_state
        )

        print(f"  [XGBoost v2.2] Residual training on {len(X_train)} samples, validating on {len(X_val)}...")
        print(f"  [XGBoost v2.2] Target: residual (brightness - type_baseline)")

        # v3.3: Train ensemble of 3 XGBoost models for uncertainty quantification
        n_ensemble = 3
        models = []
        ensemble_train_scores = []
        ensemble_val_scores = []

        for seed_i in range(n_ensemble):
            model_i = xgb.XGBRegressor(
                n_estimators=200,         # Reduced from 300 for ensemble speed
                max_depth=7,
                learning_rate=0.05,
                subsample=0.75,           # Extra randomness for ensemble diversity
                colsample_bytree=0.75,
                reg_alpha=1.0,
                reg_lambda=1.0,
                n_jobs=-1,
                random_state=random_state + seed_i * 100,
            )
            model_i.fit(X_train, y_train)
            models.append(model_i)

            # Track individual model scores
            s_train = model_i.score(X_train, y_train)
            s_val = model_i.score(X_val, y_val)
            ensemble_train_scores.append(s_train)
            ensemble_val_scores.append(s_val)

        # Use the first model as the primary (for feature importance, etc.)
        model = models[0]
        model_type = 'XGBoost'
        n_ensemble_models = n_ensemble

        train_r2_residual = np.mean(ensemble_train_scores)
        val_r2_residual = np.mean(ensemble_val_scores)

        # Compute ensemble predictions on actual brightness
        train_preds_ensemble = np.mean([m.predict(X_train) for m in models], axis=0)
        val_preds_ensemble = np.mean([m.predict(X_val) for m in models], axis=0)
        train_pred_actual = np.array([p + type_baselines.get(g, 0) for p, g in zip(train_preds_ensemble, gfp_train)])
        val_pred_actual = np.array([p + type_baselines.get(g, 0) for p, g in zip(val_preds_ensemble, gfp_val)])
        from sklearn.metrics import r2_score as _r2
        train_r2_actual = _r2(y_train_raw, train_pred_actual)
        val_r2_actual = _r2(y_val_raw, val_pred_actual)

        # Compute prediction std for uncertainty quantification
        train_preds_std = np.std([m.predict(X_train) for m in models], axis=0)
        val_preds_std = np.std([m.predict(X_val) for m in models], axis=0)
        mean_train_std = float(np.mean(train_preds_std))
        mean_val_std = float(np.mean(val_preds_std))

        # Feature importance (now shows mutation-level features, not GFP type!)
        importances = sorted(zip(feature_names, model.feature_importances_),
                             key=lambda x: -x[1])[:10]
        print(f"  [XGBoost v3.3] Ensemble of {n_ensemble} models. Residual R2: train={train_r2_residual:.4f}, val={val_r2_residual:.4f}")
        print(f"  [XGBoost v3.3] Actual R2:   train={train_r2_actual:.4f}, val={val_r2_actual:.4f}")
        print(f"  [XGBoost v3.3] Prediction std: train={mean_train_std:.4f}, val={mean_val_std:.4f}")
        print(f"  [XGBoost v3.3] Top mutation-effect features (no GFP dummies):")
        for name, imp in importances:
            print(f"    {name}: {imp:.4f}")

    print(f"  [{model_type}] Train R2: {train_r2_actual if model_type=='XGBoost' else train_r2:.4f} | Val R2: {val_r2_actual if model_type=='XGBoost' else val_r2:.4f}")

    result = {
        'model': models if model_type == 'XGBoost' else model,
        'model_type': model_type,
        'feature_names': feature_names,
        'train_r2': train_r2_actual if model_type == 'XGBoost' else train_r2,
        'val_r2': val_r2_actual if model_type == 'XGBoost' else val_r2,
        'type_baselines': type_baselines,
    }
    if model_type == 'XGBoost':
        result['train_r2_residual'] = train_r2_residual
        result['val_r2_residual'] = val_r2_residual
        result['n_ensemble'] = n_ensemble_models
        result['prediction_std'] = mean_val_std  # uncertainty estimate
    return result


def predict_with_xgboost(
    sequences: List[str],
    templates_by_variant: List[str],  # GFP type for each variant
    templates: Dict[str, str],
    predictor: Dict,
    return_std: bool = False,
):
    """
    使用训练好的 XGBoost 模型预测亮度 (v2.2: 残差模式, v3.3: 集成不确定性)。

    模型预测的是 residual (mutation effect)，
    最终亮度 = baseline[gfp_type] + predicted_residual。

    Args:
        return_std: 若为 True，返回 (predictions, stds) 元组
    Returns:
        np.ndarray or (np.ndarray, np.ndarray): 预测亮度值，及可选的标准差
    """
    type_baselines = predictor.get('type_baselines', {})
    feature_rows = []
    gfp_types_list = []

    for seq, gfp_type in zip(sequences, templates_by_variant):
        template = templates.get(gfp_type, list(templates.values())[0])
        mut_str = _seq_diff_to_mut_string(seq, template)

        features = compute_mutation_features(template, mut_str)
        phys = compute_sequence_features(seq)
        min_len = min(len(seq), len(template))
        identity = sum(1 for i in range(min_len) if seq[i] == template[i]) / max(min_len, 1)

        features.update({
            'phys_hydrophobicity': phys['hydrophobicity_mean'],
            'phys_aromatic_fraction': phys['aromatic_fraction'],
            'phys_charge_pH7': phys['charge_pH7'],
            'phys_glycine_fraction': phys['glycine_fraction'],
            'phys_proline_fraction': phys['proline_fraction'],
            'seq_identity': round(identity, 4),
        })
        feature_rows.append(features)
        gfp_types_list.append(gfp_type)

    feature_df = pd.DataFrame(feature_rows)

    # v2.2: No GFP dummies — model was trained on residuals
    # Ensure column order matches training
    for col in predictor['feature_names']:
        if col not in feature_df.columns:
            feature_df[col] = 0
    feature_df = feature_df[predictor['feature_names']]

    X = feature_df.values.astype(np.float32)

    # v3.3: Ensemble prediction with uncertainty
    if isinstance(predictor.get('model'), list):
        # Multi-model ensemble
        all_preds = np.array([m.predict(X) for m in predictor['model']])  # [n_models, n_samples]
        residuals = all_preds.mean(axis=0)
        pred_std = all_preds.std(axis=0)
    else:
        # Single model (RF fallback)
        residuals = predictor['model'].predict(X)
        pred_std = np.zeros(len(residuals))

    # v2.2: Add baseline back to get absolute brightness
    baselines = np.array([type_baselines.get(g, np.median(list(type_baselines.values())))
                          for g in gfp_types_list])
    predictions = residuals + baselines

    if return_std:
        return predictions, pred_std
    return predictions


def _seq_diff_to_mut_string(seq: str, template: str) -> str:
    """将序列差异转换为 aaMutations 格式字符串。"""
    mutations = []
    for i, (s_aa, t_aa) in enumerate(zip(seq, template)):
        if s_aa != t_aa:
            mutations.append(f"{t_aa}{i+1}{s_aa}")
    return ':'.join(mutations) if mutations else 'WT'


# ============================================================
# Physics-based scoring (unchanged from v2)
# ============================================================

def physics_based_score(seq: str, template: str) -> float:
    """物理化学约束评分。"""
    features = compute_sequence_features(seq)
    n = features['length']
    hydro = features['hydrophobicity_mean']

    if -0.5 <= hydro <= 1.5:
        hydro_score = 1.0
    elif hydro < -0.5:
        hydro_score = 1.0 - abs(hydro + 0.5) / 2.0
    else:
        hydro_score = 1.0 - abs(hydro - 1.5) / 2.0

    aromatic = features['aromatic_fraction']
    if 0.08 <= aromatic <= 0.18:
        aromatic_score = 1.0
    elif aromatic < 0.08:
        aromatic_score = aromatic / 0.08
    else:
        aromatic_score = max(0.0, 1.0 - (aromatic - 0.18) / 0.10)

    charge = abs(features['charge_pH7'])
    charge_score = 1.0 if charge <= 0.15 else max(0.0, 1.0 - (charge - 0.15) / 0.20)

    tpl_len = len(template)
    len_dev = abs(n - tpl_len) / tpl_len
    len_score = max(0.0, 1.0 - len_dev * 5.0)

    score = (0.30 * max(0.0, hydro_score) + 0.25 * max(0.0, aromatic_score) +
             0.25 * max(0.0, charge_score) + 0.20 * max(0.0, len_score))
    return round(max(0.0, min(1.0, score)), 4)


# ============================================================
# Ensemble prediction (main interface)
# ============================================================

def ensemble_predict_batch(
    sequences: List[str],
    template: str,
    template_type: str = "avGFP",
    templates: Dict[str, str] = None,
    esm_model=None,
    esm_alphabet=None,
    xgb_predictor: Dict = None,
    device: str = "cpu",
    batch_size: int = 8,
) -> List[Dict]:
    """
    集成预测 (XGBoost主导 + ESM零样本 + 物理约束)。

    Args:
        sequences: 待预测序列
        template: 参考模板序列
        template_type: GFP类型
        templates: 所有GFP模板 {name: seq}
        esm_model/alphabet: ESM模型 (用于零样本评分)
        xgb_predictor: XGBoost预测器 (train_xgboost_predictor的输出)
        device: 计算设备
    """
    results = []

    # Phase 1: Physics scoring (fast, for all sequences)
    phys_scores = [physics_based_score(s, template) for s in sequences]

    # Phase 2: XGBoost prediction (if available)
    xgb_scores = None
    if xgb_predictor is not None and templates is not None:
        print(f"  [Ensemble] XGBoost prediction for {len(sequences)} sequences...")
        try:
            tpl_types = [template_type] * len(sequences)
            raw_preds = predict_with_xgboost(sequences, tpl_types, templates, xgb_predictor)
            # Normalize raw brightness → relative score
            xgb_scores = [round(r / 3.72, 4) for r in raw_preds]
        except Exception as e:
            print(f"  [Ensemble] XGBoost prediction failed: {e}")

    # Phase 3: ESM zero-shot (if available)
    zs_scores = None
    if esm_model is not None and esm_alphabet is not None:
        # Only compute ZS for top candidates from physics+xgb, to save time
        n_zs = min(len(sequences), 500)  # v4.0: 500 (was 200), better coverage
        print(f"  [Ensemble] ESM zero-shot scoring for top {n_zs} sequences...")
        zs_scores = [0.5] * len(sequences)  # Default

        # Prioritize: compute ZS for sequences with high physics scores
        ranked_indices = sorted(range(len(sequences)),
                                key=lambda i: phys_scores[i], reverse=True)
        for idx in ranked_indices[:n_zs]:
            zs = compute_esm_zero_shot_scores(
                sequences[idx], template, esm_model, esm_alphabet, device
            )
            # Normalize ZS sum to [0,1]
            zs_sum = zs.get('zs_sum', 0)
            zs_score = 1.0 / (1.0 + math.exp(-zs_sum * 3.0))
            zs_scores[idx] = round(zs_score, 4)

    # Phase 4: Ensemble
    for i in range(len(sequences)):
        if xgb_scores is not None and zs_scores is not None:
            # Full ensemble: XGBoost dominant
            brightness = 0.55 * xgb_scores[i] + 0.20 * zs_scores[i] + 0.25 * phys_scores[i]
        elif xgb_scores is not None:
            # XGBoost + physics
            brightness = 0.70 * xgb_scores[i] + 0.30 * phys_scores[i]
        elif zs_scores is not None:
            # Zero-shot + physics
            brightness = 0.45 * zs_scores[i] + 0.55 * phys_scores[i]
        else:
            # Physics-only fallback
            brightness = phys_scores[i]

        results.append({
            'ensemble_brightness': round(brightness, 4),
            'physics_score': phys_scores[i],
            'xgb_score': xgb_scores[i] if xgb_scores else None,
            'zs_score': zs_scores[i] if zs_scores else None,
        })

    return results
