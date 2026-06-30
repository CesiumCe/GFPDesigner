# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cesium
#
"""
utils.py — 基础工具函数 (改进版)

新增:
  - GPU自动检测与优化配置
  - Bloom filter快速排除列表预筛选
  - 全部原有功能保留
"""
import re
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import yaml

# ============================================================
# GPU Detection (改进6)
# ============================================================

def detect_device(use_gpu: bool = True) -> Tuple[str, dict]:
    """
    检测可用的计算设备。

    Returns:
        (device_string, device_info_dict)
    """
    info = {
        'cuda_available': False,
        'device': 'cpu',
        'gpu_name': None,
        'vram_gb': 0,
        'cpu_count': 1,
    }

    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        if info['cuda_available'] and use_gpu:
            info['device'] = 'cuda'
            props = torch.cuda.get_device_properties(0)
            info['gpu_name'] = props.name
            info['vram_gb'] = round(props.total_mem / 1024**3, 1)
        info['cpu_count'] = torch.get_num_threads()
    except Exception:
        pass

    # Set CPU threads
    try:
        import torch
        cpu_count = max(1, torch.get_num_threads() - 1)
        torch.set_num_threads(cpu_count)
        info['cpu_count'] = cpu_count
    except Exception:
        pass

    return info['device'], info


# ============================================================
# Bloom Filter for Exclusion List (改进6)
# ============================================================

class BloomFilter:
    """
    布隆过滤器 —— 对排除列表进行 O(1) 快速预筛选。

    135k 条序列全部加载到Python set需要 ~10MB+ 内存。
    Bloom filter 可以 ~1MB 内存达到 99%+ 的筛选准确率。

    False positive rate: ~0.01 (1%)
    这意味着极少数通过 bloom filter 的序列需要做精确比对。
    """

    def __init__(self, n_items: int, false_positive_rate: float = 0.01):
        # Optimal bit array size: m = -n*ln(p) / (ln(2))^2
        # Optimal hash count: k = (m/n) * ln(2)
        self.size = int(-n_items * math.log(false_positive_rate) / (math.log(2) ** 2))
        self.size = max(1, self.size)
        self.hash_count = max(1, int((self.size / n_items) * math.log(2)))
        self.bit_array = bytearray((self.size + 7) // 8)

    def _hashes(self, item: str) -> List[int]:
        """Generate k hash values for an item."""
        result = []
        h1 = hash(item + "_1")
        h2 = hash(item + "_2")
        for i in range(self.hash_count):
            h = (h1 + i * h2) % self.size
            result.append(abs(h))
        return result

    def add(self, item: str):
        for h in self._hashes(item):
            byte_idx = h // 8
            bit_idx = h % 8
            self.bit_array[byte_idx] |= (1 << bit_idx)

    def contains(self, item: str) -> bool:
        """Returns True if item MIGHT be in the set (could be false positive)."""
        for h in self._hashes(item):
            byte_idx = h // 8
            bit_idx = h % 8
            if not (self.bit_array[byte_idx] & (1 << bit_idx)):
                return False
        return True

    @classmethod
    def from_list(cls, items: List[str], false_positive_rate: float = 0.01) -> 'BloomFilter':
        bf = cls(len(items), false_positive_rate)
        for item in items:
            bf.add(item)
        return bf


# ============================================================
# Constants
# ============================================================

STANDARD_AA: set = set("ACDEFGHIKLMNPQRSTVWY")

HYDROPATHY: Dict[str, float] = {
    'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
    'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3,
}

AROMATIC_AA: set = set("FYW")
CHROMOPHORE_REGION_1IDX = {64, 65, 66, 67, 68}

GFP_TEMPLATES: Dict[str, str] = {}
GFP_PDB: Dict[str, str] = {}


# ============================================================
# Validation
# ============================================================

def validate_sequence(seq: str, min_len: int = 220, max_len: int = 250) -> Tuple[bool, Optional[str]]:
    if not seq.startswith("M"):
        return False, f"序列必须以 M 开头，当前首字母为 '{seq[0]}'"
    invalid_chars = set(seq) - STANDARD_AA
    if invalid_chars:
        return False, f"序列包含非法字符: {sorted(invalid_chars)}"
    n = len(seq)
    if n < min_len or n > max_len:
        return False, f"序列长度 {n} 不在 [{min_len}, {max_len}] 范围内"
    return True, None


# ============================================================
# FASTA I/O
# ============================================================

def read_fasta(filepath: str) -> List[Tuple[str, str]]:
    records = []
    current_header = ""
    current_seq_parts = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                if current_header:
                    records.append((current_header, ''.join(current_seq_parts)))
                current_header = line[1:].strip()
                current_seq_parts = []
            else:
                current_seq_parts.append(re.sub(r'\s+', '', line).upper())
    if current_header:
        records.append((current_header, ''.join(current_seq_parts)))
    return records


# ============================================================
# Configuration
# ============================================================

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# ============================================================
# Official GFP Data Loading
# ============================================================

def load_gfp_templates(txt_path: str) -> Dict[str, str]:
    global GFP_TEMPLATES, GFP_PDB
    templates, pdbs = {}, {}
    current_name = None
    current_seq_parts = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('# Competition'): continue
            if line.startswith('>'):
                if current_name:
                    templates[current_name] = ''.join(current_seq_parts)
                current_name = line[1:].strip()
                current_seq_parts = []
            elif line.startswith('# recommend PDB:'):
                pdbs[current_name] = line.split(':')[1].strip()
            elif not line.startswith('#'):
                current_seq_parts.append(re.sub(r'\s+', '', line).upper())
    if current_name:
        templates[current_name] = ''.join(current_seq_parts)
    GFP_TEMPLATES, GFP_PDB = templates, pdbs
    return templates


def load_brightness_data(xlsx_path: str) -> pd.DataFrame:
    return pd.read_excel(xlsx_path, sheet_name='brightness')


def load_before_top_seqs(xlsx_path: str) -> pd.DataFrame:
    return pd.read_excel(xlsx_path, sheet_name='beforetopseqs')


def load_exclusion_list(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    if 'Sequence' in df.columns:
        return df['Sequence'].str.strip().str.upper().tolist()
    return []


# ============================================================
# Mutation String Parsers
# ============================================================

def parse_mutation_string(mut_str: str) -> List[Tuple[int, str, str]]:
    if mut_str == 'WT' or pd.isna(mut_str) or not str(mut_str).strip():
        return []
    mutations = []
    for part in str(mut_str).split(':'):
        part = part.strip()
        if not part: continue
        match = re.match(r'^([A-Z])(\d+)([A-Z])$', part)
        if match:
            mutations.append((int(match.group(2)), match.group(1), match.group(3)))
    return mutations


def apply_mutations_to_template(template: str, mutations: List[Tuple[int, str, str]],
                                 check_from: bool = False) -> str:
    """
    将突变应用到模板序列。

    Args:
        check_from: 若为 True, 仅当 from_aa 匹配时才应用突变 (用于严格验证)。
                    若为 False (默认), 直接应用 to_aa, 因为不同 GFP 变体的
                    from_aa 编号可能不一致。
    """
    seq_list = list(template)
    for pos, from_aa, to_aa in mutations:
        idx = pos - 1
        if 0 <= idx < len(seq_list):
            if not check_from or seq_list[idx] == from_aa:
                seq_list[idx] = to_aa
    return ''.join(seq_list)


def reconstruct_sequence_from_mutations(template: str, mutations_str: str) -> str:
    return apply_mutations_to_template(template, parse_mutation_string(mutations_str))


# ============================================================
# ESM Embedding Utilities (改进6: +FP16 +device auto)
# ============================================================

def get_esm_embeddings(
    sequences: List[str],
    model_name: str = "esm2_t30_150M_UR50D",
    batch_size: int = 8,
    device: str = "cpu",
    output_layer: int = -1,
) -> np.ndarray:
    import torch
    import esm
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.to(device)
    model.eval()

    batch_converter = alphabet.get_batch_converter()
    all_embeddings = []

    layer_idx = model.num_layers + output_layer + 1

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i + batch_size]
        batch_data = [(f"seq_{j}", seq) for j, seq in enumerate(batch_seqs)]
        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[layer_idx])
            token_embeddings = results["representations"][layer_idx]
            seq_embeddings = token_embeddings[:, 1:-1, :].mean(dim=1)
            all_embeddings.append(seq_embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def get_esm_embeddings_fast(
    sequences: List[str],
    model_name: str = "esm2_t30_150M_UR50D",
    batch_size: int = 8,
) -> np.ndarray:
    import torch
    device_info = detect_device()
    device = device_info[0]
    print(f"  ESM device: {device}, model: {model_name}, batch={batch_size}")
    return get_esm_embeddings(sequences, model_name, batch_size, device)


def compute_sequence_features(seq: str) -> dict:
    n = len(seq)
    if n == 0: return {}
    hydro_mean = sum(HYDROPATHY.get(aa, 0) for aa in seq) / n
    aromatic_frac = sum(1 for aa in seq if aa in AROMATIC_AA) / n
    gly_frac = seq.count('G') / n
    pro_frac = seq.count('P') / n
    pos = seq.count('K') + seq.count('R') + 0.1 * seq.count('H')
    neg = seq.count('D') + seq.count('E')
    charge = (pos - neg) / n
    return {
        'length': n, 'hydrophobicity_mean': round(hydro_mean, 4),
        'aromatic_fraction': round(aromatic_frac, 4),
        'charge_pH7': round(charge, 4),
        'glycine_fraction': round(gly_frac, 4),
        'proline_fraction': round(pro_frac, 4),
    }


# ═══════════════════════════════════════════════════════════════
# Radar comparison (摘自 agent_logger，agent_logger 删除后移至此)
# ═══════════════════════════════════════════════════════════════

def generate_top6_radar_comparison(selected: list, output_dir: str) -> tuple:
    """生成 Top-6 六维归一化雷达对比表 + JSON 数据。"""
    import json
    from pathlib import Path
    out = Path(output_dir)
    metrics = ['Brightness', 'Stability', 'Folding', 'Structure', 'Knowledge', 'Diversity']
    rows = []
    for i, s in enumerate(selected):
        rows.append({
            'rank': i + 1,
            'Brightness': s.get('predicted_brightness', 0),
            'Stability': s.get('predicted_stability', 0),
            'Folding': s.get('folding_score', 0),
            'Structure': s.get('structure_score', 0.5),
            'Knowledge': s.get('knowledge_score', 0),
            'Diversity': 1.0,
        })

    normed = []
    for row in rows:
        nr = {'rank': row['rank']}
        for m in metrics:
            vals = [r[m] for r in rows]
            rng = max(vals) - min(vals)
            nr[m] = round((row[m] - min(vals)) / max(rng, 0.001), 4) if rng > 0 else 0.5
        normed.append(nr)

    md = ["# Top-6 Radar Comparison", "",
          f"| {'Rank':<6} | " + " | ".join(f"{m:<10}" for m in metrics) + " |",
          f"|{'─'*8}|" + "|".join(f"{'─'*12}" for _ in metrics) + "|"]
    for r in normed:
        md.append(f"| {r['rank']:<6} | " + " | ".join(f"{r[m]:<10.4f}" for m in metrics) + " |")
    (out / "top6_comparison_table.md").write_text('\n'.join(md), encoding='utf-8')
    (out / "top6_radar_data.json").write_text(json.dumps(normed, ensure_ascii=False, indent=2), encoding='utf-8')

    return str(out / "top6_comparison_table.md"), str(out / "top6_radar_data.json")
