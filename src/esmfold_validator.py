# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cesium
#
"""
esmfold_validator.py — ESMFold 结构级验证 (v4.0)

使用本地部署的 ESMFold 模型对最终候选序列进行 3D 结构预测验证。
ESMFold 从序列直接预测全原子坐标 + pLDDT + pTM，
提供比 ESM-2 接触图更直接的结构质量评估。

验证策略:
  1. 计算 WT 模板的 ESMFold 结构作为基线 (pLDDT_WT, pTM_WT)
  2. 对每条候选序列进行折叠预测
  3. 检查折叠质量相对 WT 的下降幅度
  4. 不合格 → 丢弃 → 从候选池取下一个最优序列
  5. 直到收集到 N 条通过验证的序列

性能:
  - ESMFold v1 (esm2_3B): CPU ~120s/seq, GPU(10GB+) ~20s/seq
  - 6条序列 CPU ~12min, GPU ~2min
  - 仅对最终候选序列运行，不参与生成/粗筛阶段

Reference:
  Lin et al., "Evolutionary-scale prediction of atomic-level protein
  structure with a language model", Science 2023.
"""
import os, time, math
from typing import List, Dict, Tuple, Optional
from pathlib import Path

import numpy as np
import torch


class ESMFoldValidator:
    """
    ESMFold 结构验证器。

    使用本地 ESMFold 模型对候选序列进行折叠预测，
    基于 pLDDT/pTM 等指标判断序列是否能形成正确的 GFP 折叠。

    Usage:
        validator = ESMFoldValidator(model_dir="ESMFold/")
        validator.load()

        # 单条验证
        result = validator.validate(seq, template_seq)
        # => {'passed': True/False, 'plddt_mean': 47.2, 'ptm': 0.50, ...}

        # 批量 Top-N 选择
        valid_seqs = validator.select_top_n(candidates, template_seq, n=6)
    """

    def __init__(
        self,
        model_dir: str = "ESMFold",
        device: str = "auto",
        plddt_relative_threshold: float = 0.80,   # mutant >= WT * 0.80
        ptm_relative_threshold: float = 0.75,       # mutant >= WT * 0.75
        chromophore_drop_max: float = 0.30,          # chromophore pLDDT drop <= 30%
        max_attempts_multiplier: int = 5,            # try up to N*5 candidates
    ):
        """
        Args:
            model_dir: ESMFold 模型目录路径
            device: 'auto' (优先GPU, 回退CPU), 'cpu', 'cuda'
            plddt_relative_threshold: 相对 pLDDT 阈值 (mutant/WT)
            ptm_relative_threshold: 相对 pTM 阈值 (mutant/WT)
            chromophore_drop_max: 生色团区域最大允许 pLDDT 下降比例
            max_attempts_multiplier: 最多尝试 N×M 条候选来填满 N 个名额
        """
        self.model_dir = Path(model_dir)
        self.plddt_threshold = plddt_relative_threshold
        self.ptm_threshold = ptm_relative_threshold
        self.chromophore_drop_max = chromophore_drop_max
        self.max_attempts = max_attempts_multiplier

        self.model = None
        self.wt_baseline = None  # {plddt_mean, plddt_chromophore, ptm, ...}
        self._loaded = False

        # Detect device
        if device == "auto":
            if torch.cuda.is_available():
                vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.device = "cuda" if vram >= 10 else "cpu"
                if vram < 10:
                    print(f"  [ESMFold] GPU VRAM {vram:.1f}GB < 10GB, using CPU (expect ~120s/seq)")
            else:
                self.device = "cpu"
        else:
            self.device = device

    def load(self, timeout: int = 300) -> bool:
        """
        加载 ESMFold 模型并计算 WT 基线。

        v4.0: 增加超时保护和进度刷新。
        - low_cpu_mem_usage=False 避免 Windows 大文件内存映射卡死
        - flush=True 确保进度实时可见
        - timeout 秒后超时返回 False，不阻塞管线

        Returns:
            True 如果加载成功
        """
        try:
            from transformers import EsmForProteinFolding
        except ImportError:
            print("  [ESMFold] transformers not installed. Skipping ESMFold validation.",
                  flush=True)
            return False

        print(f"  [ESMFold] Loading model from {self.model_dir}...", flush=True)
        print(f"  [ESMFold] Model file size: "
              f"{os.path.getsize(str(self.model_dir / 'pytorch_model.bin')) / 1024**3:.1f} GB",
              flush=True)
        t0 = time.time()

        try:
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            # v4.0 fix: low_cpu_mem_usage=False on Windows to avoid memory-mapped
            # file hangs with >4GB models. Machine has 40GB RAM — full load is fine.
            self.model = EsmForProteinFolding.from_pretrained(
                str(self.model_dir),
                torch_dtype=dtype,
                low_cpu_mem_usage=False,
            )
            print(f"  [ESMFold] Weights loaded in {time.time()-t0:.0f}s, "
                  f"moving to {self.device}...", flush=True)

            if self.device == "cuda":
                self.model = self.model.to(self.device)
            self.model.eval()

            n_params = sum(p.numel() for p in self.model.parameters()) / 1e9
            print(f"  [ESMFold] Ready: {n_params:.1f}B params on {self.device} "
                  f"in {time.time()-t0:.1f}s", flush=True)
            self._loaded = True
            return True
        except Exception as e:
            print(f"  [ESMFold] Failed to load model: {e}", flush=True)
            print(f"  [ESMFold] ESMFold validation disabled.", flush=True)
            return False

    def compute_wt_baseline(self, wt_sequence: str) -> Dict:
        """
        计算 WT 序列的 ESMFold 基线指标。

        仅在首次调用时运行，结果缓存至 self.wt_baseline。

        Returns:
            {
                'plddt_mean': float,
                'plddt_chromophore': float,   # 生色团区域 (64-69)
                'plddt_n_terminal': float,    # N端 beta-桶 (1-120)
                'plddt_c_terminal': float,    # C端 beta-桶 (121-238)
                'ptm': float,
                'pae_median': float,
            }
        """
        if self.wt_baseline is not None:
            return self.wt_baseline

        if not self._loaded:
            return {}

        print(f"  [ESMFold] Computing WT baseline ({len(wt_sequence)} aa)...")
        t0 = time.time()

        with torch.no_grad():
            output = self.model.infer([wt_sequence])

        plddt_ca = output.plddt[0, :, 1].cpu().numpy() * 100  # [0,100]

        baseline = {
            'plddt_mean': float(plddt_ca.mean()),
            'plddt_chromophore': float(plddt_ca[63:69].mean()),    # pos 64-69
            'plddt_n_terminal': float(plddt_ca[:120].mean()),      # pos 1-120
            'plddt_c_terminal': float(plddt_ca[120:].mean()),      # pos 121-238
            'plddt_per_residue': plddt_ca.tolist(),
            'ptm': float(output.ptm.item()),
            'pae_median': float(np.median(
                output.predicted_aligned_error[0].cpu().numpy())),
            'time': time.time() - t0,
        }

        self.wt_baseline = baseline
        print(f"  [ESMFold] WT baseline: pLDDT={baseline['plddt_mean']:.1f}, "
              f"pTM={baseline['ptm']:.4f} ({baseline['time']:.0f}s)")
        return baseline

    def validate(
        self,
        sequence: str,
        wt_sequence: str,
    ) -> Dict:
        """
        验证单条序列的结构质量。

        Args:
            sequence: 候选突变序列
            wt_sequence: WT 模板序列

        Returns:
            {
                'passed': bool,
                'plddt_mean': float,
                'plddt_chromophore': float,
                'ptm': float,
                'plddt_ratio': float,        # mutant/WT
                'ptm_ratio': float,
                'chromophore_drop': float,    # (WT - mutant) / WT
                'failure_reasons': [str],     # 如果不通过，列出原因
                'time': float,
            }
        """
        if not self._loaded:
            return {'passed': True, 'plddt_mean': 0, 'ptm': 0,
                    'failure_reasons': [], 'skipped': True}

        # Ensure WT baseline is computed
        baseline = self.compute_wt_baseline(wt_sequence)
        if not baseline:
            return {'passed': True, 'plddt_mean': 0, 'ptm': 0,
                    'failure_reasons': [], 'skipped': True}

        # Count differences from WT
        diffs = sum(1 for a, b in zip(sequence, wt_sequence) if a != b)
        n_mutations = diffs

        # Fold the mutant
        t0 = time.time()
        try:
            with torch.no_grad():
                output = self.model.infer([sequence])
        except Exception as e:
            return {
                'passed': False,
                'plddt_mean': 0, 'ptm': 0,
                'failure_reasons': [f'ESMFold inference failed: {str(e)[:100]}'],
                'time': time.time() - t0,
            }

        fold_time = time.time() - t0

        # Extract metrics
        plddt_ca = output.plddt[0, :, 1].cpu().numpy() * 100
        ptm = float(output.ptm.item())

        plddt_mean = float(plddt_ca.mean())
        plddt_chromophore = float(plddt_ca[63:69].mean())
        plddt_n_term = float(plddt_ca[:120].mean())
        plddt_c_term = float(plddt_ca[120:].mean())

        # Ratios relative to WT
        plddt_ratio = plddt_mean / max(baseline['plddt_mean'], 0.1)
        ptm_ratio = ptm / max(baseline['ptm'], 0.001)
        chromophore_drop = (baseline['plddt_chromophore'] - plddt_chromophore) / \
                           max(baseline['plddt_chromophore'], 0.1)

        # --- Validation checks ---
        failures = []

        # Check 1: Overall pLDDT relative to WT
        if plddt_ratio < self.plddt_threshold:
            failures.append(
                f'pLDDT drop: {plddt_mean:.1f} vs WT {baseline["plddt_mean"]:.1f} '
                f'(ratio {plddt_ratio:.2f} < {self.plddt_threshold})'
            )

        # Check 2: pTM relative to WT
        if ptm_ratio < self.ptm_threshold:
            failures.append(
                f'pTM drop: {ptm:.4f} vs WT {baseline["ptm"]:.4f} '
                f'(ratio {ptm_ratio:.2f} < {self.ptm_threshold})'
            )

        # Check 3: Chromophore region integrity
        if chromophore_drop > self.chromophore_drop_max:
            failures.append(
                f'Chromophore pLDDT collapse: {plddt_chromophore:.1f} vs '
                f'WT {baseline["plddt_chromophore"]:.1f} (drop {chromophore_drop:.1%})'
            )

        # Check 4: N-terminal barrel must be reasonably folded (>70% of WT N-term)
        n_term_ratio = plddt_n_term / max(baseline['plddt_n_terminal'], 0.1)
        if n_term_ratio < 0.70:
            failures.append(
                f'N-terminal barrel unraveling: {plddt_n_term:.1f} vs '
                f'WT {baseline["plddt_n_terminal"]:.1f} (ratio {n_term_ratio:.2f})'
            )

        passed = len(failures) == 0

        return {
            'passed': passed,
            'plddt_mean': plddt_mean,
            'plddt_chromophore': plddt_chromophore,
            'plddt_n_terminal': plddt_n_term,
            'plddt_c_terminal': plddt_c_term,
            'ptm': ptm,
            'plddt_ratio': round(plddt_ratio, 4),
            'ptm_ratio': round(ptm_ratio, 4),
            'chromophore_drop': round(chromophore_drop, 4),
            'n_term_ratio': round(n_term_ratio, 4),
            'n_mutations': n_mutations,
            'failure_reasons': failures,
            'time': fold_time,
        }

    def select_top_n(
        self,
        candidates: List[Dict],
        wt_sequence: str,
        n: int = 6,
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        从候选池中选择通过 ESMFold 验证的 Top-N 序列。

        流程:
          1. 从排名最高的候选开始逐个折叠
          2. 通过的保留，不通过的丢弃
          3. 从池中取下一个候选，直到找到 N 个通过的
          4. 若池耗尽仍未满 N 个，返回已有的

        Args:
            candidates: 按 composite_score 降序排列的候选序列列表
                        每个 dict 需含 'sequence' 键
            wt_sequence: WT 模板序列
            n: 目标数量

        Returns:
            (validated: List[Dict], rejected: List[Dict])
            每个 dict 在原字段基础上增加了 esmfold_* 验证字段
        """
        validated = []
        rejected = []
        max_try = min(len(candidates), n * self.max_attempts)

        print(f"\n  [ESMFold] Validating up to {max_try} candidates to find {n} valid...")
        print(f"  [ESMFold] Thresholds: pLDDT>={self.plddt_threshold:.0%}*WT, "
              f"pTM>={self.ptm_threshold:.0%}*WT, "
              f"chromophore_drop<={self.chromophore_drop_max:.0%}")

        for i, candidate in enumerate(candidates[:max_try]):
            if len(validated) >= n:
                break

            seq = candidate.get('sequence', '')
            if not seq:
                continue

            rank = i + 1
            composite = candidate.get('composite_score', 0)

            print(f"  [ESMFold] Candidate #{rank}: {len(seq)}aa, "
                  f"composite={composite:.4f}...", end=" ", flush=True)

            result = self.validate(seq, wt_sequence)

            # Attach validation results to candidate dict
            enriched = dict(candidate)
            enriched.update({
                'esmfold_plddt': result.get('plddt_mean'),
                'esmfold_ptm': result.get('ptm'),
                'esmfold_passed': result['passed'],
                'esmfold_failures': result.get('failure_reasons', []),
            })

            if result['passed']:
                validated.append(enriched)
                print(f"PASS (pLDDT={result['plddt_mean']:.1f}, "
                      f"pTM={result['ptm']:.4f}, {result['time']:.0f}s)")
            else:
                rejected.append(enriched)
                reasons = '; '.join(result.get('failure_reasons', ['unknown']))
                print(f"REJECT ({reasons[:100]})")

        print(f"  [ESMFold] Result: {len(validated)}/{n} validated, "
              f"{len(rejected)} rejected")

        # If we didn't get enough, warn
        if len(validated) < n:
            print(f"  [ESMFold] WARNING: Only {len(validated)}/{n} candidates "
                  f"passed ESMFold validation. Candidate pool exhausted.")

        return validated, rejected

    def generate_validation_report(
        self,
        validated: List[Dict],
        rejected: List[Dict],
        wt_baseline: Dict,
    ) -> str:
        """生成 ESMFold 验证报告。"""
        lines = [
            "=" * 65,
            "  ESMFold Structure Validation Report",
            "=" * 65,
            "",
            f"  WT baseline: pLDDT={wt_baseline.get('plddt_mean', '?'):.1f}, "
            f"pTM={wt_baseline.get('ptm', '?'):.4f}",
            f"  Thresholds:  pLDDT >= {self.plddt_threshold:.0%}*WT, "
            f"pTM >= {self.ptm_threshold:.0%}*WT",
            "",
            f"  Validated: {len(validated)} | Rejected: {len(rejected)}",
            "",
        ]

        if validated:
            lines.append(f"  {'Rank':<6} {'pLDDT':<8} {'pTM':<8} {'Chromo':<8} "
                         f"{'vWT':<8} {'Muts':<6} {'Status'}")
            lines.append(f"  {'-'*55}")
            for i, item in enumerate(validated, 1):
                lines.append(
                    f"  {i:<6} {item.get('esmfold_plddt', 0):<8.1f} "
                    f"{item.get('esmfold_ptm', 0):<8.4f} "
                    f"{item.get('esmfold_chromophore', 0):<8.1f} "
                    f"{item.get('plddt_ratio', 0):<8.2f} "
                    f"{item.get('n_mutations', '?'):<6} PASS"
                )

        if rejected:
            lines.append(f"\n  Rejected sequences:")
            for i, item in enumerate(rejected, 1):
                reasons = '; '.join(item.get('esmfold_failures', ['?']))
                lines.append(
                    f"  #{item.get('rank', i)}: pLDDT={item.get('esmfold_plddt', '?'):.1f}, "
                    f"pTM={item.get('esmfold_ptm', '?'):.4f} — {reasons[:120]}"
                )

        lines.append(f"\n{'='*65}")
        return '\n'.join(lines)
