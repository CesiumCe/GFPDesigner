# GFP Designer v4.4

**面向 2026 合成生物学蛋白质设计竞赛的 GFP 变体计算设计管线**
<br>*A computational GFP variant design pipeline for the 2026 Synthetic Biology Protein Design Competition*

---

## 中文

### 项目简介

基于官方 141,572 条 GFP 亮度实验数据 + 5 条参考序列 + 135,415 条排除列表，
结合**三级增益矩阵排序**、双骨架生成、TGP 物理约束、ESMFold 3D 结构终验、
ESM-Predict 跨模板底线验证，设计高亮度高热稳定性的 GFP 蛋白质变体。

### 一键运行

```bash
pip install -r requirements.txt
python setup_models.py          # 下载 ESMFold 模型 (~7.9GB, 仅首次)
python main.py                  # 完整管线 (~25 min)
```

缺少 ESMFold 时管线自动跳过 3D 结构验证（剩余步骤正常运行）。

### 系统要求

- Python 3.10+
- 16GB+ RAM
- CUDA GPU 10GB+ VRAM（可选；CPU 亦可，ESMFold ~120s/seq）
- ~8.1GB 磁盘（ESMFold 7.9GB + ESM-Predict 156MB）

### v4.4 管线架构

```
数据蒸馏(增益矩阵) → 双骨架生成(2000条) → 增益矩阵排序 →
TGP 稳定性(50/50 纯物理) → GA+HC 搜索(pop=300×50代) →
ESM-2 接触图(粗筛) → ESMFold 3D(终验) → ESM-Predict Gate(跨模板) →
Top-6 (sfGFP 优先配额 + 多样性过滤)
```

### 三层评分架构

| 层 | 方法 | 作用 | 验证 ρ |
|----|------|------|--------|
| 模板内 | Gain Matrix 三级回退 | 同模板变体精细排序 | 0.72 (5-fold CV) |
| 模板间 | sfGFP 优先配额 | 跨模板先验 | 0.988 (Frenzel 10) |
| 底线 | ESM-Predict 5-Fold | 排除亮度崩塌序列 | 0.96 (cross-type) |

### 模块说明

| 模块 | 功能 |
|------|------|
| `main.py` | 主入口，9 步管线调度 |
| `setup_models.py` | 从 HuggingFace 下载 ESMFold 模型 |
| `data_distillation_scorer.py` | 三级回退增益矩阵 + AA 偏好 + 突变数命中率 |
| `sequence_generator.py` | 蒸馏引导生成 + 双骨架 + TGP 电荷优化 |
| `ensemble_predictor.py` | XGBoost 集成 (Deep Search 回退) |
| `stability_predictor.py` | 50/50 纯物理 (TGP+折叠, ESM/PLL/ZS 全移除) |
| `folding_analyzer.py` | 上位效应 + BLOSUM 距离 + 突变数软惩罚 |
| `structure_validator.py` | ESM-2 接触图 + Contact Order |
| `esmfold_validator.py` | ESMFold 3D pLDDT/pTM/生色团终验 |
| `knowledge_constraints.py` | 文献规则: 离子对 + 5Å 壳层 + Cys CFPS + N 端 |
| `deep_search.py` | GA+HC Memetic Algorithm (gain matrix 评分) |
| `filter_selector.py` | 硬约束过滤 + sfGFP 优先 + 多样性过滤 |
| `stress_test.py` | 散点图 + MD5 终验 + 压力报告 |
| `csv_writer.py` | 竞赛规范 CSV 输出 |
| `esm_predictor.py` | ESM-2 T6 5-Fold 微调 (跨模板底线验证) |
| `utils.py` | 基础工具 + 雷达对比生成 |
| `property_predictor.py` | RF 回退预测器 (XGBoost 安装失败时) |

### 输出文件

| 文件 | 内容 |
|------|------|
| `output/submission.csv` | 6 条序列，竞赛规范格式 |
| `output/submission_detailed.csv` | 含所有评分明细 |
| `output/esmfold_seq1.pdb` | 序列 #1 的 ESMFold 3D 结构 |
| `output/stress_test_scatter.png` | 稳定性-亮度散点图 |
| `output/stress_test_report.txt` | 压力测试完整报告 |
| `output/top6_comparison_table.md` | 六维归一化雷达对比表 |
| `output/top6_radar_data.json` | 雷达图 JSON 数据 |
| `output/design_rationale.md` | 设计理由文档 (手动: `python -c "from src.utils import write_design_rationale; write_design_rationale('output')"`) |

### 配置

编辑 `config.yaml`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `esmfold.enabled` | true | ESMFold 3D 终验 |
| `scoring.brightness_floor` | 0.60 | B_pct 硬底线 (B_pct ∈ [0.60, 1.00]) |
| `stability.tgp_weight` | 0.50 | TGP 表面超充电权重 |
| `selection.sfGFP_priority` | true | sfGFP 模板优先 |

---

## English

### Overview

A computational pipeline for designing high-brightness, high-thermostability GFP variants
for the 2026 Protein Design Competition. Combines **data distillation** (gain matrix ranking
from 141K experimental brightness measurements), **physics-driven stability scoring** (TGP
surface supercharging + folding heuristics), **ESMFold 3D validation**, and **ESM-Predict
cross-template gating**.

### Quick Start

```bash
pip install -r requirements.txt
python setup_models.py          # Download ESMFold (~7.9GB, first time only)
python main.py                  # Full pipeline (~25 min)
```

The pipeline auto-skips ESMFold 3D validation if the model is not present.

### Requirements

- Python 3.10+
- 16GB+ RAM
- CUDA GPU 10GB+ VRAM (optional; CPU ~120s/seq for ESMFold)
- ~8.1GB disk (ESMFold 7.9GB + ESM-Predict 156MB)

### v4.4 Pipeline

```
Distillation → Dual-template Gen(2000) → Gain Matrix Scoring →
TGP Stability(50/50 physics) → GA+HC Search(pop=300×50gen) →
ESM-2 Contacts → ESMFold 3D → ESM-Predict Gate → Top-6
```

### Three-Tier Scoring

| Tier | Method | Role | Validation |
|------|--------|------|------------|
| Within-template | Gain Matrix (3-level fallback) | Fine-grained variant ranking | Spearman ρ=0.72 |
| Cross-template | sfGFP priority quota | Template-level prior | ρ=0.988 (Frenzel 2018) |
| Safety net | ESM-Predict 5-Fold Gate | Exclude brightness-collapse seqs | ρ=0.96 (cross-type) |

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `esmfold.enabled` | true | ESMFold 3D validation |
| `scoring.brightness_floor` | 0.60 | B_pct hard floor |
| `stability.tgp_weight` | 0.50 | TGP surface charge weight |
| `selection.sfGFP_priority` | true | sfGFP template priority |

### References

| Paper | Journal | Use |
|-------|---------|-----|
| Close et al. (2015) TGP | *Proteins* 83(7) | Surface supercharging (competition official) |
| Lin et al. (2023) ESMFold | *Science* | 3D structure validation |
| Pédelacq et al. (2006) sfGFP | *Nature Biotech* | Ion pair network |
| Hirano et al. (2022) StayGold | *Nature Biotech* | Cl⁻ pocket photostability |
| Frenzel et al. (2018) | *Biotech for Biofuels* 11:8 | sfGFP thermostable variants |
| Fraikin et al. (2025) | *Sci. Adv.* 11 | GFP variant sequence alignment |

## License

MIT — see [LICENSE](LICENSE). Bundled models (ESMFold, ESM-2) are also MIT-licensed by Meta AI.
Competition data files in `data/` are not covered by this license — see [data/README.md](data/README.md).
Full third-party notices: [THIRD_PARTY.md](THIRD_PARTY.md).
