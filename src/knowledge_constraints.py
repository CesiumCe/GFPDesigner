# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cesium
#
"""
knowledge_constraints.py — 文献先验知识编码为硬约束 (改进4)

将经典文献的结构生物学机制翻译为代码层面的可执行约束:

1. Superfolder S30R 五元离子对网络 (Pedelacq et al., 2006)
2. TGP 表面超充电策略 (Close et al., 2025)
3. StayGold / mBaoJin 氯离子口袋 (Hirano et al., 2022 / mBaoJin)

这些约束作为"免费午餐"——不增加计算成本，但可保障基础稳定性。
"""
from typing import List, Dict, Tuple, Optional, Set


# ============================================================
# 1. Superfolder 五元离子对网络
# ============================================================

# Key ion pair positions in sfGFP (1-indexed)
# Reference: Pedelacq et al., "Engineering and characterization
# of a superfolder green fluorescent protein", Nature Biotech 2006
SUPERFOLDER_ION_PAIRS = {
    # (pos_i, pos_j, expected_interaction_type)
    # S30R forms salt bridge with E17
    (30, 17, "salt_bridge"),   # R30-E17
    # Y39N interacts with D36
    (39, 36, "hbond"),         # N39-D36
    # F64L stabilizes chromophore environment
    # M153T, V163A — core packing
    (153, 163, "packing"),     # T153-A163 tight packing
    (99, 153, "packing"),      # F99-T153 packing
    (64, 68, "chromophore"),   # L64 near chromophore
}

# Positions to monitor for ion pair integrity
# These should prefer charged/ionic residues
ION_PAIR_POSITIONS = {17, 30, 36, 39, 64, 99, 153, 163}


def check_ion_pair_network(
    seq: str,
    template: str = None,
) -> Dict:
    """
    检查五元离子对网络的完整性。

    对每个关键离子对位置，检查突变是否破坏了原有的
    静电/极性相互作用。

    Returns:
        {
            'ion_pair_score': 离子对完整度 [0, 1],
            'disrupted_pairs': 被破坏的离子对列表,
            'warning_level': 'safe' | 'caution' | 'danger',
        }
    """
    disrupted = []

    for pos_i, pos_j, interaction_type in SUPERFOLDER_ION_PAIRS:
        aa_i = seq[pos_i - 1] if pos_i <= len(seq) else '?'
        aa_j = seq[pos_j - 1] if pos_j <= len(seq) else '?'

        if interaction_type == "salt_bridge":
            # Salt bridge requires positively charged (R/K) and negatively charged (D/E)
            charged_pos = {'R', 'K'}
            charged_neg = {'D', 'E'}
            # Position 30 should be R/K, position 17 should be D/E
            if pos_i == 30 and aa_i not in charged_pos:
                disrupted.append(f"Pos{pos_i}: {aa_i} lost +charge (salt bridge broken)")
            if pos_j == 17 and aa_j not in charged_neg:
                disrupted.append(f"Pos{pos_j}: {aa_j} lost -charge (salt bridge broken)")

        elif interaction_type == "hbond":
            # H-bond: prefer polar residues
            polar = {'N', 'Q', 'S', 'T', 'D', 'E', 'R', 'K', 'H', 'Y'}
            if aa_i not in polar and aa_j not in polar:
                disrupted.append(f"Pos{pos_i}-{pos_j}: polar interaction lost")

        elif interaction_type == "packing":
            # Tight packing: prefer small/medium hydrophobic
            good_packing = {'A', 'V', 'I', 'L', 'M', 'F', 'T', 'S', 'C'}
            if aa_i not in good_packing:
                disrupted.append(f"Pos{pos_i}: {aa_i} may disrupt tight packing")
            if aa_j not in good_packing:
                disrupted.append(f"Pos{pos_j}: {aa_j} may disrupt tight packing")

        elif interaction_type == "chromophore":
            # Near chromophore: keep hydrophobic
            if aa_i not in {'L', 'I', 'V', 'F', 'M', 'A'}:
                disrupted.append(f"Pos{pos_i}: polar {aa_i} near chromophore (risk)")

    n_disrupted = len(disrupted)
    n_total = len(SUPERFOLDER_ION_PAIRS)
    score = max(0.0, 1.0 - n_disrupted / n_total)

    if n_disrupted <= 1:
        warning = 'safe'
    elif n_disrupted <= 3:
        warning = 'caution'
    else:
        warning = 'danger'

    return {
        'ion_pair_score': round(score, 4),
        'disrupted_pairs': disrupted,
        'n_disrupted': n_disrupted,
        'warning_level': warning,
    }


# ============================================================
# 2. TGP 表面超充电策略
# ============================================================

# TGP (thermostable GFP) achieves extreme stability through
# surface supercharging — adding many negatively charged residues
# to the surface prevents heat-induced aggregation.
# Reference: Close et al., "TGP, an extremely stable,
# non-aggregating fluorescent protein", 2025.

# Surface-exposed positions in avGFP where charge-enhancing
# mutations are beneficial (based on TGP paper + SASA analysis)
TGP_SURFACE_NEGATIVE_SITES = {
    3, 4, 5, 6, 8, 10, 14, 15, 18, 19, 22, 23,
    29, 30, 33, 34, 45, 46, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 76, 77, 78, 79, 80, 81, 82,
    101, 102, 103, 104, 105, 106, 107, 108, 109,
    110, 111, 112, 113, 114, 115, 117, 118, 119,
    120, 122, 123, 124, 125, 126, 127, 128, 129,
    130, 131, 132, 133, 134, 135, 136, 137, 138,
    139, 140, 141, 142, 143, 144, 145, 147, 148,
    149, 150, 151, 152, 154, 155, 156, 157, 158,
    159, 160, 161, 162, 164, 165,
}


def compute_negative_surface_charge(seq: str) -> Dict:
    """
    计算表面净负电荷密度 (TGP 策略)。

    统计在已知表面暴露位点上的 D/E 替换情况。
    更高的负电荷密度 → 更好的抗高温聚集能力。

    Returns:
        {
            'surface_neg_charge_count': 表面D/E数量,
            'surface_neg_fraction': 占表面位点的比例,
            'tgp_charge_bonus': TGP策略加分 [0,1],
        }
    """
    neg_charged = {'D', 'E'}
    count = 0
    total_surface = 0

    for pos in TGP_SURFACE_NEGATIVE_SITES:
        if pos <= len(seq):
            total_surface += 1
            if seq[pos - 1] in neg_charged:
                count += 1

    if total_surface == 0:
        return {'surface_neg_charge_count': 0, 'surface_neg_fraction': 0.0,
                'tgp_charge_bonus': 0.0}

    fraction = count / total_surface

    # Bonus: sigmoid mapping, optimal around 25-35% negative surface charge
    # Too much can also be bad (over-solubilization)
    if fraction < 0.10:
        bonus = 0.0
    elif fraction < 0.25:
        bonus = (fraction - 0.10) / 0.15 * 0.5  # 0 -> 0.5
    elif fraction < 0.35:
        bonus = 0.5 + (fraction - 0.25) / 0.10 * 0.5  # 0.5 -> 1.0
    else:
        bonus = 1.0

    return {
        'surface_neg_charge_count': count,
        'surface_neg_fraction': round(fraction, 4),
        'tgp_charge_bonus': round(bonus, 4),
    }


# ============================================================
# 3. StayGold / mBaoJin 氯离子口袋
# ============================================================

# StayGold achieves extreme photostability through a stabilized
# chloride ion binding pocket near the chromophore.
# Key pocket positions (mapped to avGFP numbering via structure alignment):
# Reference: Hirano et al., "A highly photostable and bright green
# fluorescent protein", Nature Biotech 2022.

CHLORIDE_POCKET_POSITIONS = {
    42,   # Hydrophobic cage
    44,   # Hydrophobic cage
    61,   # Near chromophore
    68,   # Chromophore-proximal
    70,   # Pocket wall
    89,   # Coordinating residue
    91,   # Coordinating residue
    116,  # Bottom of pocket
    119,  # Pocket wall
    147,  # Cap residue
    196,  # Lateral wall
    217,  # C-terminal cap
}


def check_chloride_pocket_integrity(seq: str) -> Dict:
    """
    检查氯离子结合口袋的微环境完整性。

    口袋应保持疏水/芳香族特征以稳定氯离子。
    若口袋残基被极性残基替换 → 警告。

    Returns:
        {
            'pocket_score': 口袋完整度 [0,1],
            'polar_intrusions': 极性残基入侵的位置,
        }
    """
    hydrophobic = {'A', 'V', 'I', 'L', 'M', 'F', 'W', 'Y', 'C'}
    aromatic = {'F', 'Y', 'W', 'H'}
    acceptable = hydrophobic | {'G', 'S', 'T', 'P'}  # Small residues OK

    polar_intrusions = []
    for pos in CHLORIDE_POCKET_POSITIONS:
        if pos <= len(seq):
            aa = seq[pos - 1]
            if aa not in acceptable:
                polar_intrusions.append((pos, aa))

    n_bad = len(polar_intrusions)
    n_total = len([p for p in CHLORIDE_POCKET_POSITIONS if p <= len(seq)])

    score = max(0.0, 1.0 - n_bad / max(n_total, 1))

    return {
        'pocket_score': round(score, 4),
        'polar_intrusions': polar_intrusions,
        'n_polar_intrusions': n_bad,
    }


# ============================================================
# 综合知识约束评分
# ============================================================

def evaluate_knowledge_constraints(seq: str, template_name: str = "", template_seq: str = "") -> Dict:
    """
    综合评估三条文献约束，输出知识约束评分。

    Returns:
        {
            'knowledge_score': 综合知识评分 [0, 1],
            'ion_pair': {...},
            'tgp_charge': {...},
            'chloride_pocket': {...},
            'warnings': [描述性警告列表],
            'bonus_flags': [加分项描述],
        }
    """
    # v2.2: Template-conditional activation
    # S30R ion pair network is sfGFP-specific — only check when template is sfGFP
    if template_name == 'sfGFP':
        ion_pair = check_ion_pair_network(seq)
        ion_pair_weight = 0.40
    else:
        # For non-sfGFP templates, use generic chromophore/core integrity check
        ion_pair = _check_core_integrity(seq)
        ion_pair_weight = 0.40

    tgp_charge = compute_negative_surface_charge(seq)
    chloride = check_chloride_pocket_integrity(seq)
    chromo = check_chromophore_microenvironment(seq, template_seq)  # v3.1: template-relative
    n_term = check_n_terminal_protection(seq)                       # v3.0

    # Aggregate (v4.4: proper weight normalization instead of hardcoded /1.2)
    weighted_sum = (
        ion_pair_weight * ion_pair.get('ion_pair_score', ion_pair.get('core_score', 0.5))
        + 0.20 * tgp_charge['tgp_charge_bonus']
        + 0.20 * chloride['pocket_score']
        + 0.20 * chromo['chromophore_env_score']
        + 0.20 * n_term['n_term_score']
    )
    total_weight = ion_pair_weight + 0.20 + 0.20 + 0.20 + 0.20
    knowledge_score = weighted_sum / max(total_weight, 0.01)

    warnings = []
    bonuses = []

    if ion_pair['warning_level'] == 'danger':
        warnings.append(f"CRITICAL: {ion_pair['n_disrupted']} key ion pairs disrupted! "
                       f"Fold stability at risk.")
    elif ion_pair['warning_level'] == 'caution':
        warnings.append(f"CAUTION: {ion_pair['n_disrupted']} ion pairs disrupted.")

    if tgp_charge['tgp_charge_bonus'] < 0.3:
        warnings.append("Low negative surface charge — aggregation risk at high T.")
    elif tgp_charge['tgp_charge_bonus'] > 0.7:
        bonuses.append("Strong negative surface charge (TGP-like) — anti-aggregation.")

    if chloride['n_polar_intrusions'] >= 2:
        warnings.append(f"{chloride['n_polar_intrusions']} polar intrusions in Cl- pocket "
                       f"— photostability risk.")
    elif chloride['pocket_score'] == 1.0:
        bonuses.append("Intact chloride pocket — good photostability potential.")

    # v3.0: Chromophore microenvironment warnings
    if chromo['n_forbidden_intrusions'] >= 2:
        warnings.append(f"{chromo['n_forbidden_intrusions']} forbidden residues (W/R/E/P) "
                       f"near chromophore — cyclization/dehydration risk!")
    elif chromo['n_forbidden_intrusions'] == 1:
        warnings.append(f"1 forbidden residue near chromophore — monitor closely.")

    # v3.2: pH sensitivity warnings (D/H near chromophore → pKa shift → CFPS quenching)
    if chromo['n_pka_risks'] >= 2:
        warnings.append(f"{chromo['n_pka_risks']} pKa-sensitive residues (D/H) introduced "
                       f"near chromophore — pH-dependent quenching risk in CFPS. "
                       f"Penalty: {chromo['pH_sensitivity_penalty']:.2f}")
    elif chromo['n_pka_risks'] == 1:
        warnings.append(f"1 pKa-sensitive residue (D/H) near chromophore — "
                       f"minor pH risk, penalty: {chromo['pH_sensitivity_penalty']:.2f}")

    # v3.0: N-terminal protection warnings
    if n_term['has_degradation_risk']:
        warnings.append(f"N-terminal degradation risk detected: "
                       f"{n_term['risky_residues']} — may reduce CFPS yield.")
    if n_term['n_term_score'] == 1.0:
        bonuses.append("Clean N-terminus — optimal for CFPS expression.")

    return {
        'knowledge_score': round(knowledge_score, 4),
        'ion_pair': ion_pair,
        'tgp_charge': tgp_charge,
        'chloride_pocket': chloride,
        'chromophore_env': chromo,     # v3.0
        'n_term_protection': n_term,   # v3.0
        'warnings': warnings,
        'bonus_flags': bonuses,
    }


def batch_knowledge_constraints(sequences: List[str], template_name: str = "",
                                template_seq: str = "") -> List[Dict]:
    """批量评估知识约束 (v3.1: 发色团约束 template-relative + N端保护)。"""
    return [evaluate_knowledge_constraints(s, template_name, template_seq) for s in sequences]


# ============================================================
# v3.0: Chromophore Microenvironment + N-Terminal Protection
# ============================================================

# Residues forbidden near chromophore (within 5A, positions 64-68 ± 2)
# These can hinder chromophore cyclization/dehydration:
#   W (Trp) — steric bulk blocks cyclization
#   R (Arg) — guanidinium group interferes with脱水
#   E (Glu) — carboxyl competes with chromophore's own chemistry
#   P (Pro) — rigid kink disrupts chromophore geometry
FORBIDDEN_CHROMOPHORE_RESIDUES = {'W', 'R', 'E', 'P'}

# v3.2: pKa-sensitive residues near chromophore (CFPS pH drift risk)
#   D (Asp) — side-chain pKa ~3.9, protonation shifts near chromophore alter local electrostatics
#   H (His) — side-chain pKa ~6.0, exactly at physiological pH; protonation state flips with minor pH drift
# These don't block cyclization like W/R/E/P, but cause pH-dependent fluorescence quenching
PKA_SENSITIVE_RESIDUES = {'D', 'H'}
PH_SENSITIVITY_PENALTY = 0.85  # Soft penalty: reduces score by 15%

# v4.0: Cysteine in CFPS — universal penalty
# CFPS (NEBexpress Cell-free E. coli) is a strongly REDUCING environment.
# Cys residues CANNOT form correct disulfide bonds; instead they cause
# aberrant cross-linking → aggregation. Every Cys introduction is a liability.
# Reference: CFPS cytoplasm is reducing (glutathione/thioredoxin systems active).
# This replaces the v3.3 "pair-only" disulfide risk — ALL Cys are penalized.
CYS_CFPS_PENALTY = 0.90  # Each introduced Cys reduces score by 10%

# Chromophore-proximal positions (avGFP numbering, 5A zone):
# 65-67 IS the chromophore triad (SYG) — exclude from forbidden check
# Check only surrounding 5A shell positions
CHROMOPHORE_5A_ZONE = {62, 63, 64, 68, 69, 70, 93, 94, 95, 96,
                        144, 145, 146, 147, 148, 201, 202, 203, 204, 205, 221, 222, 223}

# v3.3: Spatial adjacency for disulfide risk detection within chromophore zone
# Subset of SPATIAL_NEIGHBORS_8A restricted to chromophore 5A zone positions
CHROMOPHORE_ZONE_ADJACENCY = {
    62: {63, 64, 65, 66, 222, 223, 225, 226},
    63: {62, 64, 65, 66, 108},
    64: {62, 63, 65, 66, 67},
    68: {65, 66, 67, 69, 70, 71, 112, 121},
    69: {65, 66, 67, 68, 70, 71, 72, 73, 225},
    70: {67, 68, 69, 71, 72, 73, 74},
    93: {91, 92, 94, 95, 109, 110, 111, 112},
    94: {67, 92, 93, 95, 96, 109, 110, 111},
    95: {93, 94, 96, 97, 107, 108, 109, 110},
    96: {94, 95, 97, 98, 106, 107, 108, 109},
    144: {142, 143, 145, 146, 147, 149, 150, 151},
    145: {143, 144, 146, 147, 148, 149, 150},
    146: {144, 145, 147, 148, 149},
    147: {144, 145, 146, 148, 149, 150},
    148: {145, 146, 147, 149, 150, 168, 169, 170, 171},
    201: {199, 200, 202, 203, 204, 228},
    202: {199, 200, 201, 203, 204, 205},
    203: {73, 201, 202, 204, 205, 221, 224, 225},
    204: {201, 202, 203, 205, 206, 207},
    205: {202, 203, 204, 206, 207, 208, 217, 221},
    221: {203, 205, 217, 218, 219, 220, 222, 223, 224, 225},
    222: {62, 218, 219, 220, 221, 223, 224, 225, 226},
    223: {62, 219, 220, 221, 222, 224, 225, 226, 227},
}

# N-terminal degradation-prone residues (N-end rule + CFPS-specific):
#   F, Y, W, L, I — hydrophobic N-term → recognition by ClpXP/Lon proteases
#   R, K — charged N-term → ClpS recognition in CFPS
DEGRADATION_PRONE_NTERM = {'F', 'Y', 'W', 'L', 'I', 'R', 'K'}


def check_chromophore_microenvironment(seq: str, template: str = None) -> Dict:
    """
    v3.1: 发色团微环境硬约束 (template-relative)。

    关键修复: WT 自带的残基 (如 avGFP 的 E95, R96, E222) 是"安全基线"，
    不应被标记为违规。只惩罚"从非 W/R/E/P 突变成 W/R/E/P"的位点。

    禁止引入 W, R, E, P 到生色团 5Å 壳层:
      - W: 空间位阻阻断环化
      - R: 胍基干扰脱水
      - E: 羧基竞争生色团化学
      - P: 刚性折角破坏生色团几何

    Returns:
        {
            'chromophore_env_score': [0,1],
            'forbidden_intrusions': [(pos, from_aa, to_aa), ...],
            'n_forbidden_intrusions': int,
            'pka_risk_positions': [(pos, aa), ...],    # v3.2: D/H near chromophore
            'n_pka_risks': int,
            'pH_sensitivity_penalty': float,            # v3.2: soft penalty factor
            'wt_residues_present': [(pos, aa), ...],
        }
    """
    intrusions = []
    wt_residues = []
    pka_risks = []

    for pos in CHROMOPHORE_5A_ZONE:
        if pos <= len(seq):
            aa = seq[pos - 1]

            # Check forbidden (W/R/E/P)
            if aa in FORBIDDEN_CHROMOPHORE_RESIDUES:
                if template is not None and pos <= len(template):
                    template_aa = template[pos - 1]
                    if template_aa in FORBIDDEN_CHROMOPHORE_RESIDUES:
                        wt_residues.append((pos, aa))
                    else:
                        intrusions.append((pos, template_aa, aa))
                else:
                    intrusions.append((pos, '?', aa))

            # v3.2: Check pKa-sensitive (D/H) — separate from forbidden
            if aa in PKA_SENSITIVE_RESIDUES:
                if template is not None and pos <= len(template):
                    template_aa = template[pos - 1]
                    if template_aa not in PKA_SENSITIVE_RESIDUES:
                        # New mutation TO D or H → pKa risk
                        pka_risks.append((pos, template_aa, aa))
                    # else: WT already had D/H → baseline, no penalty
                else:
                    pka_risks.append((pos, '?', aa))

    n_bad = len(intrusions)
    n_pka = len(pka_risks)
    score = max(0.0, 1.0 - n_bad * 0.25)  # Forbidden residue penalty

    # v3.2: pH sensitivity soft penalty
    pH_penalty = PH_SENSITIVITY_PENALTY ** n_pka if n_pka > 0 else 1.0
    score *= pH_penalty

    # v4.0: Cysteine universal penalty for CFPS reducing environment
    # ALL newly introduced Cys are penalized (not just pairs).
    # CFPS is a reducing environment → Cys cannot form correct disulfide bonds,
    # instead causing aberrant cross-linking and aggregation.
    cys_introductions = [
        pos for pos in CHROMOPHORE_5A_ZONE
        if pos <= len(seq) and seq[pos - 1] == 'C'
        and (template is None or pos > len(template) or template[pos - 1] != 'C')
    ]
    n_cys = len(cys_introductions)
    cys_penalty = CYS_CFPS_PENALTY ** n_cys if n_cys > 0 else 1.0
    score *= cys_penalty

    return {
        'chromophore_env_score': round(score, 4),
        'forbidden_intrusions': intrusions,
        'n_forbidden_intrusions': n_bad,
        'pka_risk_positions': pka_risks,
        'n_pka_risks': n_pka,
        'pH_sensitivity_penalty': round(pH_penalty, 4),
        'wt_residues_present': wt_residues,
        'cys_introductions': cys_introductions,       # v4.0: universal Cys penalty
        'n_cys_introductions': n_cys,
        'cys_penalty': round(cys_penalty, 4),
    }


def check_n_terminal_protection(seq: str) -> Dict:
    """
    v3.0: N端保护检查。

    确保 N 端前 3 位不包含易降解残基。
    CFPS 体系中，N端疏水残基会被 ClpXP/Lon 识别降解，
    带正电的 N 端会被 ClpS 识别。

    Returns:
        {
            'n_term_score': [0,1],
            'risky_residues': [(pos, aa), ...],
            'has_degradation_risk': bool,
        }
    """
    risky = []
    for i in range(min(3, len(seq))):
        aa = seq[i]
        if aa in DEGRADATION_PRONE_NTERM:
            risky.append((i + 1, aa))

    has_risk = len(risky) > 0
    score = max(0.0, 1.0 - len(risky) * 0.33)

    return {
        'n_term_score': round(score, 4),
        'risky_residues': risky,
        'has_degradation_risk': has_risk,
    }


def _check_core_integrity(seq: str) -> Dict:
    """
    v2.2: 通用核心完整性检查 (用于非sfGFP模板)。

    检查生色团保守性和核心疏水堆积，而非特定的S30R离子对。
    """
    # Chromophore-proximal positions (aligned to avGFP numbering)
    chromophore_near = {64, 65, 66, 67, 68, 94, 95, 96, 145, 146, 147, 202, 203, 204}
    hydrophobic = set('ACFILMPVW')

    core_violations = []
    for pos in chromophore_near:
        if pos <= len(seq):
            aa = seq[pos - 1]
            if aa not in hydrophobic and aa not in {'G', 'S', 'T'}:
                core_violations.append(f"Pos{pos}: polar {aa} near chromophore core")

    n_total = len([p for p in chromophore_near if p <= len(seq)])
    n_bad = len(core_violations)
    score = max(0.0, 1.0 - n_bad / max(n_total, 1))

    warning = 'safe' if n_bad <= 1 else ('caution' if n_bad <= 2 else 'danger')

    return {
        'core_score': round(score, 4),
        'disrupted_pairs': core_violations,
        'n_disrupted': n_bad,
        'warning_level': warning,
        'ion_pair_score': round(score, 4),  # Compat key
    }
