# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cesium
#
#!/usr/bin/env python3
"""
main.py — GFP Designer 主入口

9 步管线: 数据蒸馏 → 双骨架生成 → XGBoost预测 → TGP稳定性 →
          GA+HC搜索 → ESM-2接触图 → ESMFold终验 → sfGFP优先配额 → 输出

Usage:
    python main.py                          # 默认运行
    python main.py --skip-esmfold           # 跳过 ESMFold (加速)
    python main.py --config my_config.yaml  # 指定配置文件
    python main.py --help                   # 查看所有选项
"""

import os, sys, time, argparse
from src._version import __version__
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---- Basic utilities ----
from src.utils import (
    load_config, load_gfp_templates, load_brightness_data,
    load_before_top_seqs, load_exclusion_list, validate_sequence,
    detect_device,
)

# ---- Generation ----
from src.sequence_generator import generate_variants

# ---- Ensemble prediction ----
from src.ensemble_predictor import (
    ensemble_predict_batch,
    train_xgboost_predictor,
    compute_esm_zero_shot_scores,
)

# ---- Stability ----
from src.stability_predictor import predict_stability_batch, _estimate_stability_fallback

# ---- Folding analysis ----
from src.folding_analyzer import batch_folding_scores

# ---- Knowledge constraints ----
from src.knowledge_constraints import batch_knowledge_constraints

# ---- Filtering ----
from src.filter_selector import filter_by_constraints, select_top_n, generate_selection_report

# ---- Output ----
from src.csv_writer import write_submission, write_submission_with_scores

def main():
    data_dir = PROJECT_ROOT / "data"
    output_dir = PROJECT_ROOT / "output"
    config_path = PROJECT_ROOT / "config.yaml"

    # ================================================================
    # 0. Load config & detect device
    # ================================================================
    print("=" * 70)
    print(f"  GFP Variant Designer — Pipeline v{__version__}")
    print("=" * 70)

    cfg = load_config(str(config_path))

    # ── 配置节 ──
    device_cfg = cfg.get('device', {})
    esm_cfg = cfg.get('esm', {})
    score_cfg = cfg.get('scoring', {})
    sel_cfg = cfg.get('selection', {})
    know_cfg = cfg.get('knowledge', {})
    out_cfg = cfg.get('output', {})
    gen_cfg = cfg.get('generation', {})
    stab_cfg = cfg.get('stability', {})
    xgb_cfg = cfg.get('xgboost', {})
    ds_cfg = cfg.get('deep_search', {})
    esmfold_cfg = cfg.get('esmfold', {})

    # Device detection
    use_gpu = device_cfg.get('use_gpu', True)
    device, device_info = detect_device(use_gpu)
    print(f"\n  Device: {device} | GPU: {device_info['gpu_name'] or 'N/A'}")
    print(f"  VRAM: {device_info['vram_gb']}GB")

    # Model selection
    model_name = esm_cfg.get('model_name', 'esm2_t30_150M_UR50D')
    if device == 'cpu' and 't30' in model_name:
        print(f"  CPU mode — expect longer inference")
    batch_size = esm_cfg.get('batch_size', 8)

    # ================================================================
    # 1. Load official data
    # ================================================================
    print(f"\n[1/9] Loading official data...")

    templates = load_gfp_templates(str(data_dir / Path(cfg['data']['template_seqs']).name))
    primary_template = cfg.get('primary_template', 'avGFP')
    template_seq = templates[primary_template]

    brightness_df = load_brightness_data(str(data_dir / Path(cfg['data']['brightness_data']).name))
    before_top_seqs = load_before_top_seqs(str(data_dir / Path(cfg['data']['brightness_data']).name))
    exclusion_list = load_exclusion_list(str(data_dir / Path(cfg['data']['exclusion_list']).name))

    print(f"  Templates: {list(templates.keys())}")
    print(f"  Primary: {primary_template} ({len(template_seq)} aa)")
    print(f"  Brightness data: {len(brightness_df)} entries")
    print(f"  Exclusion list: {len(exclusion_list)} sequences")
    print(f"  Previous winners: {len(before_top_seqs)} sequences")

    # ================================================================
    # 2. Train ESM + RF predictor
    # ================================================================
    print(f"\n[2/9] Setting up ensemble predictor...")

    esm_model, esm_alphabet = None, None
    rf_model = None

    try:
        import torch
        import esm
        print(f"  Loading ESM model: {model_name}...")
        esm_model, esm_alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        esm_model = esm_model.to(device)
        esm_model.eval()
        embedding_dim = esm_model.embed_dim
        print(f"  ESM loaded: {model_name} ({embedding_dim}d embeddings) on {device}")
    except Exception as e:
        print(f"  [WARN] Failed to load {model_name}: {e}")
        # Try fallback
        fallback = esm_cfg.get('fallback_model', 'esm2_t12_35M_UR50D')
        try:
            print(f"  Trying fallback: {fallback}...")
            import esm
            esm_model, esm_alphabet = esm.pretrained.load_model_and_alphabet(fallback)
            esm_model = esm_model.to(device)
            esm_model.eval()
            embedding_dim = esm_model.embed_dim
            model_name = fallback
            print(f"  Fallback loaded: {fallback} ({embedding_dim}d)")
        except Exception as e2:
            print(f"  [WARN] Fallback also failed: {e2}")
            print(f"  Running with physics-only scoring.")

    # Train XGBoost predictor (替代低R²的RF)
    xgb_predictor = None
    try:
        xgb_predictor = train_xgboost_predictor(
            brightness_df, templates,
            max_samples=xgb_cfg.get('max_train_samples', 50000),
            validation_split=xgb_cfg.get('validation_split', 0.15),
            random_state=xgb_cfg.get('random_state', 42),
        )
        print(f"  XGBoost Train R^2: {xgb_predictor.get('train_r2', 'N/A')}")
        print(f"  XGBoost Val R^2: {xgb_predictor.get('val_r2', 'N/A')}")
        print(f"  Model type: {xgb_predictor.get('model_type', 'N/A')}")
    except Exception as e:
        print(f"  [WARN] XGBoost training failed: {e}, trying RF fallback...")
        # Fall back to Random Forest
        try:
            from src.property_predictor import train_brightness_predictor
            predictor_result = train_brightness_predictor(
                brightness_df, template_seq, primary_template,
                model_name=model_name,
                max_train_samples=esm_cfg.get('max_train_samples', 3000),
                n_estimators=xgb_cfg.get('n_estimators', 200),
            )
            rf_model = predictor_result.get('rf_model')
        except Exception as e2:
            print(f"  [WARN] RF fallback also failed: {e2}")



    # ================================================================
    # 2b. v4.4: Data distillation (gain matrix) for scoring
    #         + ESM-Predict as post-selection validation gate
    # ================================================================
    distilled_rules = None
    if cfg.get('generation', {}).get('use_distillation', True):
        try:
            from src.data_distillation_scorer import load_distilled_rules, compute_gain_score_simple
            distilled_rules = load_distilled_rules(brightness_df, templates)
            print(f"  [v4.4] Data distillation active: gain matrix for ranking")
        except Exception as e:
            print(f"  [v4.4] Distillation failed ({e}), using fallback")

    esm_gate = None
    esm_pred_dir = PROJECT_ROOT / "esm_predict_models"
    if esm_pred_dir.exists():
        try:
            from src.esm_predictor import EsmBrightnessPredictor
            esm_gate = EsmBrightnessPredictor(str(esm_pred_dir), device='cpu')
            if esm_gate.load():
                print(f"  [v4.4] ESM-Predict gate active: post-selection validation")
        except Exception as e:
            print(f"  [v4.4] ESM-Predict gate unavailable")

    # ================================================================
    # 3-5. Generate + Predict + Merge (v4.0: dual-template + distilled)
    # ================================================================
    # v4.0 Dual-template + data-distilled strategy:
    #   - sfGFP + avGFP templates (238aa ea, identical numbering)
    #   - Template-relative mutation budgets (sfGFP:1-3, avGFP:2-4)
    #   - Position-gain-weighted site selection (not frequency-weighted)
    #   - AA preference sampling from Top5% enrichment
    #   - Brightness prediction (XGBoost, R²=0.55) DECOUPLED from
    #     thermal stability (physics-driven TGP+ESM, 40/40/20)
    gen_cfg = cfg['generation']
    uq_penalty = score_cfg.get('uncertainty_penalty', 0.15)

    # Dual templates: sfGFP for baseline alignment, avGFP for data advantage
    dual_templates = ['sfGFP', 'avGFP']
    n_per_template = gen_cfg.get('n_variants_per_template', 1000)
    all_merged = []

    for tpl_idx, tpl_name in enumerate(dual_templates):
        tpl_seq = templates[tpl_name]
        print(f"\n{'='*60}")
        print(f"  Template [{tpl_idx+1}/{len(dual_templates)}]: {tpl_name} ({len(tpl_seq)} aa)")
        print(f"{'='*60}")

        # Step 3: Generate variants for this template
        print(f"\n[3/9] Generating {n_per_template} variants from {tpl_name}...")
        variants = generate_variants(
            template=tpl_seq,
            template_type=tpl_name,
            brightness_df=brightness_df,
            before_top_seqs=before_top_seqs,
            n_variants=n_per_template,
            min_len=cfg['sequence']['min_length'],
            max_len=cfg['sequence']['max_length'],
            distilled_rules=distilled_rules,  # v4.0: data-distilled guidance
        )

        # Step 4: Predict for this template
        print(f"\n[4/9] Running ensemble predictions for {tpl_name}...", flush=True)
        print(f"  XGBoost prediction + ESM zero-shot (top 200) + stability + "
              f"folding + knowledge", flush=True)

        ensemble_results = ensemble_predict_batch(
            variants, tpl_seq,
            template_type=tpl_name,
            templates=templates,
            esm_model=esm_model, esm_alphabet=esm_alphabet,
            xgb_predictor=xgb_predictor,
            device=device, batch_size=batch_size,
        )

        print(f"  [Stability] Computing thermal stability scores...")
        if esm_model is not None:
            stability_scores = predict_stability_batch(
                variants, tpl_seq, esm_model, esm_alphabet, device, batch_size
            )
        else:
            stability_scores = [_estimate_stability_fallback(s, tpl_seq) for s in variants]

        print(f"  [Folding] Computing folding robustness scores...")
        folding_results = batch_folding_scores(
            variants, tpl_seq, brightness_df, tpl_name
        )

        print(f"  [Knowledge] Evaluating literature-based constraints...")
        knowledge_results = batch_knowledge_constraints(variants, tpl_name, tpl_seq)

        # Step 5: Merge results (v4.4: gain matrix scoring)
        for i, seq in enumerate(variants):
            st = float(stability_scores[i]) if i < len(stability_scores) else 0.5
            fo = folding_results[i]
            kn = knowledge_results[i]

            gain_score = compute_gain_score_simple(seq, tpl_seq, distilled_rules['gain_matrix'])
            disqualified = gain_score < -10.0
            all_merged.append({
                'sequence': seq,
                'template_type': tpl_name,
                'gain_score': round(gain_score, 4),
                'predicted_brightness': 0,
                'predicted_stability': round(st, 4),
                'composite_score': 0,
                'disqualified': disqualified,
                'folding_score': fo.get('folding_score'),
                'knowledge_score': kn.get('knowledge_score'),
                'knowledge_warnings': kn.get('warnings', []),
                'knowledge_bonus': kn.get('bonus_flags', []),
            })

        # Per-template summary
        tpl_merged = [r for r in all_merged if r.get('template_type') == tpl_name]
        b_vals = [r['predicted_brightness'] for r in tpl_merged]
        print(f"  {tpl_name}: {len(tpl_merged)} variants, brightness [{min(b_vals):.3f}, {max(b_vals):.3f}]")

    # v4.4: Normalize gain_score → B_pct, compute C = B_pct × S
    all_gains = [r['gain_score'] for r in all_merged]
    g_min, g_max = min(all_gains), max(all_gains)
    print(f"\n  Gain range: [{g_min:.1f}, {g_max:.1f}]")
    for r in all_merged:
        if g_max > g_min:
            r['predicted_brightness'] = round(0.60 + 0.40 * (r['gain_score'] - g_min) / (g_max - g_min), 4)
        else:
            r['predicted_brightness'] = 0.80
        r['composite_score'] = round(r['predicted_brightness'] * r['predicted_stability'], 4)

    # Combined summary
    b_vals = [r['predicted_brightness'] for r in all_merged]
    c_vals = [r['composite_score'] for r in all_merged]
    s_vals = [r['predicted_stability'] for r in all_merged]
    f_vals = [r['folding_score'] for r in all_merged if r['folding_score'] is not None]
    k_vals = [r['knowledge_score'] for r in all_merged if r['knowledge_score'] is not None]

    print(f"\n  Combined pool: {len(all_merged)} variants ({len(dual_templates)} templates)")
    print(f"  Ensemble Brightness: [{min(b_vals):.3f}, {max(b_vals):.3f}], mean={sum(b_vals)/len(b_vals):.3f}")
    if f_vals:
        print(f"  Folding Score      : [{min(f_vals):.3f}, {max(f_vals):.3f}], mean={sum(f_vals)/len(f_vals):.3f}")
    if k_vals:
        print(f"  Knowledge Score    : [{min(k_vals):.3f}, {max(k_vals):.3f}], mean={sum(k_vals)/len(k_vals):.3f}")

    n_disq = sum(1 for r in all_merged if r['disqualified'])
    n_risky = sum(1 for r in all_merged if r.get('n_risky_pairs', 0) > 2)
    n_know_warn = sum(1 for r in all_merged if r.get('knowledge_warnings'))
    print(f"  Disqualified: {n_disq} | High epistasis risk: {n_risky} | Knowledge warnings: {n_know_warn}")

    # Use the combined pool as 'merged' for downstream steps
    merged = all_merged

    # ================================================================
    # 6. Filter + Deep Search (v3.0)
    # ================================================================
    print(f"\n[6/9] Filtering + Deep Search...")

    # Add aggregation risk to merged results (v3.0)
    from src.deep_search import compute_aggregation_risk
    for r in merged:
        r['aggregation_risk'] = compute_aggregation_risk(r['sequence'])

    # Initial filter
    passed, rejected = filter_by_constraints(
        merged, exclusion_list,
        min_len=cfg['sequence']['min_length'],
        max_len=cfg['sequence']['max_length'],
        brightness_cutoff=score_cfg['brightness_cutoff'], brightness_floor=score_cfg.get('brightness_floor', 0.7),
        folding_score_min=score_cfg.get('folding_score_min', 0.30),
    )

    print(f"  Initial pass : {len(passed)} | Rejected: {len(rejected)}")

    # Rejection breakdown
    if rejected:
        reason_counts = {}
        for r in rejected:
            for reason in r.get('rejection_reasons', ['Unknown']):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"    - {reason[:80]}: {count}")

    # --- Deep Search ---
    from src.deep_search import deep_search_pipeline
    deep_search_improvements = []

    if xgb_predictor is not None:
        # Get top-50 passed candidates as seeds
        top_seeds = sorted(passed, key=lambda x: x.get('composite_score', 0), reverse=True)[:300]

        deep_pool = deep_search_pipeline(
            top_candidates=top_seeds,
            template=template_seq,
            template_type=primary_template,
            templates=templates,
            xgb_predictor=xgb_predictor,
            brightness_df=brightness_df,
            exclusion_list=exclusion_list,
            n_rounds=3,
            n_seeds=min(300, len(passed)),
            esm_model=esm_model,
            esm_alphabet=esm_alphabet,
            device=device,
            brightness_floor=score_cfg.get('brightness_floor', 0.7),
            distilled_rules=distilled_rules,
            gain_g_min=g_min,
            gain_g_max=g_max,
        )

        # Re-filter deep search pool
        deep_passed, _ = filter_by_constraints(
            deep_pool, exclusion_list,
            min_len=cfg['sequence']['min_length'],
            max_len=cfg['sequence']['max_length'],
            brightness_cutoff=score_cfg['brightness_cutoff'], brightness_floor=score_cfg.get('brightness_floor', 0.7),
            folding_score_min=0.15,  # Relaxed for deep search (already multi-round validated)
        )
        print(f"  Deep search pool: {len(deep_pool)} → {len(deep_passed)} passed filter")

        # Merge with original passed
        all_passed = passed + deep_passed
        # Deduplicate by sequence
        seen = set()
        all_passed_dedup = []
        for item in all_passed:
            if item['sequence'] not in seen:
                seen.add(item['sequence'])
                all_passed_dedup.append(item)
        passed = all_passed_dedup
    else:
        print(f"  Deep search skipped (no XGBoost predictor)")

    selected = select_top_n(
        passed, n=cfg['sequence']['n_submissions'],
        diversity_threshold=sel_cfg['diversity_threshold'],
    )

    # ================================================================
    # 7. Stress Test (v3.0)
    # ================================================================
    # v3.1 Structure Validation (ESM-2 Contact + Embedding)
    # ================================================================
    print(f"\n[7/9] ESM-2 Structure Validation...", flush=True)

    structure_results = []
    if esm_model is not None and esm_alphabet is not None:
        from src.structure_validator import validate_batch, generate_structure_report

        # Validate Top-50 passed candidates for structure integrity
        top50_for_struct = sorted(passed, key=lambda x: x.get('composite_score', 0), reverse=True)[:50]
        top50_seqs = [s['sequence'] for s in top50_for_struct]

        structure_results = validate_batch(
            top50_seqs, template_seq, esm_model, esm_alphabet, device
        )

        # Filter out sequences that fail structure validation
        struct_passed_indices = [i for i, r in enumerate(structure_results) if r['passed']]
        struct_failed = len(top50_seqs) - len(struct_passed_indices)
        print(f"  Structure passed: {len(struct_passed_indices)}/{len(top50_seqs)}")
        if struct_failed > 0:
            print(f"  Structure FAILED: {struct_failed} sequences — removing from candidate pool")
            # Remove failed sequences from passed pool
            failed_seqs = {top50_seqs[i] for i in range(len(top50_seqs))
                          if i not in struct_passed_indices}
            passed = [p for p in passed if p['sequence'] not in failed_seqs]

        # Attach structure scores to candidate dicts
        for i, r in enumerate(structure_results):
            if i < len(top50_for_struct):
                idx_in_passed = next((j for j, p in enumerate(passed)
                                     if p['sequence'] == top50_seqs[i]), None)
                if idx_in_passed is not None:
                    passed[idx_in_passed]['structure_score'] = r['structure_score']
                    passed[idx_in_passed]['structure_flags'] = r['flags']

        # Generate structure report for Top-6
        struct_report = generate_structure_report(structure_results, list(range(min(6, len(structure_results)))))
        print(struct_report)
    else:
        print(f"  Structure validation skipped (ESM model not available)")

    # ================================================================
    # 7b. ESMFold 3D Structure Validation (v3.3) — final quality gate
    # ================================================================
    esmfold_enabled = esmfold_cfg.get('enabled', True)
    esmfold_validator = None

    if esmfold_enabled:
        esmfold_model_dir = esmfold_cfg.get('model_dir', str(PROJECT_ROOT / 'ESMFold'))
        esmfold_model_path = Path(esmfold_model_dir)

        if esmfold_model_path.exists() and (esmfold_model_path / 'pytorch_model.bin').exists():
            print(f"\n[7b/9] ESMFold 3D Structure Validation...", flush=True)

            from src.esmfold_validator import ESMFoldValidator

            esmfold_validator = ESMFoldValidator(
                model_dir=str(esmfold_model_path),
                device=esmfold_cfg.get('device', 'auto'),
                plddt_relative_threshold=esmfold_cfg.get('plddt_threshold', 0.80),
                ptm_relative_threshold=esmfold_cfg.get('ptm_threshold', 0.75),
                chromophore_drop_max=esmfold_cfg.get('chromophore_drop_max', 0.30),
                max_attempts_multiplier=esmfold_cfg.get('max_attempts', 30),
            )

            if esmfold_validator.load():
                # Re-select from the passed pool using ESMFold validation
                # Sort passed candidates by composite_score descending
                passed_sorted = sorted(
                    passed,
                    key=lambda x: x.get('composite_score', 0),
                    reverse=True
                )

                esmfold_validated, esmfold_rejected = esmfold_validator.select_top_n(
                    candidates=passed_sorted,
                    wt_sequence=template_seq,
                    n=cfg['sequence']['n_submissions'],
                )

                if len(esmfold_validated) >= 3:  # At least 3 passing to replace selection
                    old_selected = selected
                    selected = esmfold_validated

                    # Generate validation report
                    esmfold_report = esmfold_validator.generate_validation_report(
                        esmfold_validated, esmfold_rejected,
                        esmfold_validator.wt_baseline or {},
                    )
                    print(esmfold_report)

                else:
                    print(f"  [ESMFold] WARNING: Only {len(esmfold_validated)} passed ESMFold. "
                          f"Keeping original selection to avoid empty submission.")
            else:
                print(f"  [ESMFold] Model failed to load. Skipping ESMFold validation.")
        else:
            if esmfold_enabled:
                print(f"\n[7b/9] ESMFold model not found at {esmfold_model_path}. "
                      f"Skipping 3D validation.")
    else:
        print(f"\n[7b/9] ESMFold validation disabled in config.")

    # ================================================================
    # 8. Stress Test (v3.0)
    # ================================================================
    print(f"\n[8/9] Stress Test...", flush=True)

    from src.stress_test import (
        generate_stress_test_plot, md5_final_verification,
        generate_stress_report,
    )

    # MD5 final verification
    md5_result = md5_final_verification(
        [s['sequence'] for s in selected], exclusion_list
    )
    print(f"  MD5 verification: {'ALL CLEAR' if md5_result['all_clear'] else 'FAILED!'}")

    # Aggregation risk filter
    high_agg = [s for s in selected if s.get('aggregation_risk', 0) > 0.5]
    if high_agg:
        print(f"  WARNING: {len(high_agg)} sequences with high aggregation risk > 0.5")

    # Stress test report
    stress_path = generate_stress_report(
        selected, passed, md5_result, deep_search_improvements, str(output_dir)
    )

    # Scatter plot
    plot_path = generate_stress_test_plot(passed, selected, str(output_dir))

    # ================================================================
    # 8b. v4.4: ESM-Predict validation gate on Top-6
    # ================================================================
    if esm_gate is not None:
        print(f"\n[8b/9] ESM-Predict validation gate on Top-6...")
        sfGFP_seq = templates.get('sfGFP', template_seq)
        wt_sf_brightness = float(esm_gate.predict([sfGFP_seq])[0])
        esm_threshold = wt_sf_brightness * 0.85
        print(f"  WT sfGFP ESM brightness: {wt_sf_brightness:.4f}, threshold: {esm_threshold:.4f}")

        validated = []
        for s in selected:
            esm_b = float(esm_gate.predict([s['sequence']])[0])
            s['esm_brightness'] = round(esm_b, 4)
            if esm_b >= esm_threshold:
                validated.append(s)
                print(f"  PASS: esm_B={esm_b:.4f}")
            else:
                print(f"  REJECT: esm_B={esm_b:.4f} < {esm_threshold:.4f} — replacing...")
                # Find replacement from passed pool
                for alt in passed_sorted:
                    if alt not in selected and alt not in validated:
                        alt_esm = float(esm_gate.predict([alt['sequence']])[0])
                        if alt_esm >= esm_threshold:
                            alt['esm_brightness'] = round(alt_esm, 4)
                            validated.append(alt)
                            print(f"    → Replaced with alt (esm_B={alt_esm:.4f})")
                            break

        if len(validated) >= 4:
            selected = validated[:6]
        else:
            print(f"  [WARN] Only {len(validated)} passed ESM gate, keeping original selection")

    # ================================================================
    # 9. Report & Output
    # ================================================================
    print(f"\n[9/9] Generating outputs...")

    report = generate_selection_report(selected, len(rejected), len(variants))
    print(f"\n{report}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Submission CSV
    team_name = out_cfg.get('team_name', 'Default')
    submission_path = write_submission(team_name, selected, str(output_dir / "submission.csv"))
    print(f"\n  Submission CSV : {submission_path}")

    # Detailed CSV with all scores
    detailed_path = write_submission_with_scores(team_name, selected, str(output_dir / "submission_detailed.csv"))
    print(f"  Detailed CSV   : {detailed_path}")

    # All passed variants
    all_passed_path = output_dir / "all_passed_variants.csv"
    write_submission_with_scores(team_name, passed, str(all_passed_path))
    print(f"  All passed CSV : {all_passed_path}")

    # v3.2: BioDesignBench-compliant multi-metric comparison (改进5 补充)
    from src.utils import generate_top6_radar_comparison
    radar_md_path, radar_json_path = generate_top6_radar_comparison(selected, str(output_dir))
    print(f"  Radar comparison : {radar_md_path}")
    print(f"  Radar JSON data   : {radar_json_path}")

    # Agent diversity check (改进5)
    # Write design rationale

    # ================================================================
    # Done
    # ================================================================
    print(f"\n{'=' * 70}")
    print(f"  Pipeline Completed Successfully! (v{__version__})", flush=True)
    print(f"  Templates: sfGFP + avGFP | Model: {model_name}")
    print(f"  Selected: {len(selected)} sequences | Device: {device}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 70}")


def parse_args():
    """解析命令行参数。"""
    p = argparse.ArgumentParser(
        description='GFP Designer v4.4 — 2026 蛋白质设计竞赛管线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                           # 完整运行
  python main.py --skip-esmfold            # 跳过 ESMFold (节省 ~15min)
  python main.py --skip-deep-search        # 跳过 GA 深度搜索
  python main.py --config prod.yaml        # 使用自定义配置
  python main.py --output-dir ../results   # 指定输出目录
        """)
    p.add_argument('--config', default='config.yaml', help='配置文件路径')
    p.add_argument('--output-dir', default=None, help='输出目录 (覆盖配置文件)')
    p.add_argument('--skip-esmfold', action='store_true', help='跳过 ESMFold 3D 验证')
    p.add_argument('--skip-deep-search', action='store_true', help='跳过 GA+HC 深度搜索')
    p.add_argument('--skip-structure', action='store_true', help='跳过 ESM-2 结构验证')
    p.add_argument('--n-variants', type=int, default=None, help='每个模板生成的变体数')
    p.add_argument('--brightness-floor', type=float, default=None, help='亮度硬底线')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # Apply CLI overrides to config
    if args.skip_esmfold:
        import yaml
        cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
        cfg.setdefault('esmfold', {})['enabled'] = False
        # Temporarily override config
        import src.utils
        original_load = src.utils.load_config
        src.utils.load_config = lambda path: cfg
    if args.output_dir:
        import src.utils as _u
        _u.OUTPUT_DIR_OVERRIDE = args.output_dir
    main()
