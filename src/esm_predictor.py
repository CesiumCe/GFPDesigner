# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Cesium
#
"""
esm_predictor.py — ESM-2 微调亮度预测器 (v4.4)

封装 full_predict 的 5-Fold ESM-2 T6 微调模型 (Spearman 0.96, R² 0.94)。
使用 fair-esm 离线加载, 无需网络。预测速度 ~50ms/seq (GPU) / ~200ms/seq (CPU)。

Usage:
    predictor = EsmBrightnessPredictor(model_dir)
    predictor.load()
    brightness = predictor.predict([seq1, seq2, ...])
"""
import torch
import numpy as np
from pathlib import Path
from typing import List


class EsmBrightnessPredictor:
    def __init__(self, model_dir: str, device: str = "auto"):
        self.model_dir = Path(model_dir)
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        self.models = []
        self.alphabet = None
        self._loaded = False

    def load(self) -> bool:
        """加载 5-Fold 微调模型 + 回归头。"""
        import esm
        from torch import nn

        try:
            _, self.alphabet = esm.pretrained.load_model_and_alphabet('esm2_t6_8M_UR50D')
        except Exception as e:
            print(f"  [ESM-Predict] Failed: {e}"); return False

        class RegressionHead(nn.Module):
            """Linear(320→128) → ReLU → Dropout(0.1) → Linear(128→1)"""
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(320, 128)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.1)
                self.fc2 = nn.Linear(128, 1)

            def forward(self, x):
                return self.fc2(self.dropout(self.relu(self.fc1(x))))

        for fold in range(5):
            ft_path = self.model_dir / f"esm_fold{fold}.pt"
            if not ft_path.exists(): continue

            sd = torch.load(ft_path, map_location='cpu', weights_only=True)
            ft_state = sd.get('model_state_dict', sd)

            base = esm.pretrained.load_model_and_alphabet('esm2_t6_8M_UR50D')[0]
            # Load ESM backbone via key remap
            backbone = self._remap_keys(ft_state, base)
            base.load_state_dict(backbone, strict=False)

            # Build regression head and load weights
            head = RegressionHead()
            head_keys = {k.replace('regression_head.', ''): v
                         for k, v in ft_state.items() if 'regression_head' in k}
            # Sequential keys: "0.weight" → "fc1.weight", "3.weight" → "fc2.weight"
            head_map = {'0.weight': 'fc1.weight', '0.bias': 'fc1.bias',
                        '3.weight': 'fc2.weight', '3.bias': 'fc2.bias'}
            head_sd = {}
            for k, v in head_keys.items():
                mapped = head_map.get(k, k)
                head_sd[mapped] = v
            head.load_state_dict(head_sd)

            model = nn.Module()
            model.esm = base
            model.head = head
            model = model.to(self.device)
            model.eval()
            self.models.append(model)

        if not self.models: return False

        self._loaded = True
        print(f"  [ESM-Predict] Loaded {len(self.models)}-fold ESM-2 T6 ensemble on {self.device}")
        return True

    def _remap_keys(self, ft_state: dict, base_model) -> dict:
        """Map transformers 'esm.xxx' keys to fair-esm keys by shape matching."""
        base_sd = base_model.state_dict()
        remapped = {}
        for ft_key, ft_val in ft_state.items():
            if 'contact_head' in ft_key:
                continue
            ft_shape = ft_val.shape
            # Try direct key transform
            t_key = ft_key.replace('esm.encoder.layer.', 'layers.').replace(
                'esm.embeddings.word_embeddings.', 'embed_tokens.').replace(
                'esm.encoder.emb_layer_norm_after.', 'emb_layer_norm_after.')
            # Fallback: shape matching
            if t_key in base_sd and base_sd[t_key].shape == ft_shape:
                remapped[t_key] = ft_val
            else:
                for bk, bv in base_sd.items():
                    if bv.shape == ft_shape and bk not in remapped:
                        remapped[bk] = ft_val
                        break
        return remapped

    def predict(self, sequences: List[str], batch_size: int = 16) -> np.ndarray:
        """
        预测序列亮度 (5-Fold 集成均值)。
        Returns: numpy array of shape (n,) — 绝对亮度值 (WT avGFP ≈ 3.72)
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        batch_converter = self.alphabet.get_batch_converter()
        all_preds = []
        LAYER = self.models[0].esm.num_layers  # 6 for T6

        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            _, _, tokens = batch_converter([(f"s{j}", s) for j, s in enumerate(batch)])
            tokens = tokens.to(self.device)

            fold_preds = []
            for model in self.models:
                with torch.no_grad():
                    output = model.esm(tokens, repr_layers=[LAYER])
                    emb = output['representations'][LAYER][:, 1:-1, :].mean(dim=1)  # [B,320]
                    pred = model.head(emb).squeeze(-1)  # [B]
                fold_preds.append(pred.cpu().numpy())
            all_preds.append(np.mean(fold_preds, axis=0))

        return np.concatenate(all_preds)

