# Third-Party Notices

This project builds upon the following open-source models and libraries.

## Bundled Models

### ESMFold v1
- **Source**: Meta AI / Hugging Face (`facebook/esmfold_v1`)
- **License**: MIT
- **Paper**: Lin et al., "Evolutionary-scale prediction of atomic-level protein structure with a language model", *Science* 379, 1123–1130 (2023)
- **Included at**: `ESMFold/pytorch_model.bin` (7.9 GB)

### ESM-2 (via fair-esm)
- **Source**: Meta AI (`facebookresearch/esm`)
- **License**: MIT
- **Paper**: Lin et al., "Language models of protein sequences at the scale of evolution enable accurate structure prediction", *Science* 379, 1123–1130 (2023)

### ESM-Predict (Fine-tuned)
- **Source**: Custom 5-fold fine-tuning of ESM-2 T6 (6M parameters) on GFP brightness data
- **License**: MIT (derived from ESM-2 MIT license)
- **Included at**: `esm_predict_models/` (~156 MB)

## Python Dependencies

| Library | License | Usage |
|---------|---------|-------|
| `fair-esm` | MIT | ESM-2 protein language model inference |
| `transformers` | Apache 2.0 | ESMFold model loading |
| `torch` | BSD | Deep learning framework |
| `xgboost` | Apache 2.0 | Ensemble brightness prediction (fallback) |
| `scikit-learn` | BSD | Machine learning utilities |
| `pandas` | BSD | Data processing |
| `numpy` | BSD | Numerical computation |
| `pyyaml` | MIT | Configuration parsing |
| `openpyxl` | MIT | Excel data loading |
| `matplotlib` | PSF-based | Scatter plot generation |

## Key References

These papers informed our design but are not included as code:

| Reference | Use in Project |
|-----------|---------------|
| Close et al. (2015) *Proteins* 83(7):1225–1237 | TGP surface supercharging strategy (net=+10) |
| Pédelacq et al. (2006) *Nature Biotech* 24:79–88 | sfGFP ion pair network |
| Hirano et al. (2022) *Nature Biotech* 40:1132–1142 | Cl⁻ pocket photostability |
| Frenzel et al. (2018) *Biotech for Biofuels* 11:8 | sfGFP thermostable variant validation |
| Fraikin et al. (2025) *Science Advances* 11 | GFP variant sequence alignment |
| Meier et al. (2021) *NeurIPS* | ESM-1v masked marginal scoring method |
| Jiang et al. (2025) *Science* | EVOLVEpro local hill-climbing + Golden Pairs |
| Gelman et al. (2025) *Nature Methods* | METL protein design with physical constraints |
| Tan et al. (2025) *eLife* | ProtSSN Contact Order geometric features |
| Ding et al. (2026) *ABB* | sfGFP folding robustness review |
| Kim & Romero (2026) *bioRxiv* | BioDesignBench multi-metric comparison |
