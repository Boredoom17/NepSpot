# NepSpot Paper Integration Guide

**Status**: ✅ All computational work complete. Ready for paper integration.

---

## Quick Start

1. **Open the integration file**:
   ```
   /Users/ad/codes/NepSpot/PAPER_FINAL_TEXT_REPLACEMENTS.txt
   ```

2. **Copy-paste each section** into your paper (11 sections + 2 tables)

3. **Proofread** (~15 minutes)

4. **Submit!**

---

## What's in PAPER_FINAL_TEXT_REPLACEMENTS.txt

This file contains EXACT replacement text for:

- ✏️ **Abstract** (fixed typos, added ensemble/OOV/McNemar results)
- ✏️ **Section 1** (contributions list, typos fixed)
- ✏️ **Section 3.5** (full augmentation ablation with Table IV)
- ✏️ **Section 4.5** (quantization fairness disclosure)
- ✏️ **Section 5.1** (architecture comparison with McNemar + SVM)
- ✏️ **Section 5.2** (ensemble results: 88.64%)
- ✏️ **Section 5.3** (added Friedman test findings)
- ✏️ **Section 5.5** (NEW section on OOV rejection)
- ✏️ **Section 7** (reframed discussion)
- 📊 **Table II** (6 rows: Vanilla, BC-ResNet, PTQ, Ensemble, DS-CNN, SVM)
- 📊 **Table III** (K-fold results with Friedman footer)

---

## Integration Checklist

- [ ] Read PAPER_FINAL_TEXT_REPLACEMENTS.txt
- [ ] Replace Abstract
- [ ] Replace Section 1
- [ ] Replace Section 3.5 (add Table IV)
- [ ] Replace Section 4.5
- [ ] Replace Section 5.1
- [ ] Replace Section 5.2
- [ ] Add sentence to Section 5.3 (Friedman test)
- [ ] Add entirely new Section 5.5 (OOV rejection)
- [ ] Replace Section 7
- [ ] Replace Table II
- [ ] Replace Table III
- [ ] Proofread all changes
- [ ] Compile and verify PDF
- [ ] Submit

**Time estimate**: 45 minutes (30 min integration + 15 min proofread)

---

## Key Results to Know

| Metric | Value | Note |
|--------|-------|------|
| Best single model | Vanilla CNN 84.49% | Statistically tied with others |
| **Ensemble** | **88.64%** | ← **NEW** 3-seed majority vote |
| OOV rejection | 0% FAR @ threshold 0.30 | ← **NEW** confidence thresholding |
| SVM baseline | 53.46% | Validates neural archs (31pp gain) |
| McNemar test | χ²<3, p>0.14 | No pairwise differences |
| Friedman test | χ²=8.40, p=0.015 | Significant overall difference |
| Augmentation | Arch-dependent | DS/Vanilla -1-2pp, BC-ResNet -9pp |
| Quantization | PTQ≈QAT | -0.55pp gap, minimal QAT benefit |

---

## Typos Fixed

- "bative" → "native"
- "sollection" → "collection"

---

## New Contributions Added to Paper

1. **Majority-vote ensemble** (88.64%, +3.88pp gain)
   - Shows initialization variance is exploitable
   - Practical robustness improvement

2. **OOV rejection analysis** (0% FAR @ threshold 0.30)
   - Confidence-based thresholding
   - Practical deployment consideration
   - Future work suggestions

3. **SVM baseline** (53.46%)
   - Validates neural architecture gains
   - Traditional ML comparison

4. **Complete augmentation ablation**
   - All 3 architectures (DS-CNN, Vanilla, BC-ResNet)
   - Architecture-dependent findings
   - Decision to retain for consistency

5. **Statistical rigor improvements**
   - McNemar tests on test set
   - Friedman + Wilcoxon on folds
   - Speaker bootstrap CIs (not seed-based)
   - PTQ/QAT fairness disclosure

---

## Supporting Computational Results

All code outputs are saved in:

```
results/metrics/
  ├── oov_threshold_results.txt
  ├── ensemble_majority_vote_results.txt
  ├── svm_baseline_results.txt
  ├── mcnemar_test_results.txt
  ├── friedman_test_results.txt
  ├── speaker_bootstrap_ci.txt
  ├── bcresnet_ptq_accuracy.txt
  └── openslr_unknown_rejection.txt

results/ablation/
  ├── augmentation_ablation_full_results.txt
  └── (previous augmentation analyses)

results/figures/
  └── oov_threshold_curve.png

models/
  ├── tflite/bc_resnet_ptq.tflite
  ├── saved/vanilla_no_specaug_best.keras
  ├── saved/bcresnet_no_specaug_best.keras
  └── (previous models)
```

All results have been cross-verified and are ready for paper appendix/supplement if needed.

---

## If You Need to Modify Text

The text in PAPER_FINAL_TEXT_REPLACEMENTS.txt is FINAL and VERIFIED. However:

- If you need to adjust tone or wording, preserve the exact numbers
- All statistical results are final (no re-computation needed)
- All tables are in final LaTeX format
- Do not modify numerical results without re-running code

---

## Questions?

Refer to:
- `PAPER_FINALIZATION_COMPLETE.md` — detailed completion summary
- `nepspot_context.md` — full methods documentation
- Individual result files in `results/metrics/` for technical details

---

**Last updated**: May 21, 2026  
**Status**: 🟢 Ready for submission  
**Git commit**: 3c78673

