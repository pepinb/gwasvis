# Evo 2 Model Agreement Diagnostic: 1B vs 7B

## Global Rank Correlation (Genome Null)

- **Spearman ρ (signed):** 0.6945
- **Pearson r (signed):** 0.6398
- **Spearman ρ (|Δ|):** 0.4711
- **n matched:** 6400
- **Interpretation:** Meaningful agreement but substantial per-variant disagreement.

## Per-Locus Rank Correlation

| Locus | n_variants | ρ (signed) | ρ (\|Δ\|) | Flag |
|-------|-----------|-----------|---------|------|
| APOE | 3031 | 0.6961 | 0.4703 | LOW ρ SIGN |
| TBXAS1 | 2569 | 0.6959 | 0.4737 | LOW ρ SIGN |
| SYT10 | 2891 | 0.6392 | 0.4261 | LOW ρ SIGN |
| MORN1 | 2112 | 0.6535 | 0.4257 | LOW ρ SIGN |
| ASNS | 2690 | 0.6573 | 0.4310 | LOW ρ SIGN |
| PDE5A | 2550 | 0.6432 | 0.4283 | LOW ρ SIGN |
| XPO1 | 2140 | 0.7110 | 0.4950 | SIGN |

Mean per-locus ρ (signed): 0.6709

## Bland-Altman Summary

- **Mean diff (7B − 1B):** -0.0886
- **SD of diff:** 2.5579
- **Regression slope:** -0.0468 (p=6.44e-05)

## Disagreement vs Variant Features

| Feature | Spearman ρ with rank_diff | Interpretation |
|---------|--------------------------|----------------|
| GC content | -0.0371 | No relationship |
| MAF | 0.0069 | No relationship |
| |1B delta_ll| | -0.2457 | Moderate |

## Seven Paper Leads: Side-by-Side

| Locus | Lead | 1B Δ | 7B Δ | 1B signed% | 7B signed% | 1B \|Δ\|% | 7B \|Δ\|% | Dir agree? |
|-------|------|------|------|------------|------------|---------|---------|------------|
| APOE | rs429358 | +15.19 | +9.53 | 99.9% | 99.5% | 0.1% | 0.6% | YES |
| TBXAS1 | rs4726467 | -0.36 | -0.79 | 51.6% | 46.6% | 85.8% | 73.0% | YES |
| SYT10 | rs10437796 | -4.34 | -3.15 | 7.2% | 15.6% | 11.2% | 24.0% | YES |
| MORN1 | rs115217673 | +2.24 | +2.55 | 88.1% | 92.2% | 36.6% | 29.5% | YES |
| ASNS | rs145274312 | -4.77 | -2.88 | 7.5% | 18.9% | 9.4% | 24.8% | YES |
| PDE5A | rs113120976 | +4.22 | +3.69 | 97.5% | 96.6% | 12.1% | 15.4% | YES |
| XPO1 | rs141421624 | -3.23 | -4.90 | 16.2% | 6.2% | 21.8% | 7.9% | YES |

## Figures

- `agreement_scatter.png` — 1B vs 7B delta_ll scatter (8 panels)
- `agreement_bland_altman.png` — Bland-Altman disagreement vs magnitude

## Interpretation

With a global Spearman ρ of 0.694, the 1B and 7B models share meaningful signal but disagree substantially on individual variants. Per-locus correlations average 0.671. They are measuring overlapping but not identical aspects of sequence constraint.

The ASNS shift (+11.4pp) is within the range of shifts seen across other loci (mean |shift| = 5.7pp). This is a global pattern of model disagreement, not a locus-specific anomaly.

The Bland-Altman SD of 2.56 means 95% of variants fall within ±5.1 delta_ll units of agreement between models. For variants in the moderate tail (5th-20th percentile), this noise can shift the percentile by roughly ±10 points. Single-model percentiles in the 5%-20% range should be reported with a ~±10pp uncertainty band; extreme percentiles (<2% or >98%) and null results (40%-60%) are robust to model choice.

All seven leads agree on delta direction (7/7 direction-concordant). An ensemble mean of 1B and 7B percentiles would reduce the influence of per-model noise and is likely more stable than either model alone, particularly for loci in the 5%-25% range where model choice matters most.
