# Evo 2 Model Comparison: 1B vs 7B

| Locus | Trait | Lead | 1B Δ | 1B matched% | 7B Δ | 7B matched% | 7B – 1B Δ |
|-------|-------|------|------|-------------|------|-------------|-----------|
| APOE | mortality | rs429358 | +15.19 | 0.1% (\|Δ\|) | +9.53 | 0.6% (\|Δ\|) | -5.66 |
| TBXAS1 | mortality | rs4726467 | -0.36 | 51.6% | -0.79 | 46.6% | -0.43 |
| SYT10 | mortality | rs10437796 | -4.34 | 7.2% | -3.15 | 15.6% | +1.19 |
| MORN1 | hy3 | rs115217673 | +2.24 | 88.1% | +2.55 | 92.2% | +0.32 |
| ASNS | hy3 | rs145274312 | -4.77 | 7.5% | -2.88 | 18.9% | +1.90 |
| PDE5A | hy3 | rs113120976 | +4.22 | 97.5% | +3.69 | 96.6% | -0.53 |
| XPO1 | hy3 | rs141421624 | -3.23 | 16.2% | -4.90 | 6.2% | -1.67 |

## Key Findings

1. **APOE** (positive control): 7B preserves strong signal. |Δ| percentile: 1B=0.1% → 7B=0.6%. Delta direction: both positive (ancestral allele preference). Magnitude reduced (+15.19 → +9.53) but still top 1%.

2. **TBXAS1** (headline test for model-size effects): 1B matched%=51.6% → 7B=46.6%. 7B did NOT detect a signal that 1B missed. The lead variant remains in the middle of the distribution. Delta: -0.36 → -0.79 (slightly more negative, still weak).

3. **ASNS**: 1B matched%=7.5% → 7B=18.9%. Weakened from strong (7.5%) to moderate (18.9%). Delta: -4.77 → -2.88.

4. **XPO1**: 1B matched%=16.2% → 7B=6.2%. Sharpened from weak (16.2%) to moderate (6.2%). Delta: -3.23 → -4.90. Strongest improvement from the 7B upgrade.

5. **MORN1**: Still wrong direction. 1B=88.1% → 7B=92.2%. Delta: +2.24 → +2.55.

6. **PDE5A**: Still wrong direction. 1B=97.5% → 7B=96.6%. Delta: +4.22 → +3.69.

7. **SYT10**: Weakened slightly. 1B=7.2% → 7B=15.6%. Delta: -4.34 → -3.15.
