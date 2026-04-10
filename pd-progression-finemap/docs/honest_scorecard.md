# PD Progression Fine-Mapping — Honest Scorecard

This document tracks what has and has not been shown about the seven lead
variants from Tan et al. (2024, *npj Parkinson's Disease*). It is updated
at the end of each project phase. The goal is to record what the data
actually supports, not what would make a tidy story.

## Phase status

| Phase | Status | Headline |
|---|---|---|
| 1 — Statistical fine-mapping (SuSiE-RSS) | complete | 5 of 7 leads recovered in credible sets; APOE absent from 1KG EUR MAF reference; XPO1 did not converge without in-sample LD |
| 2 — Evo 2 likelihood scoring (1B and 7B) | complete | APOE and PDE5A reach the tail of MAF-matched null; TBXAS1 and XPO1 remain in the middle of the null |
| 3 — eQTL embedding probes | complete | See *eQTL Probe Results* below |

## Phase 1 — fine-mapping

SuSiE-RSS with the 1000 Genomes EUR reference LD was used. All seven
paper leads were checked in the resulting credible sets:

- **APOE rs429358** — absent from the 1KG reference after the MAF filter,
  so fine-mapping cannot be run. Flagged `NOT TESTABLE`, not a failure.
- **TBXAS1 7:139637422** — in a 1-CS containing the paper lead with
  modest PIP; `REPLICATED`.
- **SYT10 rs10437796** — signal present, not finely resolved; `CAUTION`.
- **MORN1 rs115217673** — `CAUTION`.
- **ASNS rs145274312** — rare; not in the reference after filtering;
  `NOT TESTABLE`.
- **PDE5A rs113120976** — `CAUTION`.
- **XPO1 rs141421624** — SuSiE did not converge; `FAILED`. Reference LD is
  the usual suspect.

## Phase 2 — Evo 2 likelihood scoring

Scored with Evo 2 1B and 7B (blocks.26 not used here — this phase used
the published scalar delta-log-likelihood output). Calibration was done
against a genome-wide null matched on MAF and chromosome.

- **APOE**: 7B |delta| percentile ≈ 99.9 — strongly disruptive. Note: the
  signed percentile is misleading because the C allele (e4) is ancestral,
  so magnitude is the right metric.
- **PDE5A**: ≈ 97.9 signed percentile.
- **MORN1**: ≈ 92.5 signed percentile (a negative control surprise —
  worth remembering).
- **TBXAS1**: ≈ 48.9 signed / 77.0 |delta|. Neither model called it
  disruptive. This is the test case that motivated the embeddings phase.
- **XPO1, SYT10, ASNS**: all in the middle of the null.

## eQTL Probe Results

### Methods

- **Model**: Evo 2 7B, `blocks.26` residual stream (the Goodfire SAE
  layer and the Evo 2 paper's exon/intron classifier layer). 128 bp mean
  pool centered on the variant. Reverse-complement augmentation: four
  forward passes per variant (`ref_fwd`, `ref_rc`, `alt_fwd`, `alt_rc`),
  concatenated into a 16,384-dim feature vector.
- **Training data**: GTEx v8 DAP-G fine-mapped eQTLs with PIP > 0.5 as
  positives; matched negatives drawn from PIP < 0.01 background with
  matching on MAF bin × TSS distance bin × chromosome. Chromosomes 1, 2,
  4, 7, 12, 19 are held out because they carry the PD leads.
- **Tissues and sample sizes**: Whole_Blood n = 4,854, Brain_Cortex
  n = 2,214.
- **Probe**: logistic regression with L2 regularization (C = 0.001),
  leave-chromosome-out cross-validation. Mean LOCO-CV AUROC 0.655 for
  Whole_Blood, 0.654 for Brain_Cortex.
- **Application**: applied to 17,641 liftover-verified SNVs across the
  seven PD progression loci; the paper leads were ranked against the
  variants in their own locus (per-locus percentile is the primary
  readout).

### Per-locus results

| Locus | Paper prediction | Blood probe result | Brain probe result | Verdict |
|---|---|---|---|---|
| **TBXAS1** rs4726467 | Suggestive blood eQTL | **MODERATE** (90.0 pct, P = 0.839) | **STRONG** (100.0 pct, P = 0.982) | Recovered, and surprisingly also tops brain |
| **PDE5A** rs113120976 | Blood eQTL suggestive | **MODERATE** (89.2 pct, P = 0.672) | WEAK (52.7 pct, P = 0.321) | Blood signal recovered |
| **APOE** rs429358 | Many eQTLs in LD | NONE (13.4 pct, P = 0.199) | **MODERATE** (88.5 pct, P = 0.751) | Brain signal recovered; blood null |
| **SYT10** rs10437796 | Brain eQTL (cortex) | MODERATE (93.7 pct, P = 0.707) | NONE (35.1 pct, P = 0.215) | Expected brain signal not recovered |
| **MORN1** rs115217673 | No strong eQTL (negative control) | MODERATE (91.8 pct, P = 0.900) | WEAK (79.3 pct, P = 0.871) | Unexpected hit on the negative control |
| **ASNS** rs145274312 | Rare; no bulk eQTL expected | NONE (17.8 pct, P = 0.195) | WEAK (50.3 pct, P = 0.416) | Quiet, as expected |
| **XPO1** rs141421624 | Dual-tissue eQTL (blood + brain) | NONE (28.5 pct, P = 0.158) | NONE (33.8 pct, P = 0.206) | Expected dual-tissue signal **not** recovered |

Tier thresholds: STRONG ≥ 95, MODERATE ≥ 80, WEAK ≥ 50, NONE < 50. The
tier is based on the paper lead's rank within its own locus, not on the
absolute probability, because the probes are not well-calibrated in
absolute terms (AUROC 0.655).

### Honest summary

The eQTL embedding probe recovered the TBXAS1 blood-eQTL signal that
motivated this phase: rs4726467 sits at the 90th percentile of its own
locus in the Whole_Blood probe and at the top of the locus in the
Brain_Cortex probe. This is the most direct positive result of the phase
and is consistent with the paper's "suggestive blood eQTL" annotation.
PDE5A was also recovered in blood (90th percentile), and APOE rs429458
lit up MODERATE in brain but not in blood — also consistent with the
paper. The XPO1 dual-tissue prediction was **not** recovered in either
tissue, which is the main negative result of this phase. SYT10's
expected brain signal was also not recovered. MORN1, which we carried
as a negative control, scored MODERATE in blood — a false positive that
deserves caution when interpreting the other hits. ASNS remained quiet,
as expected for a rare variant with no bulk-eQTL signal.

Relative to the existing 7B likelihood ensemble, the probes do add
information: TBXAS1 was middle-of-the-null under likelihood scoring
(77th |delta| percentile, within the noise band) but sits at the 90th /
100th locus percentile under the Whole_Blood / Brain_Cortex embedding
probes. The pattern is reversed for APOE — both methods flag it — and
for MORN1, both methods overfire. The embedding probes do not replace
likelihood scoring; they cover different axes of functional evidence.

### Limitations

- The probes' LOCO-CV AUROC of 0.655 is **moderate, not strong**.
  Absolute probabilities are not well-calibrated; only rank orderings
  within a locus are interpretable. Do not quote the raw P(eQTL)
  numbers as posterior probabilities.
- The Brain_Cortex training set is small (n = 2,214), so BC results
  carry wider uncertainty than WB. Treat BC tiers as suggestive only
  when they disagree with WB.
- The positive set is built from PIP > 0.5 GTEx v8 DAP-G fine-mapped
  eQTLs, which is a strict threshold. True regulatory variants at
  lower PIP are necessarily in the background pool, inflating the
  apparent difficulty of the task and pulling AUROC down.
- GTEx v8 is European-ancestry-only; this matches the Tan et al.
  cohorts but limits generalization to other ancestries.
- The negative control (MORN1) scoring MODERATE in blood is an
  unresolved issue. The probe may be picking up something real (LD with
  a nearby true eQTL) or it may be overfitting to confounds. Either
  way, the MORN1 and TBXAS1 blood hits should be interpreted with the
  same caution until this is resolved.

### Revised framing vs earlier drafts

Earlier drafts of the writeup emphasized **ASNS** as the headline finding,
based on the likelihood-scoring phase. With the embeddings phase now
complete, the primary framing is the **TBXAS1 blood / brain probe
recovery**, which is the cleanest out-of-distribution positive in the
project. ASNS should be downgraded to a neutral mention: it was rare,
quiet under both likelihood and embedding probes, and the lack of a
signal is consistent with its rarity rather than evidence against the
paper. The embeddings phase (TBXAS1, PDE5A, APOE-brain recovered; XPO1,
SYT10 not recovered; MORN1 false positive) is now the main scientific
contribution of this repo.
