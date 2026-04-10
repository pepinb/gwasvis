# DAP-G Fine-Mapping Inspection Report

GTEx v8 DAP-G eQTL credible sets (Zenodo 3517189). Two tissues extracted for PD progression fine-mapping.

### Whole Blood

**File**: `data/eqt1/raw/Whole_Blood.variants_pip.txt.gz`  
**Rows**: 420,370  
**Unique variant_ids**: 367,370  
**Unique genes**: 15,252  

#### Column schema

| Column | dtype |
|--------|-------|
| gene | str |
| rank | int64 |
| variant_id | str |
| pip | float64 |
| log10_abf | float64 |
| cluster_id | str |

#### Variant ID examples

- `chr20_50958797_A_C_b38`
- `chr20_50960127_T_G_b38`
- `chr20_50965358_A_G_b38`

#### PIP distribution

| Threshold | Variants | % of total |
|-----------|----------|------------|
| PIP > 0.9 | 2,147 | 0.5% |
| PIP > 0.5 | 5,080 | 1.2% |
| PIP > 0.1 | 28,559 | 6.8% |
| PIP > 0.01 | 175,063 | 41.6% |
| PIP ≤ 0.01 | 245,307 | 58.4% |

#### Chromosome distribution (all variants)

| Chr | Count | % |
|-----|-------|---|
| 1 * | 44,595 | 10.6% |
| 2 * | 29,920 | 7.1% |
| 3 | 24,327 | 5.8% |
| 4 * | 16,704 | 4.0% |
| 5 | 18,999 | 4.5% |
| 6 | 24,345 | 5.8% |
| 7 * | 21,635 | 5.1% |
| 8 | 15,085 | 3.6% |
| 9 | 15,621 | 3.7% |
| 10 | 18,597 | 4.4% |
| 11 | 24,186 | 5.8% |
| 12 * | 23,839 | 5.7% |
| 13 | 7,887 | 1.9% |
| 14 | 13,643 | 3.2% |
| 15 | 13,521 | 3.2% |
| 16 | 19,078 | 4.5% |
| 17 | 25,436 | 6.1% |
| 18 | 6,371 | 1.5% |
| 19 * | 29,543 | 7.0% |
| 20 | 11,065 | 2.6% |
| 21 | 4,398 | 1.0% |
| 22 | 11,575 | 2.8% |

\* = PD held-out chromosome

#### PIP > 0.5 by PD-held-out vs rest

| Group | Count | % of PIP>0.5 |
|-------|-------|--------------|
| PD chroms (1, 2, 4, 7, 12, 19) | 1,959 | 38.6% |
| Other chroms | 3,121 | 61.4% |
| **Total** | **5,080** | **100%** |

#### SNV vs indel

| Type | Count | % |
|------|-------|---|
| SNV | 387,644 | 92.2% |
| Indel | 32,726 | 7.8% |

### Brain Cortex

**File**: `data/eqt1/raw/Brain_Cortex.variants_pip.txt.gz`  
**Rows**: 452,122  
**Unique variant_ids**: 399,253  
**Unique genes**: 18,258  

#### Column schema

| Column | dtype |
|--------|-------|
| gene | str |
| rank | int64 |
| variant_id | str |
| pip | float64 |
| log10_abf | float64 |
| cluster_id | str |

#### Variant ID examples

- `chr20_50953866_A_G_b38`
- `chr20_50952488_C_T_b38`
- `chr20_50947674_G_A_b38`

#### PIP distribution

| Threshold | Variants | % of total |
|-----------|----------|------------|
| PIP > 0.9 | 952 | 0.2% |
| PIP > 0.5 | 2,683 | 0.6% |
| PIP > 0.1 | 20,483 | 4.5% |
| PIP > 0.01 | 158,202 | 35.0% |
| PIP ≤ 0.01 | 293,920 | 65.0% |

#### Chromosome distribution (all variants)

| Chr | Count | % |
|-----|-------|---|
| 1 * | 47,663 | 10.5% |
| 2 * | 32,721 | 7.2% |
| 3 | 25,725 | 5.7% |
| 4 * | 18,881 | 4.2% |
| 5 | 22,835 | 5.1% |
| 6 | 28,396 | 6.3% |
| 7 * | 22,137 | 4.9% |
| 8 | 18,010 | 4.0% |
| 9 | 17,818 | 3.9% |
| 10 | 19,032 | 4.2% |
| 11 | 23,616 | 5.2% |
| 12 * | 24,475 | 5.4% |
| 13 | 9,390 | 2.1% |
| 14 | 14,519 | 3.2% |
| 15 | 15,486 | 3.4% |
| 16 | 18,764 | 4.2% |
| 17 | 26,827 | 5.9% |
| 18 | 6,783 | 1.5% |
| 19 * | 30,742 | 6.8% |
| 20 | 12,112 | 2.7% |
| 21 | 4,440 | 1.0% |
| 22 | 11,750 | 2.6% |

\* = PD held-out chromosome

#### PIP > 0.5 by PD-held-out vs rest

| Group | Count | % of PIP>0.5 |
|-------|-------|--------------|
| PD chroms (1, 2, 4, 7, 12, 19) | 1,008 | 37.6% |
| Other chroms | 1,675 | 62.4% |
| **Total** | **2,683** | **100%** |

#### SNV vs indel

| Type | Count | % |
|------|-------|---|
| SNV | 416,599 | 92.1% |
| Indel | 35,523 | 7.9% |
