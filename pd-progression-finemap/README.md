# PD Progression GWAS Fine-Mapping

Statistical fine-mapping pipeline for Parkinson's disease progression GWAS loci.

## Project Structure

- `src/` — Pipeline modules (download, loci extraction, LD, fine-mapping, evaluation)
- `app/` — Streamlit interactive dashboard
- `notebooks/` — Exploratory analysis notebooks
- `data/` — Raw and processed data (git-ignored)
- `tests/` — Test suite

## Setup

```bash
uv sync
```

## Pipeline

```bash
# 1. Download GWAS summary statistics from Zenodo
uv run python -m src.download

# 2. Extract loci (±500 kb around each lead SNP)
uv run python -m src.loci

# 3. Download 1KG EUR reference and compute LD matrices
uv run python -m src.ld --download
uv run python -m src.ld --all

# 4. Fine-map all loci with SuSiE-RSS
uv run python -m src.finemap --all

# 5. Summarise results
uv run python -m src.evaluate
```

## Dashboard

Launch the interactive locus viewer:

```bash
uv run streamlit run app/streamlit_app.py
```

The dashboard provides a three-panel view for each locus:

- **LocusZoom plot** — scatter of -log10(p) vs position, colored by PIP,
  with the paper's lead SNP marked as a red star
- **PIP track** — bar chart of posterior inclusion probabilities, with
  credible set variants highlighted in blue
- **Summary panel** — CS size, top PIP/SNP, convergence status, credible
  set variant table, and full locus data expander

## Tests

```bash
uv run pytest tests/ -v
```
