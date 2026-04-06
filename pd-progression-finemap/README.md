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

## Usage

```bash
uv run streamlit run app/streamlit_app.py
```
