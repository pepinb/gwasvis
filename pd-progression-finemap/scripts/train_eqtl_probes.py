"""Train cross-chromosome eQTL probes across three feature constructions.

For each tissue (Whole_Blood, Brain_Cortex), runs leave-one-chromosome-out
CV on concat / delta / hybrid, declares a winner, refits a production probe
on the winning construction, and saves winner + runner-ups + metadata.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault('PYTHONWARNINGS', 'ignore')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eqtl.probes import (  # noqa: E402
    DEFAULT_C_GRID,
    FEATURE_CONSTRUCTIONS,
    EQTLProbe,
    _chrom_sort_key,
    _git_commit,
    build_features,
    train_with_leave_chr_out,
)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

TISSUES = ['Whole_Blood', 'Brain_Cortex']
EMBED_DIR = ROOT / 'data' / 'eqtl' / 'embeddings'
PROBE_DIR = ROOT / 'data' / 'eqtl' / 'probes'
PROBE_DIR.mkdir(parents=True, exist_ok=True)

HELD_OUT = [1, 2, 4, 7, 12, 19]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_tissue(tissue: str):
    path = EMBED_DIR / f'{tissue}_features.parquet'
    sha = _sha256(path)
    df = pd.read_parquet(path)
    emb = np.stack(
        [np.frombuffer(b, dtype=np.float32) for b in df['embedding']], axis=0
    )
    labels = df['label'].to_numpy().astype(int)
    chroms = df['chr'].astype(str).to_numpy()
    return df, emb, labels, chroms, path, sha


def print_preflight(tissue, emb, labels, chroms):
    n = len(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    uniq = sorted(np.unique(chroms).tolist(), key=_chrom_sort_key)
    counts = [int((chroms == c).sum()) for c in uniq]
    min_i = int(np.argmin(counts))
    max_i = int(np.argmax(counts))

    print(f'=== {tissue} probe training ===')
    print(f'n_samples:       {n}')
    print(f'n_positives:     {n_pos}')
    print(f'n_negatives:     {n_neg}')
    print(f'chromosomes:     {len(uniq)} unique')
    print(f'min chr count:   {counts[min_i]}  ({uniq[min_i]})')
    print(f'max chr count:   {counts[max_i]}  ({uniq[max_i]})')
    print(
        'feature_dim:     16384 (cached) -> 16384/8192/16384 (concat/delta/hybrid)'
    )
    print()


def print_per_construction(tissue, construction, result):
    print(f'--- {tissue} / {construction} ---')
    best_C_lookup = dict(result['all_C_choices'])
    ordered = sorted(result['per_chromosome_auroc'].keys(), key=_chrom_sort_key)
    for c in ordered:
        auroc = result['per_chromosome_auroc'][c]
        n = result['per_chromosome_n'][c]
        print(f'  {c:<6} AUROC={auroc:.3f}  (n={n}, best C={best_C_lookup[c]:g})')
    min_c = min(result['per_chromosome_auroc'], key=result['per_chromosome_auroc'].get)
    max_c = max(result['per_chromosome_auroc'], key=result['per_chromosome_auroc'].get)
    print(f'  Mean AUROC:     {result["mean_auroc"]:.3f}')
    print(f'  Std AUROC:      {result["std_auroc"]:.3f}')
    print(
        f'  Min fold:       {result["min_auroc"]:.3f}  '
        f'({min_c}, n={result["per_chromosome_n"][min_c]})'
    )
    print(
        f'  Max fold:       {result["max_auroc"]:.3f}  '
        f'({max_c}, n={result["per_chromosome_n"][max_c]})'
    )
    print(f'  Modal best C:   {result["modal_best_C"]:g}')
    print()


def choose_winner(results: dict) -> str:
    ranked = sorted(
        results.items(), key=lambda kv: kv[1]['mean_auroc'], reverse=True
    )
    top_name, top = ranked[0]
    if len(ranked) > 1:
        second_name, second = ranked[1]
        if abs(top['mean_auroc'] - second['mean_auroc']) < 0.005:
            if top_name == 'concat' or second_name == 'concat':
                return 'concat'
    return top_name


def print_shootout(tissue, results, winner):
    print(f'=== {tissue} construction shootout ===')
    print(f'{"construction":<16}{"mean AUROC":<14}{"std":<9}winner')
    for con in FEATURE_CONSTRUCTIONS:
        r = results[con]
        mark = '[X]' if con == winner else '   '
        print(
            f'{con:<16}{r["mean_auroc"]:.3f}         '
            f'{r["std_auroc"]:.3f}    {mark}'
        )
    print()


def print_diagnostics(tissue, construction, result, probe, chroms, labels):
    coefs = probe.model.named_steps['lr'].coef_[0]
    top20 = np.argsort(-np.abs(coefs))[:20]
    print(f'Top 20 |coef| features ({tissue} / {construction}):')
    for i, idx in enumerate(top20):
        print(f'  {i+1:2d}. dim={int(idx):5d}  coef={coefs[idx]:+.4f}')
    print()

    print('Per-fold label balance:')
    for c in sorted(np.unique(chroms).tolist(), key=_chrom_sort_key):
        mask = chroms == c
        n = int(mask.sum())
        pos = int((labels[mask] == 1).sum())
        neg = int((labels[mask] == 0).sum())
        flag = '  !! unreliable (n < 40)' if n < 40 else ''
        print(f'  {c:<6} n={n}  pos={pos} neg={neg}{flag}')
    print()


def emit_warnings(tissue, construction, result):
    msgs = []
    mean = result['mean_auroc']
    std = result['std_auroc']

    if tissue == 'Whole_Blood':
        if mean < 0.58:
            msgs.append(
                'Whole_Blood winning AUROC below expected 0.62-0.72 range. '
                'Matching may be too tight or blocks.26 may not carry enough '
                'eQTL signal at this tissue.'
            )
        elif mean > 0.80:
            msgs.append(
                'Whole_Blood winning AUROC above expected range (>0.80). '
                'Check for chromosome leakage (a variant on both sides of a fold) '
                "or label leakage (the embedding parquet might have label info "
                "somewhere it shouldn't)."
            )
    elif tissue == 'Brain_Cortex':
        if mean < 0.52:
            msgs.append(
                'Brain_Cortex AUROC < 0.52: likely too little signal to work with.'
            )
        elif mean < 0.55:
            msgs.append(
                'Brain_Cortex AUROC in 0.52-0.55 borderline range: underpowered '
                'but not broken. Report honestly with explicit disclaimer.'
            )
        elif mean > 0.80:
            msgs.append(
                'Brain_Cortex winning AUROC above expected range (>0.80). '
                'Check for chromosome leakage or label leakage.'
            )

    if std > 0:
        for c, a in result['per_chromosome_auroc'].items():
            z = (a - mean) / std
            if abs(z) > 3:
                msgs.append(
                    f'Unstable fold on {c}; '
                    f'n={result["per_chromosome_n"][c]}, AUROC={a:.3f}'
                )

    if std > 0.10:
        msgs.append(
            'High fold variance (std > 0.10); probe may be fitting '
            'chromosome-specific noise. Consider stronger L2 regularization '
            '(extend C_grid downward).'
        )

    if construction != 'concat':
        msgs.append(
            f'Feature construction shootout winner was {construction}, not '
            f'concat. This is an interesting finding and may reflect that eQTL '
            f'prediction is more delta-shaped than protein-function prediction '
            f"(the Evo 2 paper's concat result came from BRCA1 LOF, a different "
            f'task class).'
        )

    if msgs:
        print('WARNINGS:')
        for m in msgs:
            print(f'  ! {m}')
        print()
    else:
        print('No diagnostic warnings.')
        print()


def serialize_constructions(per_construction):
    out = []
    for con in FEATURE_CONSTRUCTIONS:
        r = per_construction[con]
        out.append(
            {
                'construction': r['construction'],
                'feature_dim': r['feature_dim'],
                'mean_auroc': r['mean_auroc'],
                'std_auroc': r['std_auroc'],
                'min_auroc': r['min_auroc'],
                'max_auroc': r['max_auroc'],
                'modal_best_C': r['modal_best_C'],
                'per_chromosome_auroc': r['per_chromosome_auroc'],
                'per_chromosome_n': r['per_chromosome_n'],
                'all_C_choices': [[c, C] for c, C in r['all_C_choices']],
            }
        )
    return out


def main():
    all_results = {}
    winners = {}

    for tissue in TISSUES:
        print('#' * 70)
        print(f'# {tissue}')
        print('#' * 70)

        df, emb, labels, chroms, path, sha = load_tissue(tissue)
        print_preflight(tissue, emb, labels, chroms)

        per_construction = {}
        for construction in FEATURE_CONSTRUCTIONS:
            print(f'--- {tissue} / {construction} (running) ---', flush=True)
            result = train_with_leave_chr_out(
                embeddings=emb,
                labels=labels,
                chromosomes=chroms,
                construction=construction,
                C_grid=DEFAULT_C_GRID,
                verbose=True,
            )
            per_construction[construction] = result
            print_per_construction(tissue, construction, result)

        winner = choose_winner(per_construction)
        print_shootout(tissue, per_construction, winner)

        win_result = per_construction[winner]
        modal_C = win_result['modal_best_C']

        X_win = build_features(emb, winner)
        probe = EQTLProbe(C=modal_C, construction=winner)
        probe.fit(X_win, labels)
        winner_path = PROBE_DIR / f'{tissue}_evo2_7b_blocks26.joblib'
        probe.save(winner_path)
        print(f'Saved winning probe: {winner_path}')

        for con in FEATURE_CONSTRUCTIONS:
            if con == winner:
                continue
            r = per_construction[con]
            X_ru = build_features(emb, con)
            runner = EQTLProbe(C=r['modal_best_C'], construction=con)
            runner.fit(X_ru, labels)
            ru_path = (
                PROBE_DIR
                / f'{tissue}_evo2_7b_blocks26_{con}_runnerup.joblib'
            )
            runner.save(ru_path)
            print(f'Saved runner-up probe: {ru_path}')
        print()

        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        metadata = {
            'winning_construction': winner,
            'all_constructions': serialize_constructions(per_construction),
            'training_data_path': str(path.relative_to(ROOT)),
            'training_data_sha256': sha,
            'n_positives': n_pos,
            'n_negatives': n_neg,
            'C_grid': DEFAULT_C_GRID,
            'held_out_chromosomes': HELD_OUT,
            'embedding_layer': 'blocks.26',
            'model': 'evo2_7b',
            'pool_window': 128,
            'training_timestamp': dt.datetime.now(dt.timezone.utc).isoformat(),
            'git_commit': _git_commit(),
            'modal_best_C': win_result['modal_best_C'],
            'per_chromosome_auroc': win_result['per_chromosome_auroc'],
            'per_chromosome_n': win_result['per_chromosome_n'],
            'mean_auroc': win_result['mean_auroc'],
            'std_auroc': win_result['std_auroc'],
        }
        metadata_path = (
            PROBE_DIR / f'{tissue}_evo2_7b_blocks26_metadata.json'
        )
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f'Saved metadata: {metadata_path}')
        print()

        print_diagnostics(tissue, winner, win_result, probe, chroms, labels)
        emit_warnings(tissue, winner, win_result)

        all_results[tissue] = per_construction
        winners[tissue] = winner

    bar = '=' * 68
    print(bar)
    print('                       CONSTRUCTION SHOOTOUT RESULTS')
    print(bar)
    header = f'{"":<16}{"concat":<15}{"delta":<15}{"hybrid":<15}winner'
    print(header)
    for tissue in TISSUES:
        r = all_results[tissue]
        cols = []
        for con in FEATURE_CONSTRUCTIONS:
            cols.append(f'{r[con]["mean_auroc"]:.3f}+/-{r[con]["std_auroc"]:.3f} ')
        w = winners[tissue]
        print(f'{tissue:<16}{cols[0]:<15}{cols[1]:<15}{cols[2]:<15}{w}')
    print(bar)
    print()
    for tissue in TISSUES:
        w = winners[tissue]
        mean = all_results[tissue][w]['mean_auroc']
        print(f'{tissue} winning probe: {w}, mean AUROC {mean:.3f}')
    print()

    bc_winner = winners['Brain_Cortex']
    bc_mean = all_results['Brain_Cortex'][bc_winner]['mean_auroc']
    if bc_mean < 0.55:
        print(
            'WARNING: Brain_Cortex probe underperformed expected range even at'
        )
        print('full training scale. Two interpretations:')
        print(
            '(a) The training set (n=2,214) is genuinely underpowered for this'
        )
        print('    task at 16k features.')
        print('(b) Brain cortex eQTL signal is not well-represented in Evo 2')
        print('    blocks.26 at 128bp pooling.')
        print()
        print('Recommendation for prompt 8: report BC results with explicit')
        print('underpowered disclaimer. Focus headline analysis on Whole_Blood.')
        print(
            'Optionally run a sensitivity analysis later with PIP threshold 0.3'
        )
        print('to roughly double the positive count.')
        print()


if __name__ == '__main__':
    main()
