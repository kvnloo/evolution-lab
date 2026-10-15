"""Verified loop on RECONSTRUCTED strata (prereg ``docs/prereg/verified-loop-reconstructed-v0.md``).

Stage A is the v0 learner unchanged (``verified_loop.evaluate``) on the stratum's train table. Stage B applies one
frozen policy (v0 model + train-matched ``tau_cov``) to the held-out eval-window table. Stage C applies the same
frozen policy, unchanged, to each LIVE observed table of the harness (one cohort at a time, never pooled). Pure
Python, deterministic, CPU; metrics and counts only.

    python -m evolution_lab.verified_loop_reconstructed --train T.train.jsonl --eval T.eval.jsonl \
        [--live L.jsonl ...] --out result.json [--md result.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import verified_loop as v0

PREREG = 'docs/prereg/verified-loop-reconstructed-v0.md'
RESULT_SCHEMA = 'evolution_lab.verified_loop_reconstructed.result.v0'
RECONSTRUCTED = 'reconstructed'
VERDICTS = ('INSUFFICIENT_DATA', 'NO_IMPROVEMENT', 'PENDING_LIVE_VALIDATION', 'NO_IMPROVEMENT_ON_LIVE',
            'SHADOW_CANDIDATE')


class Refused(ValueError):
    pass


def load(path: Path, *, reconstructed: bool) -> list[dict[str, Any]]:
    rows = v0.load_table(path) if path.exists() and path.stat().st_size else []
    bad = sum(1 for r in rows if (r.get('provenance') == RECONSTRUCTED) != reconstructed)
    if bad:
        raise Refused(f'{path.name}: {bad} rows of the wrong provenance (reconstructed and observed are never pooled)')
    return rows


def _schema(rows: Sequence[Mapping[str, Any]]) -> set:
    return {(r.get('feature_schema_sha'), r.get('table_version')) for r in rows}


def heldout_sufficiency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    y = [int(r['label']['y_success']) for r in rows]
    groups = {r['group'] for r in rows}
    checks = {f'rows>={v0.MIN_ROWS}': len(rows) >= v0.MIN_ROWS, f'not_success>={v0.MIN_NEG}': y.count(0) >= v0.MIN_NEG,
              f'groups>={v0.MIN_GROUPS}': len(groups) >= v0.MIN_GROUPS, 'both_classes': 0 in y and 1 in y}
    return {'passed': all(checks.values()), 'checks': checks,
            'observed': {'rows': len(rows), 'not_success': y.count(0), 'success': y.count(1), 'groups': len(groups)}}


class Frozen:
    """The v0 model fitted on the full train analysis set, with tau matched to G's train ACT coverage."""

    def __init__(self, train: Sequence[Mapping[str, Any]], *, steps: int = v0.STEPS):
        self.names = v0.feature_names(train)
        y = [int(r['label']['y_success']) for r in train]
        self.model = v0.Logistic(steps=steps).fit(v0.matrix(train, self.names), y)
        p = self.model.predict(v0.matrix(train, self.names))
        self.g_cov = v0.coverage([v0.gate_act(r) for r in train]) or 0.0
        self.tau = v0.tau_matched_coverage(p, [v0.legal_act(r) for r in train], self.g_cov)

    def apply(self, rows: Sequence[Mapping[str, Any]], *, boot_b: int = v0.BOOT_B) -> dict[str, Any]:
        suff = heldout_sufficiency(rows)
        if not rows:
            return {'sufficiency': suff, 'passed': False, 'criteria': None, 'metrics': None}
        y = [int(r['label']['y_success']) for r in rows]
        groups = [r['group'] for r in rows]
        p = self.model.predict(v0.matrix(rows, self.names))
        legal = [v0.legal_act(r) for r in rows]
        L = [lg and pi >= self.tau for lg, pi in zip(legal, p)]
        G = [v0.gate_act(r) for r in rows]
        rg, rl = v0.sel_risk(G, y), v0.sel_risk(L, y)
        delta = None if rg is None or rl is None else rl - rg
        ci = v0.cluster_bootstrap(y, groups, L, G, B=boot_b) if len(set(groups)) > 1 else \
            {'lo': None, 'hi': None, 'B': boot_b, 'skipped': boot_b}
        illegal = sum(1 for a, lg in zip(L, legal) if a and not lg)
        crit = {'delta<0_and_ci_hi<0': delta is not None and delta < 0 and ci['hi'] is not None and ci['hi'] < 0,
                'never_acts_when_illegal': illegal == 0}
        return {'sufficiency': suff, 'criteria': crit, 'passed': suff['passed'] and all(crit.values()),
                'metrics': {'coverage_G': v0.coverage(G), 'coverage_L_cov': v0.coverage(L),
                            'risk_G': rg, 'risk_L_cov': rl, 'delta_L_cov_minus_G': delta, 'delta_ci95': ci,
                            'illegal_act_rows': illegal, 'auroc': v0.auroc(p, y), 'brier': v0.brier(p, y)}}


def verdict(a: str, b: Mapping[str, Any] | None, live: Mapping[str, Mapping[str, Any]]) -> tuple[str, str]:
    """(final verdict, verdict from the reconstructed data alone) per the prereg decision rule."""
    if a == v0.INSUFFICIENT or not b or not b['sufficiency']['passed']:
        return 'INSUFFICIENT_DATA', 'INSUFFICIENT_DATA'
    if a == v0.NO_IMPROVEMENT or not b['passed']:
        return 'NO_IMPROVEMENT', 'NO_IMPROVEMENT'
    sufficient = [c for c in live.values() if c['sufficiency']['passed']]
    if any(c['passed'] for c in sufficient):
        return 'SHADOW_CANDIDATE', 'RECONSTRUCTED_PROPOSAL'
    if sufficient:
        return 'NO_IMPROVEMENT_ON_LIVE', 'RECONSTRUCTED_PROPOSAL'
    return 'PENDING_LIVE_VALIDATION', 'RECONSTRUCTED_PROPOSAL'


def run(train_path: Path, eval_path: Path, live_paths: Sequence[Path] = (), *, steps: int = v0.STEPS,
        boot_b: int = v0.BOOT_B) -> dict[str, Any]:
    import hashlib
    train = load(train_path, reconstructed=True)
    held = load(eval_path, reconstructed=True)
    lives = {p.stem: load(p, reconstructed=False) for p in live_paths}
    schemas = _schema(train) | _schema(held) | set().union(*[_schema(r) for r in lives.values()])
    if len(schemas) > 1:
        raise Refused(f'feature schema / table version differ across tables: {sorted(map(str, schemas))}')
    tr_ana, tr_excl = v0.analysis_set(train)
    ev_ana, ev_excl = v0.analysis_set(held)
    a = v0.evaluate(tr_ana, steps=steps, boot_b=boot_b)
    frozen = Frozen(tr_ana, steps=steps) if tr_ana and len({int(r['label']['y_success']) for r in tr_ana}) > 1 \
        else None
    b = frozen.apply(ev_ana, boot_b=boot_b) if frozen else {'sufficiency': heldout_sufficiency(ev_ana),
                                                             'passed': False, 'criteria': None, 'metrics': None,
                                                             'why': 'no frozen policy (train set empty or one class)'}
    live: dict[str, dict[str, Any]] = {}
    for name, rows in sorted(lives.items()):
        ana, excl = v0.analysis_set(rows)
        live[name] = dict(frozen.apply(ana, boot_b=boot_b) if frozen else
                          {'sufficiency': heldout_sufficiency(ana), 'passed': False, 'criteria': None, 'metrics': None},
                          input={'rows': len(rows), 'analysis_rows': len(ana), 'excluded': excl})
    final, recon = verdict(a['decision'], b, live)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None  # noqa: E731
    return {'schema': RESULT_SCHEMA, 'prereg': PREREG, 'base_prereg': v0.PREREG, 'verdict': final,
            'reconstructed_verdict': recon,
            'inputs': {'train': {'sha256': sha(train_path), 'rows': len(train), 'analysis_rows': len(tr_ana),
                                 'excluded': tr_excl},
                       'eval': {'sha256': sha(eval_path), 'rows': len(held), 'analysis_rows': len(ev_ana),
                                'excluded': ev_excl},
                       'live': {p.stem: sha(p) for p in live_paths}},
            'stage_a_train_cv': {k: a.get(k) for k in ('decision', 'sufficiency', 'criteria', 'k', 'pooled')},
            'frozen_policy': None if frozen is None else {'tau_cov': v0._fin(frozen.tau), 'train_G_coverage':
                                                          frozen.g_cov, 'features': len(frozen.names)},
            'stage_b_eval_window': b, 'stage_c_live': live,
            'note': 'A reconstructed-trained policy is only a proposal until stage C passes on LIVE rows; even a '
                    'SHADOW_CANDIDATE earns shadow only, with owner approval. Nothing is configured or promoted.',
            'privacy': 'metrics and counts only; no rows, ids or text'}


def markdown(res: Mapping[str, Any]) -> str:
    f = v0._f
    L = ['# Verified loop on reconstructed strata -- result', '', f'Pre-registration: `{res["prereg"]}` '
         f'(addendum to `{res["base_prereg"]}`). Metrics and counts only.', '',
         f'## Verdict: **{res["verdict"]}** (reconstructed data alone: {res["reconstructed_verdict"]})', '']
    i = res['inputs']
    a = res['stage_a_train_cv']
    L += [f'- Stage A (train, grouped CV, v0 rule): {a["decision"]}; analysis rows {i["train"]["analysis_rows"]} of '
          f'{i["train"]["rows"]}; sufficiency {a["sufficiency"]["observed"]}']
    b = res['stage_b_eval_window']
    m = b.get('metrics') or {}
    L += [f'- Stage B (eval window, frozen policy): sufficiency {b["sufficiency"]["observed"]} passed='
          f'{b["sufficiency"]["passed"]}; delta {f(m.get("delta_L_cov_minus_G"))} CI '
          f'[{f((m.get("delta_ci95") or {}).get("lo"))}, {f((m.get("delta_ci95") or {}).get("hi"))}]; '
          f'coverage G {f(m.get("coverage_G"))} vs L {f(m.get("coverage_L_cov"))}']
    for name, c in res['stage_c_live'].items():
        m = c.get('metrics') or {}
        L += [f'- Stage C live `{name}`: analysis rows {c["input"]["analysis_rows"]}; sufficient='
              f'{c["sufficiency"]["passed"]}; delta {f(m.get("delta_L_cov_minus_G"))}; passed={c["passed"]}']
    return '\n'.join(L + ['', res['note']]) + '\n'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='python -m evolution_lab.verified_loop_reconstructed', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', type=Path, required=True)
    ap.add_argument('--eval', type=Path, required=True)
    ap.add_argument('--live', type=Path, action='append', default=[])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--md', type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        res = run(args.train, args.eval, args.live)
    except Refused as exc:
        print(f'refused: {exc}')
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
    if args.md:
        args.md.write_text(markdown(res))
    print(f"verdict={res['verdict']} train_analysis_rows={res['inputs']['train']['analysis_rows']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
