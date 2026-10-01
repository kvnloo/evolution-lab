"""Verified loop v0: offline learned ACT gate vs the deterministic gate (evolution-lab#24).

Implements exactly the pre-registration in ``docs/prereg/verified-loop-v0.md``. Pure Python
(stdlib only), deterministic, CPU. Input is the privacy-safe ``z0int.loop.training_row.v0`` table
from ``z0int outcomes export`` (no text); output is a counts/metrics-only JSON + markdown report.

    python -m evolution_lab.verified_loop --table T.jsonl [--manifest T.manifest.json] \
        --out results/verified-loop-v0.json [--md results/verified-loop-v0.md]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PREREG = 'docs/prereg/verified-loop-v0.md'
ROW_SCHEMA = 'z0int.loop.training_row.v0'
TABLE_VERSION = '0.1.0'
RESULT_SCHEMA = 'evolution_lab.verified_loop.result.v0'

# --- pre-registered constants (do not tune; a change is a new pre-registration) -----------------
K = 5
LAMBDA = 1.0
LR = 0.1
STEPS = 2000
ALPHA = 0.02
ALPHA_SLACK = 0.01
COVERAGE_TOL = 0.02
BOOT_B = 2000
BOOT_SEED = 0
MIN_ROWS = 300
MIN_NEG = 30
MIN_GROUPS = 10
MIN_FOLDS_NEG = 3  # delta < 0 in at least 3 of 5 folds
TAU_GRID = [i / 100 for i in range(101)]
FALLBACK = ('ASK', 'OBSERVE', 'ABSTAIN')

INSUFFICIENT, NO_IMPROVEMENT, SHADOW = 'INSUFFICIENT_DATA', 'NO_IMPROVEMENT', 'SHADOW_CANDIDATE'


# ----------------------------------------------------------------------------- data
def load_table(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get('schema') != ROW_SCHEMA:
                raise ValueError(f'unexpected row schema {r.get("schema")!r}')
            if r.get('table_version') != TABLE_VERSION:
                raise ValueError(f'table_version {r.get("table_version")!r} != {TABLE_VERSION} (pre-registered)')
            rows.append(r)
    return rows


def analysis_set(rows: Iterable[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
    """has_opportunity AND label resolved AND observed action ACT (outcome is the outcome of acting)."""
    keep, why = [], Counter()
    for r in rows:
        if not r.get('has_opportunity') or r.get('features') is None:
            why['no_opportunity'] += 1
        elif not (r.get('label') or {}).get('resolved'):
            why['unverified_or_no_label'] += 1
        elif (r.get('observed') or {}).get('action') != 'ACT':
            why['observed_not_act' if r.get('observed') else 'no_observed_row'] += 1
        else:
            keep.append(r)
    return keep, dict(why)


def feature_names(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return list(rows[0]['features']) if rows else []


def matrix(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> list[list[float]]:
    return [[float(r['features'].get(n, 0)) for n in names] for r in rows]


def legal_act(r: Mapping[str, Any]) -> bool:
    return bool(r['features'].get('legal.ACT'))


def gate_act(r: Mapping[str, Any]) -> bool:
    return r.get('gate') == 'ACT'


def fallback(r: Mapping[str, Any]) -> str:
    for a in FALLBACK[:-1]:
        if r['features'].get(f'legal.{a}'):
            return a
    return 'ABSTAIN'


# ----------------------------------------------------------------------------- model
class Logistic:
    """L2 logistic regression, standardised inputs, full-batch GD from zero. Deterministic."""

    def __init__(self, lam: float = LAMBDA, lr: float = LR, steps: int = STEPS):
        self.lam, self.lr, self.steps = lam, lr, steps
        self.mu: list[float] = []
        self.sd: list[float] = []
        self.w: list[float] = []
        self.b = 0.0

    def _z(self, X: Sequence[Sequence[float]]) -> list[list[float]]:
        return [[(x - m) / s if s > 0 else 0.0 for x, m, s in zip(row, self.mu, self.sd)] for row in X]

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> 'Logistic':
        n, d = len(X), len(X[0]) if X else 0
        self.mu = [sum(r[j] for r in X) / n for j in range(d)] if n else [0.0] * d
        self.sd = [math.sqrt(sum((r[j] - self.mu[j]) ** 2 for r in X) / n) for j in range(d)] if n else [0.0] * d
        Z = self._z(X)
        active = [j for j in range(d) if self.sd[j] > 0]
        w = [0.0] * d
        b = 0.0  # zero init (pre-registered); penalty is lam * ||w||^2 / (2n), bias unpenalised
        for _ in range(self.steps if n else 0):
            gw = [0.0] * d
            gb = 0.0
            for zi, yi in zip(Z, y):
                p = _sigmoid(b + sum(w[j] * zi[j] for j in active))
                e = p - yi
                gb += e
                for j in active:
                    gw[j] += e * zi[j]
            for j in active:
                w[j] -= self.lr * (gw[j] / n + self.lam * w[j] / n)
            b -= self.lr * gb / n
        self.w, self.b = w, b
        return self

    def predict(self, X: Sequence[Sequence[float]]) -> list[float]:
        return [_sigmoid(self.b + sum(wj * zj for wj, zj in zip(self.w, z))) for z in self._z(X)]


def _sigmoid(t: float) -> float:
    if t >= 0:
        return 1.0 / (1.0 + math.exp(-t))
    e = math.exp(t)
    return e / (1.0 + e)


# ----------------------------------------------------------------------------- thresholds
def tau_matched_coverage(p: Sequence[float], legal: Sequence[bool], target_cov: float) -> float:
    """Smallest-coverage tau whose coverage is >= target (never below the gate's coverage)."""
    n = len(p)
    m = math.ceil(target_cov * n - 1e-9)
    scores = sorted((pi for pi, li in zip(p, legal) if li), reverse=True)
    if m <= 0:
        return math.inf
    if m > len(scores):
        return -math.inf
    return scores[m - 1]


def tau_crc(p: Sequence[float], legal: Sequence[bool], y: Sequence[int], alpha: float = ALPHA) -> float:
    """Conformal risk control on loss 1[ACT and y=0] (monotone non-increasing in tau)."""
    n = len(p)
    for tau in TAU_GRID:
        loss = sum(1 for pi, li, yi in zip(p, legal, y) if li and pi >= tau and yi == 0)
        if (loss + 1) / (n + 1) <= alpha:
            return tau
    return math.inf


# ----------------------------------------------------------------------------- metrics
def coverage(act: Sequence[bool]) -> float | None:
    return sum(act) / len(act) if act else None


def sel_risk(act: Sequence[bool], y: Sequence[int]) -> float | None:
    k = sum(act)
    return sum(1 for a, yi in zip(act, y) if a and yi == 0) / k if k else None


def auroc(p: Sequence[float], y: Sequence[int]) -> float | None:
    pos = [pi for pi, yi in zip(p, y) if yi == 1]
    neg = [pi for pi, yi in zip(p, y) if yi == 0]
    if not pos or not neg:
        return None
    s = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in pos for b in neg)
    return s / (len(pos) * len(neg))


def brier(p: Sequence[float], y: Sequence[int]) -> float | None:
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y)) / len(y) if y else None


# ----------------------------------------------------------------------------- CV
def group_folds(groups: Sequence[str], k: int = K) -> tuple[int, dict[str, int]]:
    uniq = sorted(set(groups))
    k = min(k, len(uniq))
    return k, {g: i % k for i, g in enumerate(uniq)}


def run_cv(rows: Sequence[Mapping[str, Any]], *, k: int = K, steps: int = STEPS) -> dict[str, Any]:
    names = feature_names(rows)
    y = [int(r['label']['y_success']) for r in rows]
    groups = [r['group'] for r in rows]
    k, fold_of = group_folds(groups, k)
    oof = [{} for _ in rows]
    folds = []
    for f in range(k):
        tr = [i for i, g in enumerate(groups) if fold_of[g] != f]
        te = [i for i, g in enumerate(groups) if fold_of[g] == f]
        Xtr = matrix([rows[i] for i in tr], names)
        ytr = [y[i] for i in tr]
        model = Logistic(steps=steps).fit(Xtr, ytr)
        ptr = model.predict(Xtr)
        g_cov = coverage([gate_act(rows[i]) for i in tr]) or 0.0
        t_cov = tau_matched_coverage(ptr, [legal_act(rows[i]) for i in tr], g_cov)
        # CRC: last quarter of training groups (hash order) calibrates; refit on the rest
        tr_groups = sorted({groups[i] for i in tr})
        n_cal = len(tr_groups) // 4
        cal_g = set(tr_groups[len(tr_groups) - n_cal:]) if n_cal else set()
        fit_i = [i for i in tr if groups[i] not in cal_g]
        cal_i = [i for i in tr if groups[i] in cal_g]
        m2 = Logistic(steps=steps).fit(matrix([rows[i] for i in fit_i], names), [y[i] for i in fit_i]) if fit_i else model
        pcal = m2.predict(matrix([rows[i] for i in cal_i], names)) if cal_i else []
        t_crc = tau_crc(pcal, [legal_act(rows[i]) for i in cal_i], [y[i] for i in cal_i])
        Xte = matrix([rows[i] for i in te], names)
        pte = model.predict(Xte)
        pte2 = m2.predict(Xte)
        prev = sum(ytr) / len(ytr) if ytr else 0.5
        for i, p1, p2 in zip(te, pte, pte2):
            leg = legal_act(rows[i])
            oof[i] = {'fold': f, 'p': p1, 'p_prev': prev, 'G': gate_act(rows[i]),
                      'L_cov': leg and p1 >= t_cov, 'L_crc': leg and p2 >= t_crc, 'legal': leg}
        yte = [y[i] for i in te]
        rg = sel_risk([oof[i]['G'] for i in te], yte)
        rl = sel_risk([oof[i]['L_cov'] for i in te], yte)
        folds.append({'fold': f, 'n_test': len(te), 'neg_test': yte.count(0), 'pos_test': yte.count(1),
                      'groups_test': len({groups[i] for i in te}), 'tau_cov': _fin(t_cov), 'tau_crc': _fin(t_crc),
                      'n_cal': len(cal_i), 'risk_G': rg, 'risk_L_cov': rl,
                      'delta': None if rg is None or rl is None else rl - rg,
                      'cov_G': coverage([oof[i]['G'] for i in te]), 'cov_L_cov': coverage([oof[i]['L_cov'] for i in te])})
    return {'k': k, 'oof': oof, 'folds': folds, 'y': y, 'groups': groups}


def _fin(x: float) -> float | str:
    return x if math.isfinite(x) else ('inf' if x > 0 else '-inf')


def cluster_bootstrap(y: Sequence[int], groups: Sequence[str], a: Sequence[bool], b: Sequence[bool],
                      B: int = BOOT_B, seed: int = BOOT_SEED) -> dict[str, Any]:
    """95% percentile CI of risk(a) - risk(b), resampling groups with replacement."""
    rng = random.Random(seed)
    by_g: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        by_g.setdefault(g, []).append(i)
    keys = sorted(by_g)
    stats, skipped = [], 0
    for _ in range(B):
        idx = [i for g in (rng.choice(keys) for _ in keys) for i in by_g[g]]
        ra = sel_risk([a[i] for i in idx], [y[i] for i in idx])
        rb = sel_risk([b[i] for i in idx], [y[i] for i in idx])
        if ra is None or rb is None:
            skipped += 1
            continue
        stats.append(ra - rb)
    if not stats:
        return {'lo': None, 'hi': None, 'B': B, 'skipped': skipped}
    stats.sort()
    q = lambda t: stats[min(len(stats) - 1, max(0, int(round(t * (len(stats) - 1)))))]
    return {'lo': q(0.025), 'hi': q(0.975), 'B': B, 'skipped': skipped}


# ----------------------------------------------------------------------------- decision
def sufficiency(rows: Sequence[Mapping[str, Any]], cv: Mapping[str, Any] | None) -> dict[str, Any]:
    y = [int(r['label']['y_success']) for r in rows]
    checks = {
        'rows>=300': len(rows) >= MIN_ROWS,
        'not_success>=30': y.count(0) >= MIN_NEG,
        'groups>=10': len({r['group'] for r in rows}) >= MIN_GROUPS,
        'k==5_and_every_test_fold_has_both_classes': bool(cv) and cv['k'] == K and all(
            f['neg_test'] >= 1 and f['pos_test'] >= 1 for f in cv['folds']),
    }
    return {'passed': all(checks.values()), 'checks': checks,
            'observed': {'rows': len(rows), 'not_success': y.count(0), 'success': y.count(1),
                         'groups': len({r['group'] for r in rows}), 'k': (cv or {}).get('k')}}


def evaluate(rows: Sequence[Mapping[str, Any]], *, steps: int = STEPS, boot_b: int = BOOT_B) -> dict[str, Any]:
    if not rows:
        suff = sufficiency(rows, None)
        return {'decision': INSUFFICIENT, 'sufficiency': suff, 'descriptive_only': True, 'pooled': None}
    if len({r['group'] for r in rows}) < 2:  # grouped CV impossible: no held-out session exists
        y = [int(r['label']['y_success']) for r in rows]
        g = [gate_act(r) for r in rows]
        return {'decision': INSUFFICIENT, 'descriptive_only': True, 'sufficiency': sufficiency(rows, None),
                'k': 1, 'cv': 'not possible with fewer than 2 groups',
                'pooled': {'G': {'coverage': coverage(g), 'selective_not_success': sel_risk(g, y)},
                           'n': len(y), 'not_success': y.count(0)}}
    cv = run_cv(rows, steps=steps)
    oof, y, groups = cv['oof'], cv['y'], cv['groups']
    act = {pol: [o[pol] for o in oof] for pol in ('G', 'L_cov', 'L_crc')}
    pooled = {pol: {'coverage': coverage(a), 'selective_not_success': sel_risk(a, y)} for pol, a in act.items()}
    rg, rl = pooled['G']['selective_not_success'], pooled['L_cov']['selective_not_success']
    delta = None if rg is None or rl is None else rl - rg
    ci = cluster_bootstrap(y, groups, act['L_cov'], act['G'], B=boot_b)
    joint_crc = sum(1 for a, yi in zip(act['L_crc'], y) if a and yi == 0) / len(y)
    illegal = sum(1 for o in oof if (o['L_cov'] or o['L_crc']) and not o['legal'])
    folds_neg = sum(1 for f in cv['folds'] if f['delta'] is not None and f['delta'] < 0)
    p = [o['p'] for o in oof]
    suff = sufficiency(rows, cv)
    crit = {
        'coverage_matched': pooled['L_cov']['coverage'] is not None and
        abs(pooled['L_cov']['coverage'] - pooled['G']['coverage']) <= COVERAGE_TOL,
        'delta<0_and_ci_hi<0': delta is not None and delta < 0 and ci['hi'] is not None and ci['hi'] < 0,
        'delta<0_in>=3_folds': folds_neg >= MIN_FOLDS_NEG,
        'crc_joint_risk<=alpha+slack': joint_crc <= ALPHA + ALPHA_SLACK,
        'never_acts_when_illegal': illegal == 0,
    }
    decision = INSUFFICIENT if not suff['passed'] else SHADOW if all(crit.values()) else NO_IMPROVEMENT
    fb = Counter(fallback(r) for r, a in zip(rows, act['L_crc']) if not a)
    return {
        'decision': decision, 'descriptive_only': not suff['passed'], 'sufficiency': suff,
        'criteria': crit, 'k': cv['k'],
        'pooled': {**pooled, 'delta_L_cov_minus_G': delta, 'delta_ci95': ci, 'folds_with_delta<0': folds_neg,
                   'crc_joint_risk': joint_crc, 'illegal_act_rows': illegal,
                   'auroc_oof': auroc(p, y), 'brier_oof': brier(p, y),
                   'brier_prevalence': brier([o['p_prev'] for o in oof], y),
                   'L_crc_fallback': dict(fb)},
        'folds': cv['folds'],
    }


def sensitivity_unverified_as_failure(all_rows: Sequence[Mapping[str, Any]], *, steps: int, boot_b: int) -> dict[str, Any]:
    """Secondary, never decisional: unverified rows (observed ACT) counted as not-success."""
    rows = []
    for r in all_rows:
        lab = r.get('label') or {}
        if r.get('has_opportunity') and r.get('features') and (r.get('observed') or {}).get('action') == 'ACT' \
                and lab.get('present'):
            rows.append({**r, 'label': {**lab, 'y_success': lab.get('y_success') or 0, 'resolved': True}})
    if not rows:
        return {'rows': 0}
    ev = evaluate(rows, steps=steps, boot_b=boot_b)
    return {'rows': len(rows), 'pooled': ev['pooled'], 'decision_if_primary': ev['decision']}


# ----------------------------------------------------------------------------- projection
def required_n_halving(p0: float, power_z: float = 0.8416, alpha_z: float = 1.6449) -> int | None:
    """Two-proportion normal approximation, one-sided alpha .05, power .8, p1 = p0 / 2."""
    if not 0 < p0 < 1:
        return None
    p1 = p0 / 2
    pb = (p0 + p1) / 2
    num = alpha_z * math.sqrt(2 * pb * (1 - pb)) + power_z * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))
    return math.ceil((num / (p0 - p1)) ** 2)


def _epoch(iso: str | None) -> float | None:
    if not iso:
        return None
    from datetime import datetime
    try:
        return datetime.fromisoformat(iso.replace('Z', '+00:00')).timestamp()
    except ValueError:
        return None


def observed_span_days(sweep: Mapping[str, Any], nominal: float = 7.0) -> float:
    """Days the sweep actually covers: first turn found -> measurement (>= 1 day, <= nominal).

    A '7d' sweep over a transcript store that only retains a day or two of history must not be
    divided by 7.
    """
    a, b = _epoch(sweep.get('first_turn_at')), _epoch(sweep.get('measured_at'))
    if a is None or b is None:
        return nominal
    return min(nominal, max(1.0, (b - a) / 86400))


POPULATIONS = {
    # the population the live gate serves (Claude Code with the z0 hooks): real work, not eval arms
    'live (interactive+agent)': ('interactive', 'agent'),
    'all cohorts (incl. eval/probe harness)': ('interactive', 'agent', 'harness', 'unknown'),
}


def _sum(by_cohort: Mapping[str, Mapping[str, int]], cohorts: Iterable[str], key: str) -> int:
    return sum((by_cohort.get(c) or {}).get(key, 0) for c in cohorts)


def projection(sweep: Mapping[str, Any] | None, manifest_counts: Mapping[str, Any] | None,
               analysis_rows: int, analysis_neg: int, analysis_groups: int = 0, *,
               sweep_days: float | None = None) -> dict[str, Any]:
    """Weeks until the sufficiency gate / 80% power, per population and feature-coverage scenario."""
    if not sweep:
        return {'available': False, 'why': 'no manifest.sweep (run z0int outcomes export --sweep-since 7d)'}
    sweep_days = sweep_days or observed_span_days(sweep)
    wk = 7.0 / sweep_days
    by_c = sweep.get('by_cohort') or {'unknown': sweep.get('total') or {}}
    sess = sweep.get('sessions_by_cohort') or {}
    all_c = POPULATIONS['all cohorts (incl. eval/probe harness)']
    all_res = sum(_sum(by_c, all_c, k) for k in ('verified_success', 'verified_failure', 'contested'))
    all_neg = _sum(by_c, all_c, 'verified_failure') + _sum(by_c, all_c, 'contested')
    pooled_p0 = all_neg / all_res if all_res and all_neg else None
    pops = {}
    for pop, cohorts in POPULATIONS.items():
        turns = _sum(by_c, cohorts, 'turns')
        resolved = sum(_sum(by_c, cohorts, k) for k in ('verified_success', 'verified_failure', 'contested'))
        neg = _sum(by_c, cohorts, 'verified_failure') + _sum(by_c, cohorts, 'contested')
        after = _sum(by_c, cohorts, 'turns_after_first_opportunity')
        cov_now = _sum(by_c, cohorts, 'with_opportunity_after_first_opportunity') / after if after else 0.0
        sessions_wk = sum(sess.get(c, 0) for c in cohorts) * wk
        p0 = neg / resolved if resolved and neg else None
        # a rate from fewer than 10 events is not a planning rate: use the pooled one instead
        p_plan, p_src = (p0, 'population') if neg >= 10 else (pooled_p0, 'pooled_all_cohorts')
        n_power = required_n_halving(p_plan) if p_plan else None
        scen = {}
        for name, cov in (('current_feature_coverage', cov_now), ('full_feature_coverage', 1.0)):
            r_wk = resolved * wk * cov
            n_wk = r_wk * p_plan if p_plan else 0.0
            g_wk = sessions_wk * (1.0 if cov > 0 else 0.0)
            parts = {'rows': max(MIN_ROWS - analysis_rows, 0) / r_wk if r_wk else math.inf,
                     'not_success': max(MIN_NEG - analysis_neg, 0) / n_wk if n_wk else math.inf,
                     'groups': max(MIN_GROUPS - analysis_groups, 0) / g_wk if g_wk else math.inf}
            weeks_power = max(n_power - analysis_rows, 0) / r_wk if (n_power and r_wk) else math.inf
            scen[name] = {'feature_coverage': cov, 'resolved_per_week': r_wk, 'not_success_per_week': n_wk,
                          'sessions_per_week': g_wk, 'weeks_to_sufficiency_gate': _fin(max(parts.values())),
                          'binding_constraint': max(parts, key=parts.get),
                          'rows_for_80pct_power_halving': n_power, 'weeks_to_power': _fin(weeks_power)}
        pops[pop] = {'turns_per_week': turns * wk, 'resolved_per_week': resolved * wk,
                     'not_success_per_week': neg * wk, 'sessions_per_week': sessions_wk,
                     'resolved_fraction': resolved / turns if turns else None,
                     'base_not_success_rate': p0, 'not_success_observed': neg,
                     'planning_not_success_rate': p_plan, 'planning_rate_source': p_src,
                     'feature_coverage_since_emission_began': cov_now,
                     'turns_since_emission_began': after, 'scenarios': scen}
    return {'available': True, 'sweep_days': sweep_days, 'sessions': sweep.get('sessions'),
            'sessions_by_cohort': sess, 'populations': pops,
            'caveat': 'linear extrapolation of a short window at current utilization; tiny not-success counts make '
                      'the base rate (and so the power target) very uncertain; labels mature over the fix window '
                      '(up to 7 days), so resolved counts lag'}


# ----------------------------------------------------------------------------- report
def run(table: Path, manifest: Path | None = None, *, steps: int = STEPS, boot_b: int = BOOT_B) -> dict[str, Any]:
    rows = load_table(table)
    man = json.loads(manifest.read_text()) if manifest and manifest.exists() else {}
    ana, excluded = analysis_set(rows)
    primary = evaluate(ana, steps=steps, boot_b=boot_b)
    neg = sum(1 for r in ana if r['label']['y_success'] == 0)
    return {
        'schema': RESULT_SCHEMA, 'prereg': PREREG, 'table_version': TABLE_VERSION,
        'feature_schema_sha': man.get('feature_schema_sha'), 'sources': man.get('sources'),
        'input': {'rows': len(rows), 'analysis_rows': len(ana), 'excluded': excluded,
                  'gate': dict(Counter(str(r.get('gate')) for r in rows if r.get('has_opportunity'))),
                  'states': dict(Counter(str((r.get('label') or {}).get('state')) for r in rows))},
        'primary': primary,
        'sensitivity_unverified_as_not_success': sensitivity_unverified_as_failure(rows, steps=steps, boot_b=boot_b),
        'projection': projection(man.get('sweep'), man.get('counts'), len(ana), neg, len({r['group'] for r in ana})),
        'privacy': 'metrics and counts only; no rows, ids or text',
    }


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return 'n/a'
    if isinstance(x, float):
        return f'{x:.{nd}f}'
    return str(x)


def markdown(res: Mapping[str, Any]) -> str:
    p = res['primary']
    L = ['# Verified loop v0 -- result', '',
         f'Pre-registration: `{res["prereg"]}`. Table `{res["table_version"]}`, feature schema '
         f'`{res.get("feature_schema_sha")}`. Metrics and counts only.', '',
         f'## Decision: **{p["decision"]}**' + (' (analysis is descriptive only)' if p.get('descriptive_only') else ''),
         '', '| sufficiency check | observed | passed |', '|---|---|---|']
    obs = p['sufficiency']['observed']
    vals = {'rows>=300': obs['rows'], 'not_success>=30': obs['not_success'], 'groups>=10': obs['groups'],
            'k==5_and_every_test_fold_has_both_classes': f'k={obs["k"]}'}
    L += [f'| {k} | {vals[k]} | {v} |' for k, v in p['sufficiency']['checks'].items()]
    i = res['input']
    L += ['', f'Input rows {i["rows"]}; analysis rows {i["analysis_rows"]}; excluded {i["excluded"]}.', '']
    if p.get('pooled') and 'L_cov' not in p['pooled']:
        q = p['pooled']
        L += [f'Grouped CV not possible ({p.get("cv")}). Gate only, descriptive: n={q["n"]}, '
              f'not-success={q["not_success"]}, G ACT coverage {_f(q["G"]["coverage"])}, '
              f'selective not-success {_f(q["G"]["selective_not_success"])}.']
    if p.get('pooled') and 'L_cov' in p['pooled']:
        q = p['pooled']
        L += ['## Pooled out-of-fold (grouped CV, k=%s)' % p.get('k'), '',
              '| policy | ACT coverage | selective not-success |', '|---|---:|---:|']
        L += [f'| {pol} | {_f(q[pol]["coverage"])} | {_f(q[pol]["selective_not_success"])} |' for pol in ('G', 'L_cov', 'L_crc')]
        ci = q['delta_ci95']
        L += ['', f'- delta (L_cov - G): {_f(q["delta_L_cov_minus_G"])}, 95% cluster-bootstrap CI '
                  f'[{_f(ci["lo"])}, {_f(ci["hi"])}] (B={ci["B"]}, skipped {ci["skipped"]})',
              f'- folds with delta < 0: {q["folds_with_delta<0"]}; L_crc joint risk {_f(q["crc_joint_risk"])} '
              f'(alpha {ALPHA}); illegal ACT rows {q["illegal_act_rows"]}',
              f'- AUROC {_f(q["auroc_oof"])}; Brier {_f(q["brier_oof"])} vs prevalence {_f(q["brier_prevalence"])}',
              f'- L_crc fallback when not acting: {q["L_crc_fallback"]}', '',
              '| criterion | met |', '|---|---|']
        L += [f'| {k} | {v} |' for k, v in p['criteria'].items()]
    s = res['sensitivity_unverified_as_not_success']
    if s.get('pooled') and 'L_cov' in s['pooled']:
        q = s['pooled']
        L += ['', f'Sensitivity (unverified counted as not-success, {s["rows"]} rows, not decisional): '
                  f'G {_f(q["G"]["selective_not_success"])} vs L_cov {_f(q["L_cov"]["selective_not_success"])}, '
                  f'AUROC {_f(q["auroc_oof"])}.']
    pr = res['projection']
    L += ['', '## Data-volume projection', '']
    if not pr.get('available'):
        L += [f'- unavailable: {pr.get("why")}']
    else:
        L += [f'- sweep span actually covered: {pr["sweep_days"]:.2f} days (rates scaled to a week); '
              f'sessions {pr.get("sessions")} by cohort {pr.get("sessions_by_cohort")}', '',
              '| population | turns/wk | resolved/wk | not-success/wk (observed n) | sessions/wk | not-success rate (planning, source) | feature coverage since emission began |',
              '|---|---:|---:|---:|---:|---|---:|']
        for pop, c in pr['populations'].items():
            L += [f'| {pop} | {c["turns_per_week"]:.0f} | {c["resolved_per_week"]:.0f} | '
                  f'{c["not_success_per_week"]:.1f} ({c["not_success_observed"]}) | {c["sessions_per_week"]:.0f} | '
                  f'{_f(c["base_not_success_rate"])} ({_f(c["planning_not_success_rate"])}, {c["planning_rate_source"]}) | '
                  f'{_f(c["feature_coverage_since_emission_began"])} |']
        L += ['', '| population | scenario | weeks to sufficiency gate | binding | rows for 80% power (halving) | weeks to power |',
              '|---|---|---:|---|---:|---:|']
        for pop, c in pr['populations'].items():
            for name, sc in c['scenarios'].items():
                L += [f'| {pop} | {name} ({_f(sc["feature_coverage"], 2)}) | {_f(sc["weeks_to_sufficiency_gate"], 1)} | '
                      f'{sc["binding_constraint"]} | {_f(sc["rows_for_80pct_power_halving"])} | {_f(sc["weeks_to_power"], 1)} |']
        L += ['', f'- {pr["caveat"]}']
    return '\n'.join(L) + '\n'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog='python -m evolution_lab.verified_loop', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--table', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, default=None, help='default: <table>.manifest.json')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--md', type=Path, default=None)
    args = ap.parse_args(argv)
    res = run(args.table, args.manifest or args.table.with_suffix('.manifest.json'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
    if args.md:
        args.md.write_text(markdown(res))
    print(f"decision={res['primary']['decision']} analysis_rows={res['input']['analysis_rows']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
