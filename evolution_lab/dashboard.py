"""Frontier + MAP-Elites dashboard from archive JSONL (no empty charts)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import html as htmlmod
import json

from .engine import summarize
from .pareto import Point, pareto_front
from .engine import points_from_archive


def render_dashboard(rows: list[dict[str, Any]]) -> str:
    summary = summarize(rows)
    pts = points_from_archive(rows)
    learned_ids = set(summary.get("front_learned") or [])
    front_ids = {p.experiment_id for p in pareto_front(pts)}
    # SVG scatter: x=cost, y=success
    W, H, pad = 640, 360, 40
    dots = []
    for p in pts:
        x = pad + (1.0 - min(p.cost, 1.0)) * (W - 2 * pad)
        y = H - pad - p.success * (H - 2 * pad)
        fill = "#f5c542" if p.experiment_id in front_ids else ("#3fb950" if p.experiment_id in learned_ids else "#7aa2f7")
        r = 6 if p.experiment_id in front_ids else 4
        dots.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}">'
            f'<title>{htmlmod.escape(p.experiment_id)} success={p.success:.3f} cost={p.cost:.3f}</title></circle>'
        )
    table_rows = []
    for r in rows:
        m = r.get("metrics") or {}
        table_rows.append(
            "<tr>"
            f"<td>{htmlmod.escape(str(r.get('experiment_id')))}</td>"
            f"<td>{htmlmod.escape(str(r.get('status')))}</td>"
            f"<td>{htmlmod.escape(str(r.get('role')))}</td>"
            f"<td>{m.get('success_rate', '')}</td>"
            f"<td>{m.get('ood_score', '')}</td>"
            f"<td>{m.get('params', '')}</td>"
            f"<td>{m.get('delta_hv', r.get('delta_hv'))}</td>"
            f"<td>{htmlmod.escape(str(r.get('epistemic')))}</td>"
            "</tr>"
        )
    dots_join = "\n".join(dots) if dots else '<text x="320" y="180" fill="#888" text-anchor="middle">no ok experiments yet</text>'
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Evolution Lab — Frontier</title>
<style>
body {{ font-family: ui-sans-serif, system-ui, sans-serif; background:#0e1116; color:#e6edf3; margin:2rem; }}
h1 {{ font-size:1.4rem; }}
.goal {{ color:#9aa4b2; max-width:52rem; }}
svg {{ background:#161b22; border:1px solid #30363d; border-radius:8px; }}
table {{ border-collapse:collapse; margin-top:1.5rem; width:100%; font-size:0.85rem; }}
th,td {{ border-bottom:1px solid #30363d; padding:0.4rem 0.5rem; text-align:left; }}
.star {{ color:#f5c542; }}
</style></head><body>
<h1>Evolution Lab</h1>
<p class="goal">Continuously discover, verify, and explain computational systems that expand the achievable frontier between capability and resources.</p>
<p>n={summary['n']} ok={summary['ok']} HV={summary['hypervolume_2d']:.5f} HV(learned)={summary.get('hypervolume_2d_learned', 0):.5f}</p>
<p>all-front={', '.join(summary['front']) or '—'} · learned-front={', '.join(summary.get('front_learned') or []) or '—'}</p>
<p class="star">Gold = all-systems 2-D front (often the free teacher). Green = learned-only front. Empty gold means no successful runs imported.</p>
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <text x="{pad}" y="18" fill="#9aa4b2" font-size="12">success ↑</text>
  <text x="{W-120}" y="{H-8}" fill="#9aa4b2" font-size="12">cheaper →</text>
  {dots_join}
</svg>
<table>
<thead><tr><th>id</th><th>status</th><th>role</th><th>success</th><th>ood</th><th>params</th><th>ΔHV</th><th>epistemic</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
<script type="application/json" id="archive">{json.dumps(rows, default=str)}</script>
</body></html>
"""


def write_dashboard(rows: list[dict[str, Any]], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(render_dashboard(rows), encoding="utf-8")
    return dest
