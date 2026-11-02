"""McNemar exact test, scenario-cluster bootstrap, charts, summary (§9)."""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

from pt_cost import summarize as cost_summary
from pt_prompt import ROOT
from pt_score import by_modality

CHARTS = ROOT / "results" / "charts"


def _paired(a_rows, b_rows, field):
    a = {r["id"]: r for r in a_rows if field in r}
    b = {r["id"]: r for r in b_rows if field in r}
    ids = sorted(a.keys() & b.keys())
    return ids, a, b


def mcnemar(a_rows: list[dict], b_rows: list[dict], field: str = "urgency_correct") -> dict:
    """a = reference (base), b = candidate. b_only = rows b got right and a got wrong."""
    ids, a, b = _paired(a_rows, b_rows, field)
    b_only = sum(1 for i in ids if b[i][field] and not a[i][field])
    a_only = sum(1 for i in ids if a[i][field] and not b[i][field])
    n = b_only + a_only
    p = binomtest(b_only, n, 0.5).pvalue if n else 1.0
    return {"field": field, "n_pairs": len(ids), "b_only": b_only, "a_only": a_only, "p_exact": float(p)}


def cluster_bootstrap(a_rows: list[dict], b_rows: list[dict], field: str, cluster: str = "scenario_id",
                      reps: int = 5000, seed: int = 0, rate_filter=None) -> dict:
    """Resample scenarios with replacement, carrying all their rows. Gap = mean(b) - mean(a)."""
    ids, a, b = _paired(a_rows, b_rows, field)
    if rate_filter is not None:
        ids = [i for i in ids if rate_filter(a[i])]
    if not ids:
        return {"field": field, "mean_gap": None, "ci_low": None, "ci_high": None, "frac_positive": None}
    clusters = defaultdict(list)
    for i in ids:
        clusters[a[i][cluster]].append(i)
    keys = sorted(clusters)
    sa = np.array([sum(float(a[i][field]) for i in clusters[k]) for k in keys])
    sb = np.array([sum(float(b[i][field]) for i in clusters[k]) for k in keys])
    cnt = np.array([len(clusters[k]) for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(reps, len(keys)))
    gaps = (sb[idx].sum(1) - sa[idx].sum(1)) / cnt[idx].sum(1)
    point = (sb.sum() - sa.sum()) / cnt.sum()
    return {"field": field, "n_clusters": len(keys), "n_rows": len(ids), "mean_gap": float(point),
            "ci_low": float(np.percentile(gaps, 2.5)), "ci_high": float(np.percentile(gaps, 97.5)),
            "frac_positive": float((gaps > 0).mean())}


def is_real(mc: dict, bs: dict, higher_is_better: bool = True) -> bool:
    if bs["mean_gap"] is None:
        return False
    excl = bs["ci_low"] > 0 if higher_is_better else bs["ci_high"] < 0
    return mc["p_exact"] < 0.05 and excl


def compare_all(systems: dict[str, list[dict]], ref: str = "base", modality_sets=("text", "image", "all")) -> dict:
    out = {}
    base = systems[ref]
    for name, rows in systems.items():
        from pt_cost import deployment_usd_per_1k
        cost = cost_summary(rows)
        if name.startswith("sft"):
            dep = deployment_usd_per_1k(name)
            if dep:
                cost = {**cost, **dep, "usd_per_1k_chats_token_price": cost["usd_per_1k_chats"],
                        "usd_per_1k_chats": round(dep["usd_per_1k_chats_dedicated"] + (sum(r.get("router_cost_usd", 0) for r in rows) / len(rows) * 1000), 4)}
        entry = {"metrics": by_modality(rows), "cost": cost, "vs_base": {}}
        if name != ref:
            for mod in modality_sets:
                pick = (lambda r: r["modality"] in ("text", "image")) if mod == "all" else (lambda r, m=mod: r["modality"] == m)
                a, b = [r for r in base if pick(r)], [r for r in rows if pick(r)]
                if not a or not b:
                    continue
                fields = ["urgency_correct", "cond_top1_hit"] + (["image_finding_correct"] if mod == "image" else [])
                res = {}
                for f in fields:
                    mc = mcnemar(a, b, f)
                    bs = cluster_bootstrap(a, b, f)
                    res[f] = {"mcnemar": mc, "bootstrap": bs, "real_gain": is_real(mc, bs)}
                ut = cluster_bootstrap(a, b, "under_triage",
                                       rate_filter=lambda r: r["gold_urgency"] in ("emergency_now", "vet_today"))
                res["under_triage"] = {"bootstrap": ut}
                for f in ("harmful_advice", "plan_recall", "cond_set_f1"):
                    res[f] = {"bootstrap": cluster_bootstrap(a, b, f)}
                entry["vs_base"][mod] = res
            # Video: numbers + wide interval only, never a p-value claim.
            va, vb = [r for r in base if r["modality"] == "video"], [r for r in rows if r["modality"] == "video"]
            if va and vb:
                entry["vs_base"]["video"] = {"n": len(vb), "note": "n too small for significance; no p-value claim",
                                             "urgency_correct": {"bootstrap": cluster_bootstrap(va, vb, "urgency_correct")}}
        out[name] = entry
    return out


# ------------------------------------------------------------------ charts
def charts(summary: dict, ref: str = "base", field: str = "urgency_correct") -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    CHARTS.mkdir(parents=True, exist_ok=True)
    names = [n for n in summary if n != ref and summary[n]["vs_base"]]
    mods = [m for m in ("text", "image", "all") if any(m in summary[n]["vs_base"] for n in names)]
    paths = []

    # (1) discordant-pair bars
    fig, axes = plt.subplots(1, len(mods), figsize=(5 * len(mods), 3.6), squeeze=False)
    for ax, m in zip(axes[0], mods):
        ns = [n for n in names if m in summary[n]["vs_base"]]
        bo = [summary[n]["vs_base"][m][field]["mcnemar"]["b_only"] for n in ns]
        ao = [summary[n]["vs_base"][m][field]["mcnemar"]["a_only"] for n in ns]
        y = np.arange(len(ns))
        ax.barh(y + 0.2, bo, 0.4, color="#2a9d8f", label="system right, base wrong")
        ax.barh(y - 0.2, ao, 0.4, color="#e76f51", label="base right, system wrong")
        for yi, n in zip(y, ns):
            p = summary[n]["vs_base"][m][field]["mcnemar"]["p_exact"]
            ax.text(max(bo + ao + [1]) * 1.02, yi, f"p={p:.2g}", va="center", fontsize=8)
        ax.set_yticks(y, ns)
        ax.set_title(f"Discordant pairs vs {ref} ({m}, {field})", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    p1 = CHARTS / "mcnemar.png"
    fig.savefig(p1, dpi=160)
    plt.close(fig)
    paths.append(str(p1))

    # (2) bootstrap CI dot-and-whisker
    fig, axes = plt.subplots(1, len(mods), figsize=(5 * len(mods), 3.6), squeeze=False, sharex=True)
    for ax, m in zip(axes[0], mods):
        ns = [n for n in names if m in summary[n]["vs_base"]]
        bs = [summary[n]["vs_base"][m][field]["bootstrap"] for n in ns]
        y = np.arange(len(ns))
        g = np.array([b["mean_gap"] for b in bs]) * 100
        lo = g - np.array([b["ci_low"] for b in bs]) * 100
        hi = np.array([b["ci_high"] for b in bs]) * 100 - g
        ax.errorbar(g, y, xerr=[lo, hi], fmt="o", color="#264653", capsize=4)
        ax.axvline(0, color="#999", lw=1, ls="--")
        ax.set_yticks(y, ns)
        ax.set_xlabel("gap vs base (pp), 95% cluster-bootstrap CI")
        ax.set_title(f"{m} rows", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p2 = CHARTS / "bootstrap.png"
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    paths.append(str(p2))
    return paths


def markdown_table(summary: dict, mod: str = "all") -> str:
    hdr = ("| system | urgency acc | under-triage | harmful advice | top-1 cond | plan recall | $/1k chats "
           "| p95 ms | McNemar p | bootstrap 95% CI (pp) | real gain |")
    lines = [hdr, "|" + "---|" * 11]
    for name, e in summary.items():
        m = e["metrics"].get(mod, {})
        if not m.get("n"):
            continue
        vb = e["vs_base"].get(mod, {}).get("urgency_correct")
        p = f"{vb['mcnemar']['p_exact']:.2g}" if vb else "—"
        ci = (f"{vb['bootstrap']['mean_gap']*100:+.1f} [{vb['bootstrap']['ci_low']*100:+.1f}, "
              f"{vb['bootstrap']['ci_high']*100:+.1f}]") if vb else "—"
        real = ("yes" if vb["real_gain"] else "no") if vb else "—"
        c = e["cost"]

        def pct(k):
            return f"{m[k]*100:.1f}%" if k in m else "—"
        lines.append(f"| {name} | {pct('urgency_correct')} | {pct('under_triage')} | {pct('harmful_advice')} | "
                     f"{pct('cond_top1_hit')} | {pct('plan_recall')} | ${c.get('usd_per_1k_chats', 0):.3f} | "
                     f"{c.get('p95_latency_ms')} | {p} | {ci} | {real} |")
    return "\n".join(lines)


def write_summary(summary: dict) -> None:
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=1))
