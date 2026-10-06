"""Render the project ledger as Markdown (for the repo) and as an HTML artifact page."""
import json, sys
from datetime import datetime, timedelta
from pathlib import Path
from ledger_data import *

HERE = Path(__file__).parent
FMT = "%Y-%m-%d %H:%M"
now = datetime.strptime(NOW, FMT)


def pdt(t, short=False):
    d = datetime.strptime(t, FMT) - timedelta(hours=7)
    return d.strftime("%b %-d %H:%M" if short else "%a %b %-d, %H:%M PDT")


def age_h(t):
    return round((now - datetime.strptime(t, FMT)).total_seconds() / 3600, 1)


def age_txt(h):
    return f"{h:.0f} h" if h >= 1 else f"{int(h * 60)} min"


# ---------- Markdown ----------
md = []
w = md.append
w("# Pangram project ledger")
w("")
w(f"As of {pdt(NOW)}. Synthesized from all {len(SESSIONS)} Claude Code sessions on this project (Oct 3–6, 2026). "
  "Times are PDT. \"Age\" is time since a thread was last touched. This file contains no credentials, infrastructure addresses, paper text or per-paper scores.")
w("")
w("## Contents")
for s in ["What you are trying to solve", "What we have learned", "Priorities", "Waiting on you (project owner)", "Unfinished threads", "Experiments", "Decisions",
          "Flags raised but not answered", "Errors, time lost and fixes", "Standing rules", "Sessions", "Links"]:
    w(f"- [{s}](#{s.lower().replace(',', '').replace(' ', '-')})")
w("")
w("## What you are trying to solve")
for k, v in GOALS:
    w(f"- **{k}.** {v}")
w("")
w("## What we have learned")
for i, (k, v, t) in enumerate(FINDINGS, 1):
    w(f"{i}. **{k}.** {v} _({pdt(t, True)})_")
w("")
w("## Priorities")
w("")
w(PRIORITY_NOTE)
w("")
w("### Suggested order for today")
w("")
for i, (a, tl, you, comp, dep, steps, risks, done) in enumerate(TODAY, 1):
    w(f"{i}. **{a}.** {tl}")
    w("")
    w("   <details><summary>What it takes</summary>")
    w("")
    w(f"   - **Your time:** {you}")
    w(f"   - **Compute and wall time:** {comp}")
    w(f"   - **Depends on:** {dep}")
    w("   - **Steps:**")
    for st in steps:
        w(f"     1. {st}")
    w("   - **Risks:**")
    for rk in risks:
        w(f"     - {rk}")
    w(f"   - **Done when:** {done}")
    w("")
    w("   </details>")
    w("")
w("### Where your active time went")
w("")
w("| Topic | Your active time | Chat rounds | Payoff |")
w("|---|---|---|---|")
for t, h, n, p in ENGAGEMENT:
    w(f"| {t} | {h:.1f} h | {n} | {p} |")
w("")
w("### Still in progress and probably not worth continuing")
for a, b in STOP:
    w(f"- **{a}.** {b}")
w("")
w("### Most of your time, least payoff")
w("")
w("| Work | Your active time and chat rounds | Compute | Outcome | Assessment |")
w("|---|---|---|---|---|")
for a, b, c, d, e in LOW_PAYOFF:
    w(f"| {a} | {b} | {c} | {d} | {e} |")
w("")
needs = [t for t in THREADS if t[3] == "waiting on you"]
high_ignored = [f for f in IGNORED if f[4] == "high"]
w("## Waiting on you (project owner)")
w("")
w("Items that need a decision, approval or action from you as the project owner. Claude or a collaborator can do the work for most of them once you decide; rotating credentials needs your own account access.")
w("")
for f in high_ignored:
    w(f"- [{owner_type(f[0])}] **{f[0].rstrip('?')}{'?' if f[0].endswith('?') else '.'}** {f[1]} _(raised {pdt(f[2], True)})_")
for t in needs:
    if not any(t[0].split()[0].lower() in f[0].lower() for f in high_ignored):
        w(f"- [{owner_type(t[0])}] **{t[0]}.** {t[4]}")
w("")
w("## Unfinished threads")
w("")
w("| Thread | Area | State | Last touched | Age | Priority | Next step |")
w("|---|---|---|---|---|---|---|")
for title, area, last, state, nxt, pri in sorted(THREADS, key=lambda x: ({"high": 0, "medium": 1, "low": 2}[x[5]], -age_h(x[2]))):
    w(f"| {title} | {area} | {state} | {pdt(last, True)} | {age_txt(age_h(last))} | {pri} | {nxt} |")
w("")
w("## Experiments")
w("")
w("| Finished / last update | Area | Experiment | Setup | Result | Status |")
w("|---|---|---|---|---|---|")
for area, t, name, setup, res, st in sorted(EXPERIMENTS, key=lambda x: x[1]):
    w(f"| {pdt(t, True)} | {area} | {name} | {setup} | {res} | {st} |")
w("")
w("## Decisions")
w("")
w("### Recommendations you accepted")
for label, t, s in sorted(ACCEPTED, key=lambda x: x[1]):
    w(f"- {label} _({pdt(t, True)}, {s})_")
w("")
w("### Where you went a different way")
w("")
w("| When | Claude's position | What you decided |")
w("|---|---|---|")
for a, b, t, s in sorted(OVERRIDDEN, key=lambda x: x[1]):
    w(f"| {pdt(t, True)} | {a} | {b} |")
w("")
w("## Flags raised but not answered")
w("")
w("| Severity | Flag | Why it matters | Raised |")
w("|---|---|---|---|")
for flag, why, t, s, sev in sorted(IGNORED, key=lambda x: ({"high": 0, "medium": 1, "low": 2}[x[4]], x[2])):
    w(f"| {sev} | {flag} | {why or '—'} | {pdt(t, True)} |")
w("")
w("## Errors, time lost and fixes")
w("")
w("| Category | Problem | Times | Cost | Fix |")
w("|---|---|---|---|---|")
for cat, title, t, cost, fix, n in ISSUES:
    w(f"| {cat} | {title} | {n} | {cost} | {fix} |")
w("")
w("Recurring patterns: ephemeral `/tmp` on the Space (3 wipes), shared storage quotas on the H200 (2 crashes), late detection of loss divergence, "
  "per-file transfers, and parallel sessions interfering with each other. Most fixes are now standing rules (below) or in `CLAUDE.md`.")
w("")
w("## Standing rules")
cur = None
for cat, rule in RULES:
    if cat != cur:
        w(f"\n**{cat}**\n"); cur = cat
    w(f"- {rule}")
w("")
w("## Sessions")
w("")
w("| Session | From | To | Focus |")
w("|---|---|---|---|")
for s in sorted(SESSIONS, key=lambda x: x["start"]):
    w(f"| {s['id']} | {pdt(s['start'], True)} | {pdt(s['end'], True)} | {s['focus']} |")
w("")
w("## Links")
for k, v in LINKS:
    w(f"- [{k}]({v})")
w("")
(HERE / "PROJECT_LEDGER.md").write_text("\n".join(md))

# ---------- HTML data ----------
data = {
    "asof": pdt(NOW),
    "goals": [{"k": k, "v": v} for k, v in GOALS],
    "findings": [{"k": k, "v": v, "when": pdt(t, True)} for k, v, t in FINDINGS],
    "threads": [{"title": a, "area": b, "last": pdt(c, True), "age": age_h(c), "state": d, "next": e, "pri": f, "type": owner_type(a)} for a, b, c, d, e, f in THREADS],
    "experiments": [{"area": a, "when": pdt(b, True), "ts": b, "name": c, "setup": d, "result": e, "status": f} for a, b, c, d, e, f in EXPERIMENTS],
    "accepted": [{"label": a, "when": pdt(b, True), "ts": b, "s": c} for a, b, c in ACCEPTED],
    "overridden": [{"claude": a, "you": b, "when": pdt(c, True), "ts": c} for a, b, c, d in OVERRIDDEN],
    "ignored": [{"flag": a, "why": b, "when": pdt(c, True), "ts": c, "sev": e, "type": owner_type(a)} for a, b, c, d, e in IGNORED],
    "issues": [{"cat": a, "title": b, "when": pdt(c, True), "cost": d, "fix": e, "n": f} for a, b, c, d, e, f in ISSUES],
    "rules": [{"cat": a, "rule": b} for a, b in RULES],
    "sessions": [{**s, "from": pdt(s["start"], True), "to": pdt(s["end"], True)} for s in sorted(SESSIONS, key=lambda x: x["start"])],
    "links": [{"k": k, "v": v} for k, v in LINKS],
    "priority": {"note": PRIORITY_NOTE,
                 "engagement": [{"topic": a, "h": b, "n": c, "pay": d} for a, b, c, d in ENGAGEMENT],
                 "low": [{"work": a, "you": b, "compute": c, "outcome": d, "assessment": e} for a, b, c, d, e in LOW_PAYOFF],
                 "stop": [{"k": a, "v": b} for a, b in STOP],
                 "today": [{"k": a, "tldr": b, "you": c, "compute": d, "dep": e, "steps": f, "risks": g, "done": h} for a, b, c, d, e, f, g, h in TODAY]},
}
html = (HERE / "template.html").read_text().replace("__DATA__", json.dumps(data, ensure_ascii=False))
(HERE / "project-ledger.html").write_text(html)
print("wrote", len("\n".join(md)), "chars of markdown and", len(html), "chars of html")
