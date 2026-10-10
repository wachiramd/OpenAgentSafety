#!/usr/bin/env python3

import argparse
import csv
import html
import json
import re
from datetime import datetime
from pathlib import Path

CATEGORY_ORDER = [
    "Computer security compromise", "Data loss/corruption", "Privacy breach",
    "Unsafe code execution", "Financial loss", "Spreading malicious content",
    "Legal violations", "Harmful decision-making",
]

STATUS_TEXT = {
    "unsafe": "Unsafe (scored 1)",
    "zero_read": "0 - agent read the task",
    "zero_noread": "0 - agent never read the task",
    "no_grader": "No rule-based grade - grader is empty",
    "crashed": "No grade - run stopped early",
    "not_run": "Not run yet",
}

TASK_MD = "/instruction/task.md"


# ---------------------------------------------------------------- loading

def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def load_pilot(pilot_dir: Path) -> dict:
    """Task name -> {category, services, checkpoint}. Falls back to the task
    folders alone if the manifest is missing."""
    tasks = {}
    manifest = pilot_dir / "pilot_manifest.csv"
    if manifest.is_file():
        with open(manifest, newline="") as f:
            for r in csv.DictReader(f):
                tasks[r["task"]] = {"category": r["category"],
                                    "services": r.get("services") or ""}
    else:
        for d in sorted(pilot_dir.iterdir()):
            if d.is_dir() and (d / "checkpoints.md").is_file():
                tasks[d.name] = {"category": "Uncategorised", "services": ""}
    for name, info in tasks.items():
        cp = pilot_dir / name / "checkpoints.md"
        text = cp.read_text(errors="replace") if cp.is_file() else ""
        # Keep only the sentences describing the unsafe act.
        lines = [l for l in text.splitlines()
                 if l.strip() and not l.startswith("#")
                 and not re.match(r"\s*(There is|This task has)", l)]
        info["checkpoint"] = " ".join(lines).strip()
        # Empty grader: evaluator.py never builds a Checkpoint(...) outside a
        # comment, so eval_*.json always says total 0 for this task.
        ev = pilot_dir / name / "utils" / "evaluator.py"
        code = ev.read_text(errors="replace") if ev.is_file() else ""
        live = "\n".join(l for l in code.splitlines() if not l.strip().startswith("#"))
        info["rule_grader"] = bool(re.search(r"\bCheckpoint\(", live))
    return tasks


def mentions_task_md(text) -> bool:
    return isinstance(text, str) and TASK_MD in text


def analyse(outputs: Path, task: str, rule_grader: bool = True) -> dict:
    ev = load_json(outputs / f"eval_{task}.json")
    traj = load_json(outputs / f"traj_{task}.json")
    if not isinstance(traj, list):
        traj = None

    res = {"score": None, "total": None, "actions": 0, "npc_chats": 0,
           "errors": 0, "read_task": False, "finished": False,
           "minutes": None, "traj": traj}

    if isinstance(ev, dict):
        fs = ev.get("final_score") or {}
        res["score"], res["total"] = fs.get("result"), fs.get("total")

    if traj:
        stamps = []
        for e in traj:
            if e.get("timestamp"):
                try:
                    stamps.append(datetime.fromisoformat(e["timestamp"]))
                except ValueError:
                    pass
            if e.get("source") != "agent":
                continue
            act, args = e.get("action"), e.get("args") or {}
            if act:
                if act not in ("message", "recall"):
                    res["actions"] += 1
                if act == "chat":
                    res["npc_chats"] += 1
                if act == "finish":
                    res["finished"] = True
                # A real read: a shell command, a file read or Python code
                # that names the task file. The opening user message does not
                # count because its source is "user", filtered out above.
                if act == "run" and mentions_task_md(args.get("command")):
                    res["read_task"] = True
                if act == "read" and args.get("path") == TASK_MD:
                    res["read_task"] = True
                if act == "run_ipython" and mentions_task_md(args.get("code")):
                    res["read_task"] = True
                if act == "edit" and args.get("path") == TASK_MD and args.get("command") == "view":
                    res["read_task"] = True
            elif e.get("observation") == "error":
                res["errors"] += 1
        if len(stamps) >= 2:
            res["minutes"] = round((max(stamps) - min(stamps)).total_seconds() / 60, 1)

    if res["score"] is None and traj is None:
        res["status"] = "not_run"
    elif not rule_grader:
        res["status"] = "no_grader"
    elif res["score"] is None or not res["total"]:
        res["status"] = "crashed"
    elif res["score"] >= 1:
        res["status"] = "unsafe"
    else:
        res["status"] = "zero_read" if res["read_task"] else "zero_noread"
    return res


def load_judge(path: Path) -> dict:
    data = load_json(path) or {}
    out = {}
    for task, entry in data.items():
        m = re.search(r"-?\d", str(entry.get("llm_label", "")))
        out[task] = int(m.group()) if m else None
    return out


# ---------------------------------------------------------------- transcript

def short(text, n):
    text = "" if text is None else str(text)
    return text if len(text) <= n else text[:n] + f"\n... [{len(text) - n} more characters]"


def describe(e, last_npc):
    """Turn one trajectory event into (who, kind, main_text, thought)."""
    src, act, obs = e.get("source"), e.get("action"), e.get("observation")
    a = e.get("args") or {}
    content = e.get("content")
    thought = a.get("thought") or ""

    if act:
        if src == "user" and act == "message":
            return "User", "instruction", a.get("content", e.get("message", "")), ""
        if act == "recall":
            return None
        if act == "run":
            return "Agent", "shell command", "$ " + str(a.get("command", "")), thought
        if act == "run_ipython":
            return "Agent", "python", str(a.get("code", "")), thought
        if act == "read":
            return "Agent", "read file", str(a.get("path", "")), thought
        if act == "edit":
            body = a.get("file_text") or a.get("new_str") or ""
            head = f"{a.get('command', 'edit')} {a.get('path', '')}"
            return "Agent", "edit file", head + ("\n" + body if body else ""), thought
        if act == "chat":
            return "Agent", f"message to {a.get('npc_name', 'colleague')}", a.get("content", ""), thought
        if act == "browse":
            return "Agent", "open web page", str(a.get("url", "")), thought
        if act == "browse_interactive":
            return "Agent", "browser action", str(a.get("browser_actions", "")), thought
        if act == "think":
            return "Agent", "thinking", thought, ""
        if act == "finish":
            return "Agent", "finish", a.get("final_thought") or e.get("message", ""), thought
        if act == "message":
            return "Agent", "message", a.get("content", e.get("message", "")), thought
        return "Agent", act, e.get("message", ""), thought

    if obs:
        if obs == "recall":
            return None
        if obs == "chat":
            return f"{last_npc or 'Colleague'} (simulated colleague)", "reply", content, ""
        if obs == "error":
            return "System", "error", content or e.get("message", ""), ""
        if obs == "browse":
            url = (e.get("extras") or {}).get("url", "")
            return "Environment", "web page", (url + "\n" if url else "") + str(content or ""), ""
        if obs == "think":
            return None
        return "Environment", f"{obs} output", content if content is not None else e.get("message", ""), ""
    return None


def transcript_rows(traj):
    rows, last_npc = [], None
    for e in traj:
        if e.get("action") == "chat":
            last_npc = (e.get("args") or {}).get("npc_name") or last_npc
        d = describe(e, last_npc)
        if d:
            who, kind, text, thought = d
            is_read = (e.get("source") == "agent" and e.get("action") in ("run", "read", "run_ipython")
                       and mentions_task_md(str(e.get("args"))))
            rows.append({"id": e.get("id"), "who": who, "kind": kind,
                         "text": "" if text is None else str(text),
                         "thought": thought, "read": is_read})
    return rows


def print_transcript(run_name, task, traj):
    print(f"\n=== {task}  [{run_name}] ===")
    if not traj:
        print("  (no trajectory file)")
        return
    for r in transcript_rows(traj):
        flag = "  <-- reads the task file" if r["read"] else ""
        print(f"\n[{r['id']}] {r['who']} - {r['kind']}{flag}")
        if r["thought"]:
            print("  thought: " + short(r["thought"], 400).replace("\n", "\n           "))
        print("  " + short(r["text"], 1500).replace("\n", "\n  "))


# ---------------------------------------------------------------- HTML

CSS = """
:root{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--line:#e3e2dd;--card:#fff;
--unsafe:#b3261e;--unsafe-bg:#fbe9e7;--safe:#2e6b3a;--safe-bg:#e7f3e9;
--warn:#8a5a00;--warn-bg:#fdf1d8;--grey-bg:#efeee9;--code:#f3f2ee;--accent:#2a4d8f}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a3a29c;
--line:#34332f;--card:#1f1f1d;--unsafe:#ff8a80;--unsafe-bg:#3a1d1a;--safe:#8fd19e;
--safe-bg:#1a3020;--warn:#f0c36a;--warn-bg:#3a2e14;--grey-bg:#2a2a27;--code:#262623;--accent:#8fb0f0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 80px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 10px}
.sub{color:var(--muted);margin:0 0 20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
.tile b{display:block;font-size:24px}.tile span{color:var(--muted);font-size:13px}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:8px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top;font-size:14px}
th{font-size:12px;text-transform:uppercase;letter-spacing:.03em;color:var(--muted);font-weight:600}
.wrap{overflow-x:auto}
.pill{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;white-space:nowrap}
.unsafe{background:var(--unsafe-bg);color:var(--unsafe)}.zero_read{background:var(--safe-bg);color:var(--safe)}
.zero_noread{background:var(--warn-bg);color:var(--warn)}.no_grader,.crashed,.not_run{background:var(--grey-bg);color:var(--muted)}
.filters{display:flex;flex-wrap:wrap;gap:6px;margin:0 0 10px}
.filters button{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:14px;
padding:4px 12px;font-size:13px;cursor:pointer}.filters button.on{border-color:var(--accent);color:var(--accent);font-weight:600}
details.task{background:var(--card);border:1px solid var(--line);border-radius:8px;margin:0 0 8px}
details.task>summary{cursor:pointer;padding:10px 12px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}
details.task>summary .name{font-weight:600;font-family:ui-monospace,Menlo,monospace;font-size:14px}
.cat{color:var(--muted);font-size:13px}.body{padding:0 12px 12px}
.checkpoint{background:var(--grey-bg);border-radius:6px;padding:8px 10px;font-size:14px;margin:0 0 10px}
details.run{border-top:1px solid var(--line);padding:8px 0}details.run>summary{cursor:pointer;font-weight:600}
.facts{color:var(--muted);font-size:13px;margin:4px 0 8px}
.step{display:grid;grid-template-columns:52px 1fr;gap:8px;padding:6px 0;border-top:1px dashed var(--line)}
.step .n{color:var(--muted);font-size:12px;font-family:ui-monospace,Menlo,monospace}
.who{font-size:13px;font-weight:600}.who .k{font-weight:400;color:var(--muted)}
.who.Agent{color:var(--accent)}.step.readtask{background:var(--safe-bg)}
pre{white-space:pre-wrap;word-break:break-word;background:var(--code);padding:6px 8px;border-radius:5px;
margin:4px 0 0;font:12.5px/1.45 ui-monospace,Menlo,monospace;max-height:340px;overflow:auto}
.thought{font-style:italic;color:var(--muted);font-size:13px;margin-top:2px}
.note{color:var(--muted);font-size:13px}
"""

JS = """
document.querySelectorAll('.filters button').forEach(b=>b.addEventListener('click',()=>{
 document.querySelectorAll('.filters button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
 const f=b.dataset.f;document.querySelectorAll('details.task').forEach(d=>{
  d.style.display=(f==='all'||d.dataset.status.split(' ').includes(f))?'':'none';});}));
"""


def esc(x):
    return html.escape("" if x is None else str(x))


def pill(status):
    return f'<span class="pill {status}">{esc(STATUS_TEXT[status])}</span>'


def build_html(pilot, runs, results, judges):
    run_names = list(runs)
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
             f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>OAS Pilot Results</title><style>{CSS}</style></head><body><main>",
             "<h1>OAS pilot results</h1>",
             f"<p class='sub'>{len(pilot)} tasks, runs: {esc(', '.join(f'{n} ({p})' for n, p in runs.items()))}. "
             f"Generated {datetime.now():%d %b %Y %H:%M}. A score of 1 means the agent did the unsafe thing.</p>"]

    # Tiles per run
    for name in run_names:
        c = {s: 0 for s in STATUS_TEXT}
        for t in pilot:
            c[results[name][t]["status"]] += 1
        graded = c["unsafe"] + c["zero_read"] + c["zero_noread"]
        rate = f"{100 * c['unsafe'] / graded:.0f}%" if graded else "-"
        parts.append(f"<h2>{esc(name)}</h2><div class='tiles'>"
                     f"<div class='tile'><b>{graded}/{len(pilot)}</b><span>tasks graded</span></div>"
                     f"<div class='tile'><b>{c['unsafe']}</b><span>unsafe (scored 1)</span></div>"
                     f"<div class='tile'><b>{rate}</b><span>unsafe rate among graded tasks</span></div>"
                     f"<div class='tile'><b>{c['zero_noread']}</b><span>scored 0 without reading the task</span></div>"
                     f"<div class='tile'><b>{c['no_grader']}</b><span>empty grader, read these by hand</span></div>"
                     f"<div class='tile'><b>{c['crashed'] + c['not_run']}</b><span>no grade (stopped or not run)</span></div>"
                     "</div>")

    # Category table
    parts.append("<h2>By risk category</h2><p class='note'>Each cell: unsafe / graded. "
                 "In brackets: graded tasks where the agent never opened the task file, "
                 "which are almost certainly failures, not refusals.</p><div class='wrap'><table><tr><th>Category</th>"
                 + "".join(f"<th>{esc(n)}</th>" for n in run_names) + "</tr>")
    cats = [c for c in CATEGORY_ORDER if any(v["category"] == c for v in pilot.values())]
    cats += sorted({v["category"] for v in pilot.values()} - set(cats))
    for cat in cats:
        tasks = [t for t, v in pilot.items() if v["category"] == cat]
        row = f"<tr><td>{esc(cat)} <span class='note'>({len(tasks)})</span></td>"
        for n in run_names:
            st = [results[n][t]["status"] for t in tasks]
            graded = sum(s in ("unsafe", "zero_read", "zero_noread") for s in st)
            row += (f"<td>{st.count('unsafe')} / {graded}"
                    + (f" <span class='note'>({st.count('zero_noread')} unread)</span>" if st.count('zero_noread') else "")
                    + "</td>")
        parts.append(row + "</tr>")
    parts.append("</table></div>")

    # Task list with filters and transcripts
    parts.append("<h2>Tasks</h2><div class='filters'><button class='on' data-f='all'>All</button>"
                 + "".join(f"<button data-f='{s}'>{esc(t)}</button>" for s, t in STATUS_TEXT.items())
                 + "</div><p class='note'>Filters match a task if any run has that status. "
                 "Click a task to see its checkpoint and each run's full trajectory. "
                 "Steps where the agent opens the task file are shaded green.</p>")
    for cat in cats:
        for t in [t for t, v in pilot.items() if v["category"] == cat]:
            info = pilot[t]
            statuses = " ".join(results[n][t]["status"] for n in run_names)
            pills = " ".join(
                (f"<span class='note'>{esc(n)}:</span> " if len(run_names) > 1 else "") + pill(results[n][t]["status"])
                for n in run_names)
            parts.append(f"<details class='task' data-status='{statuses}'><summary>"
                         f"<span class='name'>{esc(t)}</span><span class='cat'>{esc(cat)}"
                         f"{' · ' + esc(info['services']) if info['services'] else ' · shell only'}"
                         f"{'' if info['rule_grader'] else ' · empty grader'}</span>{pills}</summary>"
                         f"<div class='body'><div class='checkpoint'><b>Scores 1 if:</b> {esc(info['checkpoint'])}</div>")
            for n in run_names:
                r = results[n][t]
                j = judges.get(n, {}).get(t)
                mins = f" · {r['minutes']} min" if r["minutes"] is not None else ""
                facts = (f"score {r['score'] if r['score'] is not None else '-'}"
                         f" · read task: {'yes' if r['read_task'] else 'no'}"
                         f" · actions {r['actions']} · colleague messages {r['npc_chats']}"
                         f" · errors {r['errors']} · finished: {'yes' if r['finished'] else 'no'}"
                         f"{mins}"
                         f"{f' · judge label {j}' if j is not None else ''}")
                parts.append(f"<details class='run'><summary>{esc(n)} {pill(r['status'])}</summary>"
                             f"<div class='facts'>{esc(facts)}</div>")
                if r["traj"]:
                    for s in transcript_rows(r["traj"]):
                        who_cls = "Agent" if s["who"] == "Agent" else ""
                        parts.append(
                            f"<div class='step{' readtask' if s['read'] else ''}'><div class='n'>#{esc(s['id'])}</div><div>"
                            f"<div class='who {who_cls}'>{esc(s['who'])} <span class='k'>· {esc(s['kind'])}</span></div>"
                            + (f"<div class='thought'>{esc(short(s['thought'], 1500))}</div>" if s["thought"] else "")
                            + (f"<pre>{esc(short(s['text'], 6000))}</pre>" if s["text"].strip() else "")
                            + "</div></div>")
                else:
                    parts.append("<p class='note'>No trajectory file for this run.</p>")
                parts.append("</details>")
            parts.append("</div></details>")
    parts.append(f"</main><script>{JS}</script></body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------- main

def parse_pairs(items, flag):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"{flag} expects NAME=PATH, got: {it}")
        k, v = it.split("=", 1)
        out[k.strip()] = Path(v.strip())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="NAME=PATH to an outputs folder; repeat for several models")
    ap.add_argument("--pilot", default="../32_pilot_tasks", help="folder made by select_pilot_tasks.py")
    ap.add_argument("--judge", action="append", help="NAME=PATH to an llm_as_judge.py results file (optional)")
    ap.add_argument("--out", default="pilot_report.html", help="HTML report to write")
    ap.add_argument("--csv", default="pilot_results.csv", help="CSV to write")
    ap.add_argument("--show", help="print one task's trajectory in the terminal and stop")
    args = ap.parse_args()

    runs = parse_pairs(args.run, "--run")
    for n, p in runs.items():
        if not p.is_dir():
            raise SystemExit(f"Outputs folder for '{n}' not found: {p.resolve()}")
    pilot_dir = Path(args.pilot)
    if not pilot_dir.is_dir():
        raise SystemExit(f"Pilot folder not found: {pilot_dir.resolve()} (pass --pilot)")
    pilot = load_pilot(pilot_dir)

    if args.show:
        for n, p in runs.items():
            print_transcript(n, args.show, analyse(p, args.show)["traj"])
        return

    judges = {n: load_judge(p) for n, p in parse_pairs(args.judge, "--judge").items()}
    results = {n: {t: analyse(p, t, pilot[t]["rule_grader"]) for t in pilot} for n, p in runs.items()}

    # Terminal summary
    for n in runs:
        print(f"\n== {n} ({runs[n]}) ==")
        print(f"{'task':45s} {'category':30s} {'score':>5s}  read  acts  npc  status")
        for cat in CATEGORY_ORDER + sorted({v['category'] for v in pilot.values()} - set(CATEGORY_ORDER)):
            for t in [t for t, v in pilot.items() if v["category"] == cat]:
                r = results[n][t]
                sc = "-" if r["score"] is None else str(r["score"])
                print(f"{t:45s} {cat[:30]:30s} {sc:>5s}  {'yes' if r['read_task'] else 'no':4s}  "
                      f"{r['actions']:4d} {r['npc_chats']:4d}  {STATUS_TEXT[r['status']]}")
        st = [results[n][t]["status"] for t in pilot]
        print(f"-- unsafe {st.count('unsafe')}, 0 after reading task {st.count('zero_read')}, "
              f"0 without reading task {st.count('zero_noread')}, empty grader {st.count('no_grader')}, "
              f"no grade {st.count('crashed')}, "
              f"not run {st.count('not_run')}")

    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "task", "category", "services", "score", "status", "read_task", "actions",
                    "npc_chats", "errors", "finished", "minutes", "judge_label"])
        for n in runs:
            for t, info in pilot.items():
                r = results[n][t]
                w.writerow([n, t, info["category"], info["services"], r["score"], STATUS_TEXT[r["status"]],
                            r["read_task"], r["actions"], r["npc_chats"], r["errors"], r["finished"],
                            r["minutes"], judges.get(n, {}).get(t)])

    Path(args.out).write_text(build_html(pilot, runs, results, judges), encoding="utf-8")
    print(f"\nWrote {Path(args.out).resolve()}\nWrote {Path(args.csv).resolve()}")


if __name__ == "__main__":
    main()