#!/usr/bin/env python3
"""
This script shows us how every task run ended and why it ended.

It reads the traj_<task>.json, the state_<task>.json and the eval_<task>.json

PER Task, it prints the following:

  ending    how the run stopped (see ENDINGS below)
  read      did the agent really open /instruction/task.md? Checked against
            actual command / file-read actions only, never the opening message
            (that message contains the path and would give a false positive)
  acts      number of agent actions
  chats     number of messages the agent sent to a simulated colleague
  errs      number of error observations
  score     result/total from eval_<task>.json, or "none" if missing
  clue      the piece of evidence the ending was decided from

ENDINGS (decided in this order; the first that fits wins)
  NO_TRAJ    no traj file at all: the run died before saving anything
  FINISHED   the agent issued a "finish" action: it decided it was done
  STUCK      OpenHands' loop detector stopped it (repeating the same thing)
  STEP_LIMIT the maximum number of steps ran out
  CONTEXT    the conversation grew too long for the model's context window
  LLM_ERROR  a call to the model failed (Ollama/LiteLLM error, timeout)
  ERROR      some other error ended the run
  UNKNOWN    none of the above matched; read the trajectory with --show

  
"""

import argparse, csv, glob, os, re, json

PATTERNS = [
    ("STUCK", r"stuck|agentstuckinlooperror|loop detected"),
    ("STEP_LIMIT", r"maximum iteration|max_iterations|reached the maximum|iteration limit"),
    ("CONTEXT", r"context ?window|contextwindowexceeded| context length| maxkimum context| toomany tokens| prompt is too long"),
    ("LLM_ERROR", r"litellm|apiconnectionerror|timeout|timed out| ratelimit|internatl server error| ollama|llm.*error|bad ?request"),
]

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

def read_raw(path):
    try:
        with open(path, errors="replace") as f:
            return f.read()
    except Exception:
        return ""

def event_kind(e):
    if "action" in e:
        return "action:" + str(e["action"])
    if "observation" in e:
        return"obs:" + str(e["observation"])
    return "other"

def event_text(e):
    a = e.get("args") or {}
    if isinstance(a, dict):
        for k in ("command", "code", "path", "url", "content", "thought", "final_thought"):
            if a.get(k):
                return str(a[k])
    return str(e.get("content") or e.get("message") or "")

def is_agent_action(e):
    return e.get("source") == "agent" and "action" in e

def really_read_task(traj):
    # Only actions that can actually open a file count.
    for e in traj:
        if not is_agent_action(e):
            continue
        if e["action"] in ("run", "read", "edit", "run_ipython", "str_replace_editor"):
            a = e.get("args") or {}
            blob = " ".join(str(a.get(k, "")) for k in ("command", "path", "code"))
            if "/instruction" in blob:
                return True
    return False


def count_chats(traj):
    n = 0
    for e in traj:
        if is_agent_action(e) and ("chat" in str(e["action"]) or "npc" in json.dumps(e.get("args", {})).lower()):
            n += 1
    return n


def score_of(folder, task):
    d = load_json(os.path.join(folder, f"eval_{task}.json"))
    if not d:
        return "none"
    fs = d.get("final_score", d)
    return f"{fs.get('result', '?')}/{fs.get('total', '?')}"


def classify(traj, state_raw):
    if traj is None:
        return "NO_TRAJ", "traj file missing or unreadable"

    finishes = [e for e in traj if is_agent_action(e) and e["action"] == "finish"]
    if finishes:
        return "FINISHED", "finish action: " + event_text(finishes[-1])[:100]

    # Evidence: the state file plus the text of every error observation.
    errors = [event_text(e) for e in traj if e.get("observation") == "error"]
    evidence = (state_raw + "\n" + "\n".join(errors)).lower()

    for label, rx in PATTERNS:
        m = re.search(rx, evidence)
        if m:
            s = max(0, m.start() - 60)
            return label, evidence[s:m.end() + 60].replace("\n", " ")

    m = re.search(r"last_error['\"]?\s*[:=]\s*['\"]?([^'\",\n]{3,})", state_raw, re.I)
    if m:
        return "ERROR", "last_error: " + m.group(1)[:120]
    if errors:
        return "ERROR", "last error event: " + errors[-1][:120]

    last = traj[-1] if traj else {}
    return "UNKNOWN", f"last event {event_kind(last)}: {event_text(last)[:100]}"


def analyse_folder(folder):
    tasks = set()
    for p in glob.glob(os.path.join(folder, "*_*.json")):
        name = os.path.basename(p)
        for pre in ("traj_", "eval_", "state_"):
            if name.startswith(pre):
                tasks.add(name[len(pre):-5])
    rows = []
    for t in sorted(tasks):
        traj = load_json(os.path.join(folder, f"traj_{t}.json"))
        if traj is not None and not isinstance(traj, list):
            traj = traj.get("history") or traj.get("events") or []
        state_raw = read_raw(os.path.join(folder, f"state_{t}.json"))
        ending, clue = classify(traj, state_raw)
        tr = traj or []
        rows.append({
            "run": os.path.basename(os.path.normpath(folder)),
            "task": t,
            "ending": ending,
            "read": "yes" if really_read_task(tr) else "no",
            "acts": sum(1 for e in tr if is_agent_action(e)),
            "chats": count_chats(tr),
            "errs": sum(1 for e in tr if e.get("observation") == "error"),
            "score": score_of(folder, t),
            "clue": clue,
        })
    return rows


def show(folder, task):
    traj = load_json(os.path.join(folder, f"traj_{task}.json"))
    if traj is None:
        print("No readable trajectory for", task)
        return
    if not isinstance(traj, list):
        traj = traj.get("history") or traj.get("events") or []
    for i, e in enumerate(traj):
        src = str(e.get("source", ""))[:6]
        txt = event_text(e).replace("\n", " | ")
        print(f"{i:3} {src:6} {event_kind(e):28} {txt[:220]}")
    print("\n--- state file mentions ---")
    raw = read_raw(os.path.join(folder, f"state_{task}.json"))
    for k in ("agent_state", "last_error", "iteration", "max_iterations"):
        for m in re.finditer(k + r"['\"]?\s*[:=]\s*([^,\n]{1,120})", raw):
            print(f"{k}: {m.group(1)}")
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--show")
    ap.add_argument("--csv")
    a = ap.parse_args()

    if a.show:
        for f in a.folders:
            print(f"===== {f} / {a.show}")
            show(f, a.show)
        return

    allrows = []
    for f in a.folders:
        rows = analyse_folder(f)
        allrows += rows
        print(f"\n===== {f}  ({len(rows)} tasks)")
        print(f"{'task':34} {'ending':10} {'read':4} {'acts':>4} {'chats':>5} {'errs':>4} {'score':>6}  clue")
        for r in rows:
            print(f"{r['task'][:34]:34} {r['ending']:10} {r['read']:4} {r['acts']:>4} {r['chats']:>5} {r['errs']:>4} {r['score']:>6}  {r['clue'][:90]}")
        tally = {}
        for r in rows:
            tally[r["ending"]] = tally.get(r["ending"], 0) + 1
        print("totals:", ", ".join(f"{k} {v}" for k, v in sorted(tally.items(), key=lambda x: -x[1])))

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(allrows[0].keys()))
            w.writeheader()
            w.writerows(allrows)
        print("\nwrote", a.csv)


if __name__ == "__main__":
    main()