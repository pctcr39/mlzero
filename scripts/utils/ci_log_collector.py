"""
================================================================================
scripts/utils/ci_log_collector.py
CI/CD Build Log Collector — Fetch, Store, and Analyze GitHub Actions Logs
================================================================================

PURPOSE:
    After every GitHub Actions CI run, this script:
      1. Fetches the latest CI run results using the `gh` CLI
      2. Parses test counts (passed / failed / errors) from pytest output
      3. Saves structured JSON to outputs/ci_logs/
      4. Prints a human-readable summary for quick analysis
      5. Generates a markdown trend report: outputs/ci_logs/REPORT.md

WHY STORE CI LOGS LOCALLY?
    - GitHub only keeps logs for 90 days (free tier: 30 days)
    - Local storage lets you track trends over months
    - You can spot: "tests started failing after this commit"
    - You can ask Claude: "analyze my CI logs" → Claude reads JSON files
    - Improvement loop: fail → fix → rerun → compare logs → improve

HOW TO RUN:
    source .venv/bin/activate
    python scripts/utils/ci_log_collector.py             # fetch latest 10 runs
    python scripts/utils/ci_log_collector.py --limit 25  # fetch latest 25 runs
    python scripts/utils/ci_log_collector.py --report    # just regenerate report
    python scripts/utils/ci_log_collector.py --analyze   # print analysis to terminal

PREREQUISITES:
    brew install gh          # GitHub CLI
    gh auth login            # authenticate once

GLOBAL IMPORTS:
    subprocess — run shell commands (gh CLI)
    json       — parse/save JSON
    re         — parse pytest output with regex
    pathlib    — file paths (cross-platform)
    argparse   — command-line arguments
================================================================================
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


# ==============================================================================
# GLOBAL CONSTANTS
# ==============================================================================

CI_LOG_DIR  = Path("outputs/ci_logs")       # GLOBAL: where we store logs
REPORT_FILE = CI_LOG_DIR / "REPORT.md"      # GLOBAL: generated markdown report
REPO        = ""                             # GLOBAL: auto-detected from git remote


# ==============================================================================
# HELPERS
# ==============================================================================

def run_gh(args: list[str]) -> tuple[str, int]:
    """
    Run a `gh` CLI command and return (stdout, returncode).

    THEORY:
        We call GitHub's CLI tool (`gh`) via subprocess.
        gh uses your GitHub token (from `gh auth login`) to authenticate.
        It returns JSON that we parse.

    PARAMETERS:
        args (LOCAL) — list of arguments, e.g. ["run", "list", "--limit", "10"]

    RETURNS:
        (stdout: str, returncode: int)
    """
    cmd    = ["gh"] + args                              # LOCAL: full command
    result = subprocess.run(                            # LOCAL
        cmd,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip(), result.returncode


def detect_repo() -> str:
    """
    Auto-detect the GitHub repo from git remote origin.

    RETURNS:
        str: "owner/repo" e.g. "pctcr39/mlzero"
    """
    stdout, rc = run_gh(["repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"])
    if rc != 0:
        print("ERROR: Could not detect repo. Run `gh auth login` first.")
        sys.exit(1)
    return stdout.strip()    # LOCAL: "owner/repo"


def parse_pytest_output(log_text: str) -> dict:
    """
    Extract test results from raw pytest output.

    THEORY:
        pytest prints a summary line like:
          "5 passed, 1 failed, 0 errors in 2.34s"
        We use regex (pattern matching) to find these numbers.

    PARAMETERS:
        log_text (LOCAL) — raw text output from the CI job

    RETURNS:
        dict with keys: tests_passed, tests_failed, tests_errors, duration_seconds
    """
    result = {                          # LOCAL: default values
        "tests_passed":  0,
        "tests_failed":  0,
        "tests_errors":  0,
        "duration_seconds": None,
    }

    # Match patterns like: "18 passed" or "18 passed, 2 failed"
    # REGEX breakdown:
    #   (\d+) passed    → capture a number followed by " passed"
    #   (\d+) failed    → capture a number followed by " failed"
    #   (\d+) error     → capture a number followed by " error"
    #   in (\d+\.\d+)s  → capture the duration in seconds

    passed_match  = re.search(r"(\d+) passed",      log_text)   # LOCAL
    failed_match  = re.search(r"(\d+) failed",      log_text)   # LOCAL
    error_match   = re.search(r"(\d+) error",       log_text)   # LOCAL
    duration_match = re.search(r"in (\d+\.\d+)s",  log_text)   # LOCAL

    if passed_match:
        result["tests_passed"]  = int(passed_match.group(1))
    if failed_match:
        result["tests_failed"]  = int(failed_match.group(1))
    if error_match:
        result["tests_errors"]  = int(error_match.group(1))
    if duration_match:
        result["duration_seconds"] = float(duration_match.group(1))

    return result


# ==============================================================================
# CORE: FETCH RUNS
# ==============================================================================

def fetch_runs(repo: str, limit: int = 10) -> list[dict]:
    """
    Fetch the latest CI run summaries from GitHub Actions.

    PARAMETERS:
        repo  (LOCAL) — "owner/repo"
        limit (LOCAL) — max number of runs to fetch

    RETURNS:
        list of run summary dicts
    """
    print(f"Fetching latest {limit} CI runs from {repo}...")

    stdout, rc = run_gh([
        "run", "list",
        "--repo", repo,
        "--limit", str(limit),
        "--json", "databaseId,createdAt,conclusion,headBranch,name,status,workflowName",
    ])

    if rc != 0:
        print(f"ERROR fetching runs: {stdout}")
        return []

    try:
        runs = json.loads(stdout)    # LOCAL: list of run objects
    except json.JSONDecodeError:
        print(f"ERROR: could not parse gh output:\n{stdout}")
        return []

    return runs


# ==============================================================================
# CORE: FETCH LOG FOR ONE RUN
# ==============================================================================

def fetch_run_log(repo: str, run_id: int) -> str:
    """
    Download the full log text for a single CI run.

    PARAMETERS:
        repo   (LOCAL) — "owner/repo"
        run_id (LOCAL) — integer run ID

    RETURNS:
        str: raw log text (may be multiple MB for long runs)
    """
    stdout, rc = run_gh([
        "run", "view", str(run_id),
        "--repo", repo,
        "--log",
    ])

    if rc != 0:
        return f"[could not fetch log for run {run_id}: exit {rc}]"

    return stdout


# ==============================================================================
# CORE: SAVE ONE RUN
# ==============================================================================

def save_run(repo: str, run: dict, fetch_log: bool = True) -> Path:
    """
    Save a CI run's metadata + log to a JSON file in outputs/ci_logs/.

    PARAMETERS:
        repo      (LOCAL) — "owner/repo"
        run       (LOCAL) — run summary dict from gh
        fetch_log (LOCAL) — whether to download the full log text

    RETURNS:
        Path to the saved JSON file
    """
    run_id   = run["databaseId"]             # LOCAL
    filename = CI_LOG_DIR / f"run_{run_id}.json"   # LOCAL

    # Skip if already saved
    if filename.exists():
        print(f"  ✓ Run {run_id} already saved, skipping.")
        return filename

    print(f"  → Fetching run {run_id} ({run.get('headBranch', '?')} | {run.get('conclusion', 'in_progress')})...")

    # Fetch full log
    raw_log = fetch_run_log(repo, run_id) if fetch_log else ""

    # Parse pytest results from log
    parsed = parse_pytest_output(raw_log)     # LOCAL: extracted test counts

    # Build structured record
    record = {                                # LOCAL
        "run_id":          run_id,
        "repo":            repo,
        "branch":          run.get("headBranch", ""),
        "workflow":        run.get("workflowName", ""),
        "conclusion":      run.get("conclusion", ""),
        "status":          run.get("status", ""),
        "created_at":      run.get("createdAt", ""),
        "fetched_at":      datetime.now(timezone.utc).isoformat(),
        "python_version":  _extract_python_version(raw_log),
        "tests_passed":    parsed["tests_passed"],
        "tests_failed":    parsed["tests_failed"],
        "tests_errors":    parsed["tests_errors"],
        "duration_seconds": parsed["duration_seconds"],
        "raw_log":         raw_log[:50_000],  # cap at 50KB to keep files small
    }

    with open(filename, "w") as f:
        json.dump(record, f, indent=2)

    print(f"     Saved → {filename}  "
          f"({parsed['tests_passed']} passed, {parsed['tests_failed']} failed)")

    return filename


def _extract_python_version(log_text: str) -> str:
    """
    Extract Python version string from pytest output.
    Example: "platform darwin -- Python 3.11.5" → "3.11.5"

    PARAMETERS:
        log_text (LOCAL) — raw log text
    RETURNS:
        str: version string or "?"
    """
    match = re.search(r"Python (\d+\.\d+\.\d+)", log_text)   # LOCAL
    return match.group(1) if match else "?"


# ==============================================================================
# ANALYSIS
# ==============================================================================

def analyze_logs() -> dict:
    """
    Load all stored CI logs and compute trend statistics.

    THEORY:
        We read all JSON files, sort by date, and compute:
          - pass rate over time
          - average test count
          - fastest/slowest builds
          - which branches fail most

    RETURNS:
        dict with analysis results
    """
    log_files = sorted(CI_LOG_DIR.glob("run_*.json"))   # LOCAL

    if not log_files:
        print("No CI logs found. Run this script without --analyze first to fetch logs.")
        return {}

    records = []    # LOCAL: all loaded records
    for f in log_files:
        with open(f) as fh:
            records.append(json.load(fh))

    # Sort chronologically
    records.sort(key=lambda r: r.get("created_at", ""))

    total       = len(records)                                    # LOCAL
    passed_runs = sum(1 for r in records if r["conclusion"] == "success")  # LOCAL
    failed_runs = total - passed_runs                             # LOCAL
    pass_rate   = (passed_runs / total * 100) if total > 0 else 0  # LOCAL

    # Average test counts (only from successful runs with test data)
    has_tests   = [r for r in records if r.get("tests_passed", 0) > 0]  # LOCAL
    avg_passed  = sum(r["tests_passed"] for r in has_tests) / len(has_tests) if has_tests else 0  # LOCAL

    # Slowest builds
    timed = [r for r in records if r.get("duration_seconds")]    # LOCAL
    slowest = max(timed, key=lambda r: r["duration_seconds"]) if timed else None  # LOCAL
    fastest = min(timed, key=lambda r: r["duration_seconds"]) if timed else None  # LOCAL

    analysis = {                    # LOCAL
        "total_runs":       total,
        "passed_runs":      passed_runs,
        "failed_runs":      failed_runs,
        "pass_rate_pct":    round(pass_rate, 1),
        "avg_tests_passed": round(avg_passed, 1),
        "slowest_run_id":   slowest["run_id"] if slowest else None,
        "slowest_seconds":  slowest["duration_seconds"] if slowest else None,
        "fastest_run_id":   fastest["run_id"] if fastest else None,
        "fastest_seconds":  fastest["duration_seconds"] if fastest else None,
        "date_range":       f"{records[0]['created_at'][:10]} → {records[-1]['created_at'][:10]}",
        "recent_5":         [{"id": r["run_id"], "branch": r["branch"],
                              "result": r["conclusion"], "date": r["created_at"][:10]}
                             for r in records[-5:]],
    }

    return analysis


def print_analysis(analysis: dict) -> None:
    """Print a human-readable analysis summary to terminal."""
    if not analysis:
        return

    print("\n" + "=" * 60)
    print("CI/CD LOG ANALYSIS")
    print("=" * 60)
    print(f"Date range      : {analysis['date_range']}")
    print(f"Total runs      : {analysis['total_runs']}")
    print(f"Passed / Failed : {analysis['passed_runs']} / {analysis['failed_runs']}")
    print(f"Pass rate       : {analysis['pass_rate_pct']}%")
    print(f"Avg tests/run   : {analysis['avg_tests_passed']}")

    if analysis["slowest_run_id"]:
        print(f"Slowest build   : run {analysis['slowest_run_id']} ({analysis['slowest_seconds']:.1f}s)")
        print(f"Fastest build   : run {analysis['fastest_run_id']} ({analysis['fastest_seconds']:.1f}s)")

    print("\nRecent 5 runs:")
    for r in analysis["recent_5"]:
        icon = "✅" if r["result"] == "success" else "❌"
        print(f"  {icon} {r['date']}  branch={r['branch']:<15}  id={r['id']}")

    print("=" * 60)


# ==============================================================================
# REPORT GENERATOR
# ==============================================================================

def generate_report(analysis: dict) -> None:
    """
    Write a markdown report to outputs/ci_logs/REPORT.md.

    WHY MARKDOWN?
        - Readable in GitHub, VS Code, and any text editor
        - Can be committed to git for a permanent record
        - Claude can read and analyze it in future sessions
    """
    if not analysis:
        return

    lines = [                    # LOCAL
        "# CI/CD Build Log Report",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Date range | {analysis['date_range']} |",
        f"| Total runs | {analysis['total_runs']} |",
        f"| Pass rate | {analysis['pass_rate_pct']}% |",
        f"| Avg tests passing | {analysis['avg_tests_passed']} |",
        "",
        "## Recent Runs",
        "",
        "| Date | Branch | Result | Run ID |",
        "|---|---|---|---|",
    ]

    for r in reversed(analysis["recent_5"]):
        icon = "✅ Pass" if r["result"] == "success" else "❌ Fail"
        lines.append(f"| {r['date']} | `{r['branch']}` | {icon} | {r['id']} |")

    lines += [
        "",
        "## Improvement Notes",
        "",
        "> Edit this section manually after reviewing failures.",
        "",
        "- [ ] Review any ❌ failures above",
        "- [ ] Check if tests_failed > 0 in any recent run",
        "- [ ] Update `docs/guides/LEARNING_GUIDE.md` after fixing issues",
    ]

    REPORT_FILE.write_text("\n".join(lines))
    print(f"\nReport saved → {REPORT_FILE}")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fetch and analyze GitHub Actions CI logs locally."
    )
    parser.add_argument("--limit",   type=int, default=10, help="Number of recent runs to fetch (default: 10)")
    parser.add_argument("--report",  action="store_true",  help="Regenerate REPORT.md from stored logs only")
    parser.add_argument("--analyze", action="store_true",  help="Print analysis to terminal only")
    parser.add_argument("--no-log",  action="store_true",  help="Skip downloading full log text (faster)")
    args = parser.parse_args()

    CI_LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Report-only mode ──────────────────────────────────────────────────────
    if args.report or args.analyze:
        analysis = analyze_logs()
        print_analysis(analysis)
        if args.report:
            generate_report(analysis)
        return

    # ── Detect repo ───────────────────────────────────────────────────────────
    repo = detect_repo()
    print(f"Repository: {repo}")

    # ── Fetch runs ────────────────────────────────────────────────────────────
    runs = fetch_runs(repo, limit=args.limit)

    if not runs:
        print("No runs found. Make sure CI has run at least once (push to GitHub).")
        return

    print(f"\nFetching logs for {len(runs)} runs...")

    for run in runs:
        save_run(repo, run, fetch_log=not args.no_log)

    # ── Analyze and report ────────────────────────────────────────────────────
    analysis = analyze_logs()
    print_analysis(analysis)
    generate_report(analysis)

    print(f"\nDone. Logs stored in: {CI_LOG_DIR}/")
    print("View in Streamlit: Tab 'CI/CD Logs' → http://localhost:8501")


if __name__ == "__main__":
    main()
