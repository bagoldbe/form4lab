"""Pre-publish leak audit. Hard-fails on secret-shaped strings; flags a few
generic patterns for human review.

Run from the repo root:
  python scripts/leak_audit.py             scan the current working tree only
  python scripts/leak_audit.py --history   ALSO scan commit history (git log -p)

Pre-push runs MUST use --history. A clean working-tree scan only proves the
current snapshot is clean — anyone who clones the repo also gets every earlier
commit's diff (blobs and messages), which the working-tree scan never looks at.

WHY THE HARD_FAIL LIST IS GENERIC (important — do not undo this):
The patterns committed below are universal secret shapes (private-key headers,
cloud access keys, tokens) that reveal nothing about any particular project.
Your deployment's PRIVATE vocabulary — internal codenames, proprietary strategy
terms, private infra hostnames, tuned constants, non-public ticker universes —
must NOT be typed into this file. If it were, this committed, world-readable
script would itself become the disclosure it exists to prevent (a "reverse
oracle": the list of things you're hiding IS the map of what you're hiding).
Put those operator-private patterns in `scripts/leak_audit.local` (gitignored),
one regex per line. See `scripts/leak_audit.local.example` for the format.
"""
import argparse
import pathlib
import re
import subprocess
import sys

# Generic, project-agnostic secret shapes. Safe to publish — they describe the
# STRUCTURE of a leaked secret, not the content of anything private.
HARD_FAIL = [
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----",  # PEM private key
    r"\bAKIA[0-9A-Z]{16}\b",                     # AWS access key id
    r"\bASIA[0-9A-Z]{16}\b",                     # AWS temporary access key id
    r"\bghp_[A-Za-z0-9]{36}\b",                  # GitHub personal access token
    r"\bgithub_pat_[A-Za-z0-9_]{22,}\b",         # GitHub fine-grained PAT
    r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b",         # Slack token
    r"\bAIza[0-9A-Za-z_\-]{35}\b",               # Google API key
    r"\bsk-[A-Za-z0-9]{20,}\b",                  # OpenAI-style secret key
]

# Soft patterns — printed for a human to eyeball, never fail the build. Inline
# connection-string credentials are frequently the intentional dev defaults in a
# committed docker-compose.yml / .env.example; a person confirms that's all they
# are. Add your own numeric/vocabulary review patterns to leak_audit.local if you
# want them surfaced too.
REVIEW = [
    r"[a-z][a-z0-9+.\-]*://[^\s:/@]+:[^\s:/@]+@",   # scheme://user:password@host
    r"(?i)(?:api[_-]?key|secret|password|token)\s*[=:]\s*['\"][^'\"]{12,}",
]

# Operator-private patterns load from a gitignored sibling (one regex per line,
# '#' comments allowed) so they never enter this committed file. They are NOT
# applied under PUBLIC_DATA_DIRS: those hold verbatim public datasets (e.g. index
# membership lists) where omitting an entry to satisfy a private pattern would
# itself disclose information through the absence.
LOCAL_FAIL = []
_local = pathlib.Path(__file__).with_name("leak_audit.local")
if _local.exists():
    LOCAL_FAIL = [l.strip() for l in _local.read_text().splitlines()
                  if l.strip() and not l.startswith("#")]

PUBLIC_DATA_DIRS = ("form4lab/data/universes/",)


def scan_tree():
    """Scan every git-tracked file in the working tree. Returns (fails, reviews)."""
    files = subprocess.run(["git", "ls-files"], capture_output=True, text=True).stdout.split()
    fails, reviews = [], []
    for path in files:
        if path == "scripts/leak_audit.py":
            # This file contains every HARD_FAIL/REVIEW pattern as a literal
            # string (it defines them) — scanning it would always self-match.
            continue
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except IsADirectoryError:
            continue
        active_fail = HARD_FAIL if path.startswith(PUBLIC_DATA_DIRS) else HARD_FAIL + LOCAL_FAIL
        for pat in active_fail:
            for m in re.finditer(pat, text, re.IGNORECASE):
                fails.append(f"{path}: {pat} -> ...{text[max(0,m.start()-30):m.end()+30]!r}...")
        for pat in REVIEW:
            for m in re.finditer(pat, text):
                reviews.append(f"{path}: {pat}")
    return fails, reviews


_DIFF_HEADER = re.compile(r"^diff --git a/.+ b/(.+)$")


def scan_history(rev: str):
    """Scan `git log -p <rev>` for HARD_FAIL/LOCAL_FAIL hits. Returns a list of hits.

    Deliberately does NOT use `git log -p --all`: --all walks every ref that
    exists locally (e.g. a renamed-aside old-main kept around during a history
    rewrite), which is not what `git push <remote> rev` would publish. Scanning
    exactly `rev`'s history matches what a clone of the pushed branch would see.

    Tracks the current file path from `diff --git a/... b/...` headers so the
    same two rules as scan_tree() apply: this script's own source is skipped (it
    necessarily contains every pattern as a literal), and paths under
    PUBLIC_DATA_DIRS only get the HARD_FAIL set, not the operator-local guards.

    CAVEAT: because this script's own path is skipped here too, a HISTORICAL
    version of scripts/leak_audit.py that once hard-coded private patterns is NOT
    caught by this scan. Genericizing the committed file (as it is now) does not
    retroactively clean older blobs — before a public release, rewrite history
    (fresh orphan branch) so no earlier leak_audit.py blob carries private terms.
    """
    patterns_public = HARD_FAIL
    patterns_all = HARD_FAIL + LOCAL_FAIL
    proc = subprocess.run(["git", "log", "-p", rev], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"error: git log -p {rev} failed:\n{proc.stderr}", file=sys.stderr)
        sys.exit(2)
    hits = []
    current_sha = "(unknown)"
    current_path = None
    for line in proc.stdout.splitlines():
        if line.startswith("commit "):
            current_sha = line.split()[1][:12]
            current_path = None
            continue
        m = _DIFF_HEADER.match(line)
        if m:
            current_path = m.group(1)
            continue
        if current_path == "scripts/leak_audit.py":
            continue
        patterns = patterns_public if (current_path or "").startswith(PUBLIC_DATA_DIRS) else patterns_all
        for pat in patterns:
            if re.search(pat, line, re.IGNORECASE):
                hits.append(f"{current_sha} {current_path}: {pat} -> {line.strip()[:100]!r}")
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history", nargs="?", const="HEAD", default=None, metavar="REV",
        help="Also scan `git log -p REV` (default REV: HEAD, i.e. the current "
             "branch's own history — NOT --all). Required for any pre-push check.",
    )
    args = parser.parse_args()

    fails, reviews = scan_tree()
    print(f"hard-fail hits: {len(fails)}")
    for f in fails:
        print("  FAIL", f)
    print(f"review hits: {len(reviews)}")
    for r in sorted(set(reviews)):
        print("  REVIEW", r)

    if args.history is not None:
        history_hits = scan_history(args.history)
        print(f"history hard-fail hits ({args.history}): {len(history_hits)}")
        for h in history_hits:
            print("  HISTORY-FAIL", h)
        fails = fails + history_hits

    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
