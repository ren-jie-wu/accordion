#!/usr/bin/env python3
import subprocess
import sys
import os
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(REPO_ROOT, "docs")
OUT_FILE = os.path.join(OUT_DIR, "CLI.md")

SUBCOMMANDS = ["load_data", "train", "eval"]

os.makedirs(OUT_DIR, exist_ok=True)

lines = []

lines.append(
    "Run `python -m simba_plus.simba_plus <subcommand> -h` for usage examples.\n\n"
)

for sub in SUBCOMMANDS:
    lines.append(f"## simba+ `{sub}` ... \n\n")
    cmd = [sys.executable, "-m", "simba_plus.simba_plus", sub, "-h"]
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": os.path.join(REPO_ROOT, "src")},
        )
        out = proc.stdout.strip() or proc.stderr.strip()
        if proc.returncode != 0:
            out = (
                f"Error running `{ ' '.join(cmd) }` (exit {proc.returncode}):\n\n" + out
            )
    except Exception as e:
        out = f"Exception running `{ ' '.join(cmd) }`:\n\n{e}"
    lines.append("```\n")
    lines.append(out + "\n")
    lines.append("```\n\n")

new_content = "".join(lines)

# Write only if changed to avoid unnecessary commits
old_content = None
if os.path.exists(OUT_FILE):
    with open(OUT_FILE, "r", encoding="utf8") as f:
        old_content = f.read()

if old_content != new_content:
    with open(OUT_FILE, "w", encoding="utf8") as f:
        f.write(new_content)
    print(f"Wrote updated CLI docs to {OUT_FILE}")
    sys.exit(0)
else:
    print("CLI docs are up-to-date")
    sys.exit(0)
