#!/usr/bin/env python3
"""
Local helper to generate API docs the same way CI does.
Run: ./scripts/generate_api_docs.py
"""
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
os.environ["PYTHONPATH"] = (
    os.path.join(REPO_ROOT, "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
)

# ensure pdoc is installed in the environment; otherwise pip-install it
try:
    import pdoc  # noqa: F401
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pdoc"])

# generate into docs/api
out_dir = os.path.join(REPO_ROOT, "docs", "api")
cmd = [sys.executable, "-m", "pdoc", "--output-dir", out_dir, "--force", "simba_plus"]
subprocess.check_call(cmd)
print("Wrote API docs to", out_dir)
