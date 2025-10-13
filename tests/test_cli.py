import os
import sys
import subprocess

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(REPO_ROOT, "src")
PY = sys.executable


def _run_subcommand(args, timeout=15):
    env = os.environ.copy()
    env["PYTHONPATH"] = SRC_DIR + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [PY, "-m", "simba_plus.simba_plus"] + args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return proc


def test_load_data_help():
    proc = _run_subcommand(["load_data", "-h"])
    assert proc.returncode == 0, proc.stderr.decode()


def test_train_help():
    proc = _run_subcommand(["train", "-h"])
    assert proc.returncode == 0, proc.stderr.decode()


def test_eval_help():
    proc = _run_subcommand(["eval", "-h"])
    assert proc.returncode == 0, proc.stderr.decode()
