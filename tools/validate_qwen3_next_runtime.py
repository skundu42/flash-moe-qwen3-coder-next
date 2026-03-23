#!/usr/bin/env python3
"""Run a small canned validation suite against the Qwen3-Coder-Next runtime."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]

PROMPTS = [
    {
        "name": "hello",
        "kind": "plain",
        "prompt": "Hello",
        "tokens": 16,
    },
    {
        "name": "linked_list",
        "kind": "plain",
        "prompt": "Write a Python function that reverses a linked list.",
        "tokens": 48,
    },
    {
        "name": "tool_select",
        "kind": "tool_envelope",
        "prompt": (
            "You can use tools. If shell inspection is required, answer with only "
            "<tool_call>{\"name\":\"bash\",\"arguments\":{\"command\":\"pwd\"}}</tool_call>."
        ),
        "tokens": 48,
    },
    {
        "name": "json_args",
        "kind": "json",
        "prompt": (
            "Return only JSON with keys name and arguments, selecting the bash tool "
            "to run echo hi."
        ),
        "tokens": 48,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Qwen3-Coder-Next runtime with canned prompts")
    parser.add_argument("--infer", default=str(REPO_ROOT / "metal_infer" / "infer"), help="Path to infer binary")
    parser.add_argument("--model", required=True, help="Model directory")
    parser.add_argument("--output", default=None, help="Optional JSON report path")
    parser.add_argument("--k", type=int, default=10, help="Active experts per layer")
    parser.add_argument("--runs", type=int, default=3, help="Number of repeats for each canned prompt")
    parser.add_argument("--layers", default="0,1,47", help="Layers to compare for the first run of each prompt")
    parser.add_argument("--skip-compare", action="store_true", help="Skip reference_compare.py")
    return parser.parse_args()


def extract_output(stdout: str) -> str:
    marker = "--- Output ---"
    idx = stdout.find(marker)
    if idx < 0:
        return stdout.strip()
    text = stdout[idx + len(marker):]
    text = text.split("--- Statistics ---", 1)[0]
    return text.strip()


def looks_repetitive(text: str) -> bool:
    cleaned = " ".join(text.split())
    if not cleaned:
        return True
    for width in range(1, 5):
        pieces = cleaned.split()
        if len(pieces) < width * 3:
            continue
        chunk = " ".join(pieces[:width])
        if chunk and cleaned.startswith(" ".join([chunk] * 3)):
            return True
    return False


def classify_output(kind: str, output: str) -> Dict:
    verdict = {
        "empty": not output.strip(),
        "repetitive": looks_repetitive(output),
    }
    if kind == "json":
        try:
            json.loads(output)
            verdict["json_valid"] = True
        except Exception:
            verdict["json_valid"] = False
    elif kind == "tool_envelope":
        verdict["tool_envelope_valid"] = bool(
            re.search(r"<tool_call>\s*\{.*\}\s*</tool_call>", output, flags=re.DOTALL)
        )
    return verdict


def run_case(args: argparse.Namespace, case: Dict, run_idx: int) -> Dict:
    dump_dir = Path(tempfile.mkdtemp(prefix=f"qwen3-next-{case['name']}-{run_idx:02d}-", dir="/tmp"))
    cmd = [
        args.infer,
        "--model", args.model,
        "--prompt", case["prompt"],
        "--tokens", str(case["tokens"]),
        "--k", str(args.k),
        "--dump-dir", str(dump_dir),
        "--dump-layers", args.layers,
        "--dump-stages", "embedding,attn,router,shared,final",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = extract_output(proc.stdout)
    result = {
        "returncode": proc.returncode,
        "cmd": cmd,
        "dump_dir": str(dump_dir),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "output": output,
        "checks": classify_output(case["kind"], output),
    }
    if run_idx == 0 and not args.skip_compare and proc.returncode == 0:
        compare_cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "reference_compare.py"),
            "--model", args.model,
            "--dump-dir", str(dump_dir),
            "--layers", args.layers,
        ]
        compare_proc = subprocess.run(compare_cmd, capture_output=True, text=True)
        result["reference_compare"] = {
            "returncode": compare_proc.returncode,
            "stdout": compare_proc.stdout,
            "stderr": compare_proc.stderr,
            "cmd": compare_cmd,
        }
    return result


def main() -> int:
    args = parse_args()
    report = {"model": args.model, "runs": {}}
    overall_ok = True
    for case in PROMPTS:
        case_runs: List[Dict] = []
        for run_idx in range(args.runs):
            result = run_case(args, case, run_idx)
            case_runs.append(result)
            if result["returncode"] != 0:
                overall_ok = False
            checks = result["checks"]
            if checks.get("empty") or checks.get("repetitive"):
                overall_ok = False
            if case["kind"] == "json" and not checks.get("json_valid", False):
                overall_ok = False
            if case["kind"] == "tool_envelope" and not checks.get("tool_envelope_valid", False):
                overall_ok = False
            if "reference_compare" in result and result["reference_compare"]["returncode"] != 0:
                overall_ok = False
        report["runs"][case["name"]] = {
            "kind": case["kind"],
            "prompt": case["prompt"],
            "results": case_runs,
        }

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
