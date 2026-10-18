"""
Benchmark Accuracy Evaluator:
Runs Text Quality Auditor across all 11 synthetic benchmarks:
- 1 Clean Baseline (100% QI, all checks PASS)
- 7 Fatal Defect Benchmarks (testing hard gates -> BLOCKED)
- 3 Graded Quality Benchmarks (testing continuous scoring -> REVIEW_REQUIRED, zero blockers, QI 85-95)
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "data" / "benchmarks"
STD_POLICY = ROOT / "policies" / "benchmark_policy.json"
PII_POLICY = ROOT / "policies" / "benchmark_pii_policy.json"
LABELS_PATH = BENCH_DIR / "labels.json"


def main():
    gt_path = BENCH_DIR / "ground_truth.json"
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    print("=" * 120)
    print("TEXT DATASET QUALITY AUDITOR — SYNTHETIC CORRUPTION BENCHMARK EVALUATION")
    print("=" * 120)
    print(f"{'Benchmark':<33} {'Target / Type':<25} {'Quality Index':<16} {'Decision':<16} {'Blockers':<12} {'Status'}")
    print("-" * 120)

    all_passed = True

    for key, spec in ground_truth.items():
        csv_file = BENCH_DIR / spec["file"]
        target = spec["target_metric"]
        expected_defects = spec["injected_defects"]
        ground_truth_rows = set(spec.get("corrupted_row_indices", []))

        policy_to_use = PII_POLICY if key == "pii_leakage" else STD_POLICY

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "report"
            cmd = [
                sys.executable, "-m", "textaudit.cli",
                str(csv_file),
                "--label-column", "label",
                "--split-column", "split",
                "--allowed-labels", str(LABELS_PATH),
                "--policy", str(policy_to_use),
                "--out", str(out_dir),
            ]
            if key == "pii_leakage":
                cmd.append("--require-redacted")

            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
            if proc.returncode not in (0, 2):
                print(f"{spec['file']:<33} CLI CRASHED (code {proc.returncode}): {proc.stderr[:80]}")
                all_passed = False
                continue

            report_file = out_dir / "report.json"
            if not report_file.exists():
                print(f"{spec['file']:<33} CLI didn't generate report.json")
                all_passed = False
                continue

            report = json.loads(report_file.read_text(encoding="utf-8"))
            dec = report["decision"]
            qi = report["quality_index"]
            blockers = report.get("blockers", [])

            # Category 1: Pristine Control Baseline
            if key == "clean_baseline":
                passed = qi == 100.0 and dec == "CHECKS_PASSED_REVIEW_REQUIRED" and len(blockers) == 0
                status = "PASS" if passed else "FAIL"
                print(f"{spec['file']:<33} {'Pristine Baseline':<25} {f'{qi:.1f} / 100':<16} {dec:<16} {'None':<12} {status}")
                if not passed:
                    all_passed = False
                continue

            # Category 2: Graded Soft Quality Deficits (Non-blocked, QI 85-98)
            if "expected_quality_range" in spec:
                q_min, q_max = spec["expected_quality_range"]
                passed = (q_min <= qi <= q_max) and dec == "REVIEW_REQUIRED" and len(blockers) == 0
                status = "PASS (Graded)" if passed else "FAIL"
                desc_str = f"Soft: {target}"
                qi_str = f"{qi:.1f} / 100"
                print(f"{spec['file']:<33} {desc_str:<25} {qi_str:<16} {dec:<16} {'0 (Unblocked)':<12} {status}")
                if not passed:
                    all_passed = False
                continue

            # Category 3: Fatal Gate Defects (Hard Blockers)
            metric_data = next((m for m in report["metrics"] if m["metric"] == target), None)
            if not metric_data:
                print(f"{spec['file']:<33} {target:<25} {'N/A':<16} {dec:<16} {'Missing':<12} FAIL (Metric not found)")
                all_passed = False
                continue

            detected_samples = set(metric_data.get("issue_row_ids", []))
            bad_count = metric_data.get("bad_rows", len(detected_samples))
            tp_sample = len(detected_samples.intersection(ground_truth_rows))
            precision = (tp_sample / len(detected_samples)) if detected_samples else 1.0

            expected_count = 2 * expected_defects if target == "label_consistency" else expected_defects
            is_detected = metric_data["status"] != "PASS" and bad_count >= expected_count and precision == 1.0 and dec == "BLOCKED"
            status = "PASS (Blocked)" if is_detected else "FAIL"
            if not is_detected:
                all_passed = False

            eff_qi = report.get("effective_quality_index", qi)
            qi_str = f"{eff_qi:.1f} [Raw: {qi:.0f}]"
            block_str = f"{len(blockers)} blocker(s)"
            print(f"{spec['file']:<33} {f'Gate: {target}':<25} {qi_str:<16} {dec:<16} {block_str:<12} {status}")

    print("-" * 120)
    if all_passed:
        print("ALL 11 BENCHMARKS PASSED! Hard gates trigger BLOCKED, graded tests produce 85-98 Quality Index with 0 blockers.")
    else:
        print("SOME BENCHMARKS FAILED! Check the output above.")
    print("=" * 120)


if __name__ == "__main__":
    main()
