"""
Generate 8 Synthetic Quality Benchmarks for Text Quality Auditor:
- 1 Clean Baseline (100% Quality Index, all checks PASS)
- 7 Dedicated Corrupted Datasets (each isolates 1 specific failure mode)
"""

import csv
import json
import os
import random
import re
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
BENCH_DIR = DATA_DIR / "benchmarks"
BENCH_DIR.mkdir(parents=True, exist_ok=True)

SELECTED_LABELS = [
    "card_arrival",
    "card_linking",
    "exchange_rate",
    "card_payment_fee_charged",
    "extra_charge_on_statement",
    "pending_cash_withdrawal",
    "compromised_card",
    "lost_or_stolen_card",
    "top_up_reverted",
    "verify_top_up",
]

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d ()-]{7,}\d(?!\w)")


def load_clean_pool():
    train_by_lbl = {l: [] for l in SELECTED_LABELS}
    test_by_lbl = {l: [] for l in SELECTED_LABELS}
    seen_texts = set()

    with open(DATA_DIR / "banking77" / "dataset.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            txt = row["text"].strip()
            lbl = row["label"].strip()
            split = row["split"].strip()
            if lbl not in SELECTED_LABELS:
                continue
            if len(txt) < 15 or len(txt) > 300:
                continue
            if EMAIL_RE.search(txt) or PHONE_RE.search(txt):
                continue
            if any(ord(c) < 32 or ord(c) == 0xFFFD for c in txt):
                continue
            if txt.lower() in seen_texts:
                continue
            seen_texts.add(txt.lower())

            if split == "train" and len(train_by_lbl[lbl]) < 80:
                train_by_lbl[lbl].append({"text": txt, "label": lbl, "split": "train"})
            elif split == "test" and len(test_by_lbl[lbl]) < 20:
                test_by_lbl[lbl].append({"text": txt, "label": lbl, "split": "test"})

    train_rows = [r for l in SELECTED_LABELS for r in train_by_lbl[l]]
    test_rows = [r for l in SELECTED_LABELS for r in test_by_lbl[l]]
    return train_rows, test_rows


def save_csv(path: Path, rows: list[dict]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "split"])
        w.writeheader()
        w.writerows(rows)


def main():
    train_pool, test_pool = load_clean_pool()
    base_rows = train_pool + test_pool  # 800 train, 200 test = 1000 rows

    # Save allowed labels
    labels_path = BENCH_DIR / "labels.json"
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(SELECTED_LABELS, f, indent=2)

    ground_truth = {}

    # ─────────────────────────────────────────────────────────────────────────
    # 0. Clean Baseline
    # ─────────────────────────────────────────────────────────────────────────
    save_csv(BENCH_DIR / "benchmark_clean_baseline.csv", base_rows)
    ground_truth["clean_baseline"] = {
        "file": "benchmark_clean_baseline.csv",
        "description": "Pristine control baseline: 800 train, 200 test across 10 balanced classes.",
        "target_metric": None,
        "expected_decision": "CHECKS_PASSED_REVIEW_REQUIRED",
        "expected_quality_index": 100.0,
        "injected_defects": 0,
        "corrupted_row_indices": [],
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Split Leakage (Train rows copied into test split)
    # ─────────────────────────────────────────────────────────────────────────
    leak_rows = [dict(r) for r in base_rows]
    leaked_indices = []
    # Copy 25 random train rows and replace 25 test rows with them
    train_indices = [i for i, r in enumerate(leak_rows) if r["split"] == "train"]
    test_indices = [i for i, r in enumerate(leak_rows) if r["split"] == "test"]
    sampled_train = random.sample(train_indices, 25)
    sampled_test = test_indices[:25]

    for t_idx, dest_idx in zip(sampled_train, sampled_test):
        leak_rows[dest_idx]["text"] = leak_rows[t_idx]["text"]
        leaked_indices.append(dest_idx + 1)

    save_csv(BENCH_DIR / "benchmark_split_leakage.csv", leak_rows)
    ground_truth["split_leakage"] = {
        "file": "benchmark_split_leakage.csv",
        "description": "25 training texts duplicated directly into test split.",
        "target_metric": "split_disjointness",
        "expected_decision": "BLOCKED",
        "injected_defects": 25,
        "corrupted_row_indices": sorted(leaked_indices),
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Conflicting Labels (Identical text with contradictory labels)
    # ─────────────────────────────────────────────────────────────────────────
    conflict_rows = [dict(r) for r in base_rows]
    conflict_indices = []
    # Take 20 train rows and duplicate them with another valid label
    sampled_conflict = random.sample(range(len(train_pool)), 20)
    for i, idx in enumerate(sampled_conflict):
        orig = conflict_rows[idx]
        alt_label = random.choice([l for l in SELECTED_LABELS if l != orig["label"]])
        # Replace the 20 test rows at the end with duplicate texts having different labels
        dest_idx = len(conflict_rows) - 20 + i
        conflict_rows[dest_idx] = {
            "text": orig["text"],
            "label": alt_label,
            "split": orig["split"],
        }
        conflict_indices.extend([idx + 1, dest_idx + 1])

    save_csv(BENCH_DIR / "benchmark_label_conflict.csv", conflict_rows)
    ground_truth["label_conflict"] = {
        "file": "benchmark_label_conflict.csv",
        "description": "20 identical texts assigned conflicting intent labels.",
        "target_metric": "label_consistency",
        "expected_decision": "BLOCKED",
        "injected_defects": 20,
        "corrupted_row_indices": sorted(conflict_indices),
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Encoding Errors (Corrupt UTF-8, null bytes, control chars)
    # ─────────────────────────────────────────────────────────────────────────
    encoding_rows = [dict(r) for r in base_rows]
    encoding_indices = []
    corruptions = [
        "Card payment fee \ufffd charged twice on statement.",
        "Why is there a \x00 null byte in my account info?",
        "Where is my card? \x07\x08 Bell signal error.",
        "Exchange rate \ufffd\ufffd was corrupted during data export.",
        "Transaction reverted: \x1b[31mTerminal Error\x1b[0m",
    ]
    sampled_enc = random.sample(range(100, 300), 20)
    for i, idx in enumerate(sampled_enc):
        pattern = corruptions[i % len(corruptions)]
        encoding_rows[idx]["text"] = pattern
        encoding_indices.append(idx + 1)

    save_csv(BENCH_DIR / "benchmark_encoding_errors.csv", encoding_rows)
    ground_truth["encoding_errors"] = {
        "file": "benchmark_encoding_errors.csv",
        "description": "20 rows injected with Unicode replacement chars, null bytes, and terminal control codes.",
        "target_metric": "encoding_cleanliness",
        "expected_decision": "REVIEW_REQUIRED",
        "injected_defects": 20,
        "corrupted_row_indices": sorted(encoding_indices),
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 4. PII / Privacy Leakage (Customer emails & phone numbers in queries)
    # ─────────────────────────────────────────────────────────────────────────
    pii_rows = [dict(r) for r in base_rows]
    pii_indices = []
    pii_samples = [
        "Please send statement to john.smith99@gmail.com immediately.",
        "Contact me regarding my lost card at +1-555-019-2834.",
        "My email is sarah_connor@bankclient.co.uk please confirm card arrival.",
        "Call our office phone (+44) 20 7946 0991 to unlock my account.",
        "Invoice receipt requested by robert.finance@corp-enterprise.net",
    ]
    sampled_pii = random.sample(range(300, 500), 20)
    for i, idx in enumerate(sampled_pii):
        pii_rows[idx]["text"] = pii_samples[i % len(pii_samples)]
        pii_indices.append(idx + 1)

    save_csv(BENCH_DIR / "benchmark_pii_leakage.csv", pii_rows)
    ground_truth["pii_leakage"] = {
        "file": "benchmark_pii_leakage.csv",
        "description": "20 rows containing customer email addresses and phone numbers.",
        "target_metric": "privacy_pattern_clearance",
        "expected_decision": "BLOCKED",
        "injected_defects": 20,
        "corrupted_row_indices": sorted(pii_indices),
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Empty / Incomplete Texts (Blank strings, whitespace, nulls)
    # ─────────────────────────────────────────────────────────────────────────
    empty_rows = [dict(r) for r in base_rows]
    empty_indices = []
    blanks = ["", "   ", "\t\t", " \n "]
    sampled_empty = random.sample(range(500, 700), 15)
    for i, idx in enumerate(sampled_empty):
        empty_rows[idx]["text"] = blanks[i % len(blanks)]
        empty_indices.append(idx + 1)

    save_csv(BENCH_DIR / "benchmark_empty_texts.csv", empty_rows)
    ground_truth["empty_texts"] = {
        "file": "benchmark_empty_texts.csv",
        "description": "15 rows with blank, whitespace-only, or missing text strings.",
        "target_metric": "text_completeness",
        "expected_decision": "BLOCKED",
        "injected_defects": 15,
        "corrupted_row_indices": sorted(empty_indices),
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Invalid / Hallucinated Labels (Outside allowed label schema)
    # ─────────────────────────────────────────────────────────────────────────
    invalid_lbl_rows = [dict(r) for r in base_rows]
    invalid_lbl_indices = []
    fake_labels = [
        "hallucinated_intent_404",
        "cryptocurrency_scam",
        "mortgage_refinance_unsupported",
        "unknown_customer_query",
    ]
    sampled_inv = random.sample(range(700, 900), 20)
    for i, idx in enumerate(sampled_inv):
        invalid_lbl_rows[idx]["label"] = fake_labels[i % len(fake_labels)]
        invalid_lbl_indices.append(idx + 1)

    save_csv(BENCH_DIR / "benchmark_invalid_labels.csv", invalid_lbl_rows)
    ground_truth["invalid_labels"] = {
        "file": "benchmark_invalid_labels.csv",
        "description": "20 rows assigned invalid/unrecognized labels outside the 10 allowed banking classes.",
        "target_metric": "label_validity",
        "expected_decision": "BLOCKED",
        "injected_defects": 20,
        "corrupted_row_indices": sorted(invalid_lbl_indices),
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Length Anomalies (Extreme length violations)
    # ─────────────────────────────────────────────────────────────────────────
    length_rows = [dict(r) for r in base_rows]
    length_indices = []
    extreme_texts = [
        "?",  # 1 char (< min_chars 5 in custom policy)
        "x" * 2500,  # 2,500 chars (> max_chars 2000)
    ]
    sampled_len = random.sample(range(10, 200), 15)
    for i, idx in enumerate(sampled_len):
        length_rows[idx]["text"] = extreme_texts[i % len(extreme_texts)]
        length_indices.append(idx + 1)

    save_csv(BENCH_DIR / "benchmark_length_anomalies.csv", length_rows)
    ground_truth["length_anomalies"] = {
        "file": "benchmark_length_anomalies.csv",
        "description": "15 rows with extreme text lengths (single characters or 2500+ character spam).",
        "target_metric": "length_validity",
        "expected_decision": "BLOCKED",
        "injected_defects": 15,
        "corrupted_row_indices": sorted(length_indices),
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Graded Quality: High Redundancy (Zero Blockers, QI ~95, REVIEW_REQUIRED)
    # ─────────────────────────────────────────────────────────────────────────
    red_rows = []
    for lbl in SELECTED_LABELS:
        lbl_train = [r for r in train_pool if r["label"] == lbl]
        unique_part = lbl_train[:25]
        repeat_query = lbl_train[0]["text"]
        repeated_part = [{"text": repeat_query, "label": lbl, "split": "train"} for _ in range(55)]
        red_rows.extend(unique_part + repeated_part)
    red_rows.extend(test_pool)

    save_csv(BENCH_DIR / "benchmark_high_redundancy.csv", red_rows)
    ground_truth["high_redundancy"] = {
        "file": "benchmark_high_redundancy.csv",
        "description": "Graded Benchmark: 55% repeated boilerplate queries. Tests continuous scoring without hard blockers.",
        "target_metric": "normalized_uniqueness",
        "expected_decision": "REVIEW_REQUIRED",
        "expected_quality_range": [90.0, 96.0],
        "injected_defects": 550,
        "corrupted_row_indices": [],
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Graded Quality: Class Imbalance (Zero Blockers, QI ~97, REVIEW_REQUIRED)
    # ─────────────────────────────────────────────────────────────────────────
    imb_rows = []
    for i, lbl in enumerate(SELECTED_LABELS):
        lbl_train = [r for r in train_pool if r["label"] == lbl]
        if i in (8, 9):
            # 2 tail classes with only 8 training examples (below min 20)
            imb_rows.extend(lbl_train[:8])
        else:
            imb_rows.extend(lbl_train)
    imb_rows.extend(test_pool)

    save_csv(BENCH_DIR / "benchmark_class_imbalance.csv", imb_rows)
    ground_truth["class_imbalance"] = {
        "file": "benchmark_class_imbalance.csv",
        "description": "Graded Benchmark: 2 tail classes have only 8 training rows (score 0.40). Zero blockers.",
        "target_metric": "class_support",
        "expected_decision": "REVIEW_REQUIRED",
        "expected_quality_range": [94.0, 98.0],
        "injected_defects": 2,
        "corrupted_row_indices": [],
    }

    # ─────────────────────────────────────────────────────────────────────────
    # 10. Graded Quality: Realistic Noise (Zero Blockers, QI ~92-94, REVIEW_REQUIRED)
    # ─────────────────────────────────────────────────────────────────────────
    noisy_rows = []
    for i, lbl in enumerate(SELECTED_LABELS):
        lbl_train = [r for r in train_pool if r["label"] == lbl]
        if i in (8, 9):
            noisy_rows.extend(lbl_train[:8])
        else:
            u = lbl_train[:50]
            rep = [dict(lbl_train[0])] * 30
            noisy_rows.extend(u + rep)

    for r in noisy_rows[:15]:
        r["text"] = r["label"][:4] + "?"

    for r in noisy_rows[20:30]:
        r["text"] = r["text"] + " \ufffd"

    noisy_rows.extend(test_pool)

    save_csv(BENCH_DIR / "benchmark_realistic_noise.csv", noisy_rows)
    ground_truth["realistic_noise"] = {
        "file": "benchmark_realistic_noise.csv",
        "description": "Graded Benchmark: Combined mild noise (repetitions + tail classes + short queries). Zero blockers.",
        "target_metric": "normalized_uniqueness",
        "expected_decision": "REVIEW_REQUIRED",
        "expected_quality_range": [88.0, 95.0],
        "injected_defects": 3,
        "corrupted_row_indices": [],
    }

    # Save Ground Truth Manifest
    gt_path = BENCH_DIR / "ground_truth.json"
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    print(f"Generated 11 benchmark datasets in: {BENCH_DIR}")
    print(f"Saved ground truth manifest to: {gt_path}")


if __name__ == "__main__":
    main()
