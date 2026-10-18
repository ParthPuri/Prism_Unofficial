from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from .loader import inspect_columns, iter_rows

ENGINE_VERSION = "0.3.0"
PROMPT_VERSION = "text-policy-v2-context-utility"
SPLITS = {"train", "validation", "test"}
CATALOG = {
    "text_type": "Fraction of rows with a string text value.",
    "text_completeness": "Fraction of rows with nonempty string text after whitespace stripping.",
    "encoding_cleanliness": "Fraction of nonempty texts without replacement characters or unexpected controls; heuristic, not proof of correct encoding.",
    "normalized_uniqueness": "Unique NFKC/casefold/whitespace-normalized texts divided by nonempty text rows; repetition may be legitimate.",
    "length_validity": "Fraction of nonempty texts within policy character limits; not model token limits.",
    "label_completeness": "Fraction of rows with a nonempty scalar label.",
    "label_validity": "Fraction of nonempty scalar labels in the user-supplied allowed label vocabulary.",
    "label_consistency": "Fraction of labeled nonempty text rows whose normalized text does not have conflicting labels; conflict is an ambiguity signal, not proof of which label is wrong.",
    "split_completeness": "Fraction of rows with a recognized train/validation/test split.",
    "split_disjointness": "Fraction of valid evaluation text rows whose normalized text occurs in only one split; exact normalized overlap only.",
    "group_completeness": "Fraction of rows with a nonempty scalar group identifier.",
    "group_disjointness": "Fraction of evaluation rows with group IDs that occur in only one split; use when business requires independent groups.",
    "class_support": "Minimum per-class training support / policy minimum, capped at 1; uses provided vocabulary if present. Not a balance metric.",
    "privacy_pattern_clearance": "Fraction of nonempty text rows without regex email/phone candidates; enable only when the dataset must be redacted. Heuristic with false positives/negatives.",
}
EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<!\w)\+?\d[\d ()-]{7,}\d(?!\w)")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class Config:
    text_column: str = "text"
    label_column: str | None = None
    split_column: str | None = None
    group_column: str | None = None
    allowed_labels: tuple[str, ...] = ()
    require_redacted: bool = False
    expected_purpose: str | None = None

    def __post_init__(self):
        if self.expected_purpose not in (None,"supervised_training","retrieval_corpus","log_monitoring"):
            raise ValueError("Unsupported expected context purpose.")
        if self.group_column and not self.split_column:
            raise ValueError("Group isolation requires a split column.")
        if self.allowed_labels and not self.label_column:
            raise ValueError("Allowed labels require a label column.")
        if len(self.allowed_labels) != len(set(self.allowed_labels)):
            raise ValueError("Allowed labels contain duplicates.")
        if any(not isinstance(x, str) or not x.strip() for x in self.allowed_labels):
            raise ValueError("Allowed labels must be nonempty strings.")


def rows(path: Path, text_column: str = "text") -> Iterator[dict]:
    return iter_rows(path, text_column=text_column)


def scalar(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list, bool)):
        return None
    if not isinstance(value, (str, int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    value = str(value)
    return value if value.strip() else None


class LocalDataset:
    """One streaming ingest into a disposable SQLite index. Raw text is never retained."""

    def __init__(self, path: Path, config: Config):
        self.path, self.config = path, config
        self.temp = tempfile.TemporaryDirectory(prefix="textaudit-")
        self.db = sqlite3.connect(str(Path(self.temp.name) / "index.sqlite"))
        self.db.executescript("""
          PRAGMA journal_mode=OFF;
          CREATE TABLE r(i INTEGER PRIMARY KEY,h TEXT,label TEXT,split TEXT,g TEXT,
            n INTEGER,typ INTEGER,nonempty INTEGER,clean INTEGER,private INTEGER,
            label_ok INTEGER,split_ok INTEGER);
        """)
        self.labels: Counter = Counter()
        self.split_counts: Counter = Counter()
        self.examples: dict[str, list[int]] = {}
        self.count = 0
        try:
            self.input_sha256 = file_digest(path)
            self._ingest()
            if file_digest(path) != self.input_sha256:
                raise ValueError("Input changed during ingestion; retry with a stable file.")
            self.db.executescript("CREATE INDEX ix_h ON r(h); CREATE INDEX ix_g ON r(g); CREATE INDEX ix_ls ON r(label,split);")
        except BaseException:
            self.close()
            raise

    def close(self):
        self.db.close()
        self.temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _ingest(self):
        c = self.config
        batch = []
        required = [x for x in (c.text_column, c.label_column, c.split_column, c.group_column) if x]
        for i, row in enumerate(iter_rows(self.path, text_column=c.text_column), 1):
            if any(k not in row for k in required):
                raise ValueError(f"Row {i} is missing a configured column.")
            text = row[c.text_column]
            typ = isinstance(text, str)
            nonempty = typ and bool(text.strip())
            norm = " ".join(unicodedata.normalize("NFKC", text).casefold().split()) if nonempty else None
            h = hashlib.sha256(norm.encode()).hexdigest() if norm else None
            clean = nonempty and "\ufffd" not in text and not any(
                unicodedata.category(ch) == "Cc" and ch not in "\t\r\n" for ch in text
            )
            private = nonempty and bool(EMAIL.search(text) or PHONE.search(text))
            label = scalar(row.get(c.label_column)) if c.label_column else None
            split = scalar(row.get(c.split_column)) if c.split_column else None
            group = scalar(row.get(c.group_column)) if c.group_column else None
            group = hashlib.sha256(group.encode()).hexdigest() if group else None
            label_ok = label is not None and (not c.allowed_labels or label in c.allowed_labels)
            split_ok = split in SPLITS
            if label is not None:
                self.labels[label] += 1
            if split is not None:
                self.split_counts[split] += 1
            for key, flag in [("text_completeness", not nonempty), ("text_type", not typ),
                              ("encoding_cleanliness", nonempty and not clean),
                              ("label_completeness", c.label_column and label is None),
                              ("label_validity", c.allowed_labels and label is not None and not label_ok),
                              ("split_completeness", c.split_column and not split_ok),
                              ("group_completeness", c.group_column and not group),
                              ("privacy_pattern_clearance", private)]:
                if flag and len(self.examples.get(key, [])) < 10:
                    self.examples.setdefault(key, []).append(i)
            batch.append((i,h,label,split,group,len(text) if typ else 0,int(typ),int(nonempty),
                          int(clean),int(private),int(label_ok),int(split_ok)))
            self.count = i
            if len(batch) == 2000:
                self.db.executemany("INSERT INTO r VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", batch)
                batch.clear()
        if batch:
            self.db.executemany("INSERT INTO r VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", batch)
        self.db.commit()

    def number(self, sql: str, args=()) -> int:
        return self.db.execute(sql, args).fetchone()[0] or 0

    def applicable(self, context: dict | None = None) -> list[str]:
        result = ["text_type", "text_completeness", "encoding_cleanliness", "normalized_uniqueness", "length_validity"]
        if self.config.label_column:
            result += ["label_completeness", "label_consistency", "class_support"]
        if self.config.allowed_labels:
            result += ["label_validity"]
        if self.config.split_column:
            result += ["split_completeness", "split_disjointness"]
        if self.config.group_column:
            result += ["group_completeness", "group_disjointness"]
        if self.config.require_redacted or (context and context["privacy"] == "redacted"):
            result += ["privacy_pattern_clearance"]
        if context and context["purpose"] != "supervised_training":
            result = [x for x in result if not x.startswith(("label_","split_","group_")) and x != "class_support"]
        return result

    def mandatory(self, context: dict | None = None) -> dict[str, float]:
        result = {"text_type": 1.0, "text_completeness": 0.99}
        training = context is None or context["purpose"] == "supervised_training"
        if self.config.label_column and training:
            result.update(label_completeness=0.99, label_consistency=1.0)
        if self.config.allowed_labels and training:
            result.update(label_validity=1.0)
        if self.config.split_column and training:
            result.update(split_completeness=1.0, split_disjointness=1.0)
        if self.config.group_column and training:
            result.update(group_completeness=1.0, group_disjointness=1.0)
        if self.config.require_redacted or (context and context["privacy"] == "redacted"):
            result.update(privacy_pattern_clearance=1.0)
        return result

    def profile(self) -> dict:
        # Never send raw text, label names, group identifiers, file paths, or column names.
        return {
            "rows": self.count,
            "nonempty_text_rows": self.number("SELECT SUM(nonempty) FROM r"),
            "label_classes_observed": len(self.labels),
            "minimum_observed_class_count": min(self.labels.values(), default=0),
            "recognized_split_counts": {s: self.split_counts[s] for s in sorted(SPLITS)},
            "has_labels": bool(self.config.label_column),
            "has_allowed_label_vocabulary": bool(self.config.allowed_labels),
            "has_splits": bool(self.config.split_column),
            "has_groups": bool(self.config.group_column),
            "require_redacted": self.config.require_redacted,
        }

    def metric(self, name: str, parameters: dict) -> dict:
        n = self.count
        nonempty = self.number("SELECT SUM(nonempty) FROM r")
        total, good = n, 0
        details = ""
        example_ids = self.examples.get(name, [])
        if name == "text_type":
            good = self.number("SELECT SUM(typ) FROM r")
        elif name == "text_completeness":
            good = nonempty
        elif name == "encoding_cleanliness":
            total, good = nonempty, self.number("SELECT SUM(clean) FROM r")
        elif name == "normalized_uniqueness":
            total, good = nonempty, self.number("SELECT COUNT(DISTINCT h) FROM r")
            details = "Normalized repetition is a signal; recurring messages can be legitimate."
        elif name == "length_validity":
            total = nonempty
            low, high = parameters["min_chars"], parameters["max_chars"]
            good = self.number("SELECT COUNT(*) FROM r WHERE nonempty=1 AND n BETWEEN ? AND ?", (low,high))
            example_ids = [x[0] for x in self.db.execute("SELECT i FROM r WHERE nonempty=1 AND (n<? OR n>?) ORDER BY i LIMIT 10", (low,high))]
        elif name == "label_completeness":
            good = self.number("SELECT COUNT(label) FROM r")
        elif name == "label_validity":
            total, good = self.number("SELECT COUNT(label) FROM r"), self.number("SELECT SUM(label_ok) FROM r")
        elif name == "label_consistency":
            total = self.number("SELECT COUNT(*) FROM r WHERE h IS NOT NULL AND label IS NOT NULL")
            bad_sql = "SELECT i FROM r WHERE label IS NOT NULL AND h IN (SELECT h FROM r WHERE h IS NOT NULL AND label IS NOT NULL GROUP BY h HAVING COUNT(DISTINCT label)>1)"
            bad = self.number("SELECT COUNT(*) FROM ("+bad_sql+")")
            good = total-bad
            example_ids = [x[0] for x in self.db.execute(bad_sql+" ORDER BY i LIMIT 10")]
            details = "Conflicting labels require review; context-dependent ambiguity is possible."
        elif name == "split_completeness":
            good = self.number("SELECT SUM(split_ok) FROM r")
        elif name in ("split_disjointness", "group_disjointness"):
            column = "h" if name == "split_disjointness" else "g"
            total = self.number(f"SELECT COUNT(*) FROM r WHERE {column} IS NOT NULL AND split IN ('validation','test')")
            bad_sql = f"SELECT i FROM r WHERE split IN ('validation','test') AND {column} IN (SELECT {column} FROM r WHERE {column} IS NOT NULL AND split_ok=1 GROUP BY {column} HAVING COUNT(DISTINCT split)>1)"
            good = total-self.number("SELECT COUNT(*) FROM ("+bad_sql+")")
            example_ids = [x[0] for x in self.db.execute(bad_sql+" ORDER BY i LIMIT 10")]
            details = "Counts evaluation rows shared across recognized splits; normalization-based overlap, not semantic leakage."
        elif name == "group_completeness":
            good = self.number("SELECT COUNT(g) FROM r")
        elif name == "privacy_pattern_clearance":
            total, good = nonempty, nonempty-self.number("SELECT SUM(private) FROM r")
            details = "Regex candidates only; not a full PII audit or privacy certification."
        elif name == "class_support":
            where = "label IS NOT NULL AND nonempty=1 AND label_ok=1"
            if self.config.split_column:
                where += " AND split='train'"
            counts = dict(self.db.execute("SELECT label,COUNT(*) FROM r WHERE "+where+" GROUP BY label"))
            vocabulary = self.config.allowed_labels or tuple(self.labels)
            if not vocabulary:
                return {"metric":name,"score":None,"evaluated_rows":0,"reason":"No label vocabulary observed or supplied."}
            minimum = min(counts.get(label, 0) for label in vocabulary)
            return {"metric":name,"score":min(1.0,minimum/parameters["min_class_examples"]),
                    "evaluated_rows":sum(counts.values()),"minimum_class_examples":minimum,
                    "target_minimum":parameters["min_class_examples"],"issue_row_ids":[],
                    "details":"Support adequacy against a provisional target; not semantic label accuracy or representativeness."}
        else:
            raise ValueError("Unknown metric: "+name)
        return {"metric":name,"score":good/total if total else None,"evaluated_rows":total,
                "good_rows":good,"bad_rows":total-good,"issue_row_ids":example_ids,
                "details":details,"reason":"No eligible rows." if not total else None}


def numeric(x: Any, low: float, high: float) -> bool:
    return type(x) in (int,float) and math.isfinite(x) and low <= x <= high


def validate_context(context: Any, dataset: LocalDataset) -> dict:
    if not isinstance(context,dict) or set(context) != {"purpose","privacy","description","assumptions"}:
        raise ValueError("Context needs exactly purpose, privacy, description and assumptions.")
    if context["purpose"] not in ("supervised_training","retrieval_corpus","log_monitoring"):
        raise ValueError("Unsupported context purpose.")
    expected = dataset.config.expected_purpose
    if expected and context["purpose"] != expected:
        raise ValueError("AI context purpose conflicts with the declared business purpose.")
    if context["purpose"] == "supervised_training" and not dataset.config.label_column:
        raise ValueError("Supervised training assessment needs a configured label column.")
    if context["privacy"] not in ("retain","redacted"):
        raise ValueError("Privacy must be retain or redacted.")
    if dataset.config.require_redacted and context["privacy"] != "redacted":
        raise ValueError("Context cannot waive the declared redaction requirement.")
    if not isinstance(context["description"],str) or not 1 <= len(context["description"]) <= 2000:
        raise ValueError("Context description must have 1..2000 characters.")
    a = context["assumptions"]
    if not isinstance(a,list) or len(a)>20 or any(not isinstance(s,str) or not 1<=len(s)<=1000 for s in a):
        raise ValueError("Context assumptions must be at most 20 nonempty strings.")
    return context


def utility(raw: float | None, curve: dict | None) -> float | None:
    """Bounded monotone utility, where anchors express context-specific tolerances."""
    if raw is None or curve is None:
        return raw
    return max(0.0,min(1.0,(raw-curve["bad"])/(curve["good"]-curve["bad"])))


def validate_policy(policy: Any, dataset: LocalDataset) -> dict:
    if not isinstance(policy,dict) or type(policy.get("version")) is not int or policy["version"] not in (1,2):
        raise ValueError("Unsupported policy version.")
    required_keys = {"version","metrics","parameters","acceptance_score","rationale"}
    if policy["version"] == 2:
        required_keys.add("context")
    if set(policy) != required_keys:
        raise ValueError("Unexpected policy fields.")
    context = validate_context(policy["context"],dataset) if policy["version"]==2 else None
    if context is None and dataset.config.expected_purpose:
        raise ValueError("A declared purpose requires a version-2 contextual policy.")
    if not isinstance(policy["rationale"], str) or not 1 <= len(policy["rationale"]) <= 2000:
        raise ValueError("Policy rationale must contain 1..2000 characters.")
    p = policy["parameters"]
    if not isinstance(p, dict) or set(p) != {"min_chars","max_chars","min_class_examples"}:
        raise ValueError("Unexpected metric parameters.")
    if any(type(p[k]) is not int or not 1 <= p[k] <= 1_000_000 for k in p):
        raise ValueError("Parameters must be integer values between 1 and 1,000,000.")
    if p["min_chars"] > p["max_chars"]:
        raise ValueError("min_chars must not exceed max_chars.")
    if not numeric(policy["acceptance_score"],80,100):
        raise ValueError("acceptance_score must be between 80 and 100.")
    metrics = policy["metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("Policy needs metrics.")
    mandatory = dataset.mandatory(context)
    found = set()
    for m in metrics:
        keys = {"name","weight","pass_min","gate","rationale"}
        if context:
            keys.add("utility")
        if not isinstance(m, dict) or set(m) != keys:
            raise ValueError("Every metric needs name, weight, pass_min, gate, rationale.")
        name = m["name"]
        if not isinstance(name,str) or name not in dataset.applicable(context) or name in found:
            raise ValueError("Unknown, inapplicable, or repeated metric.")
        if context:
            u = m["utility"]
            if not isinstance(u,dict) or set(u)!={"type","bad","good"} or u["type"]!="linear":
                raise ValueError("Utility needs type=linear and bad/good anchors.")
            if not numeric(u["bad"],0,1) or not numeric(u["good"],0,1) or u["bad"]>=u["good"]:
                raise ValueError("Utility anchors require 0 <= bad < good <= 1.")
        found.add(name)
        if not numeric(m["weight"],0.001,0.5) or not numeric(m["pass_min"],0,1):
            raise ValueError("Weight must be 0.001..0.5 and pass_min must be 0..1.")
        if type(m["gate"]) is not bool or not isinstance(m["rationale"], str) or not 1 <= len(m["rationale"]) <= 1000:
            raise ValueError("Invalid gate or metric rationale.")
        if name in mandatory and (not m["gate"] or m["pass_min"] < mandatory[name]):
            raise ValueError(f"Mandatory gate {name} requires pass_min >= {mandatory[name]}.")
    if not set(mandatory) <= found:
        raise ValueError("Policy omitted mandatory metrics: "+", ".join(sorted(set(mandatory)-found)))
    if not math.isclose(sum(m["weight"] for m in metrics),1.0,abs_tol=1e-6,rel_tol=0):
        raise ValueError("Weights must sum to 1; invalid weights are not silently normalized.")
    return policy


def evaluate(dataset: LocalDataset, policy: dict, metric_runner=None) -> dict:
    validate_policy(policy,dataset)
    measured, coverage = [], 0.0
    for m in policy["metrics"]:
        result = (metric_runner or dataset.metric)(m["name"],policy["parameters"])
        result.setdefault("executor","local_sqlite")
        score = result["score"]
        contextual_score = utility(score,m.get("utility"))
        result.update(weight=m["weight"],pass_min=m["pass_min"],gate=m["gate"],rationale=m["rationale"])
        result.update(contextual_score=contextual_score,utility=m.get("utility",{"type":"identity"}))
        result["status"] = "NOT_EVALUATED" if score is None else "PASS" if score >= m["pass_min"] else "FAIL"
        result["weighted_points"] = 100*m["weight"]*contextual_score if contextual_score is not None else None
        if score is not None:
            coverage += m["weight"]
        measured.append(result)
    complete = math.isclose(coverage,1.0,abs_tol=1e-6,rel_tol=0)
    partial_points = sum(r["weighted_points"] or 0 for r in measured)
    score = partial_points if complete else None
    failing_gates = [r for r in measured if r["gate"] and r["status"] != "PASS"]
    blockers = [r["metric"] for r in failing_gates]
    if dataset.count == 0:
        blockers.append("empty_dataset")
    decision = "BLOCKED" if blockers else "INCOMPLETE" if not complete else "REVIEW_REQUIRED" if score < policy["acceptance_score"] or any(r["status"] == "FAIL" for r in measured) else "CHECKS_PASSED_REVIEW_REQUIRED"
    if score is None or dataset.count == 0:
        effective_score = 0.0 if dataset.count == 0 else None
        gate_penalty = 0.0
    elif blockers:
        gate_penalty = round(sum(25.0 + 50.0 * (1.0 - (r["score"] or 0.0)) for r in failing_gates), 2)
        effective_score = max(0.0, min(79.9, round(score - gate_penalty, 2)))
    else:
        gate_penalty = 0.0
        effective_score = score
    return {
        "engine_version":ENGINE_VERSION,"prompt_version":PROMPT_VERSION,
        "input_sha256":dataset.input_sha256,"config_sha256":digest(asdict(dataset.config)),
        "policy_sha256":digest(policy),"config":asdict(dataset.config),"profile":dataset.profile(),
        "quality_index":score,"effective_quality_index":effective_score,
        "raw_quality_index":score,"gate_penalty":gate_penalty,"is_blocked":bool(blockers),
        "evaluated_weight_fraction":coverage,
        "measured_weight_score":partial_points/coverage if coverage else None,
        "acceptance_score":policy["acceptance_score"],"decision":decision,"blockers":blockers,
        "metrics":measured,"policy":policy,
        "context":policy.get("context"),
        "limits":["Provisional policy score, not a calibrated probability of fitness or a production certification.",
                  "Cannot prove semantic label correctness, fairness, deployment representativeness, or future drift.",
                  "Text/group overlap is not exhaustive leakage detection; no embeddings or temporal checks in v0.1.",
                  "No raw texts or label names were sent to the planner. Business use case text is sent as entered.",
                  "Full scan of eligible rows. Missing eligibility is shown explicitly; no sampling confidence claimed.",
                  "AI-interpreted context must be reviewed. Use --purpose to constrain task scope. Utility anchors are provisional and require downstream calibration."]}
