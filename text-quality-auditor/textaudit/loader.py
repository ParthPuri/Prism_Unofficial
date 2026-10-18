"""
Multi-format dataset loader subsystem for Text Quality Auditor.

Supports standard NLP dataset formats from Hugging Face and Kaggle:
- Delimited text: .csv, .tsv, .tab
- Line-delimited JSON: .jsonl, .ndjson
- Structured JSON: .json (arrays, envelopes, split maps, columnar)
- Columnar binary: .parquet (via PyArrow)
- In-memory / IPC: .arrow, .feather (via PyArrow)
- Plain text corpora: .txt (unstructured lines)
"""

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterator


class BaseDatasetLoader(ABC):
    """Abstract base loader for textual datasets."""

    @abstractmethod
    def iter_rows(self, path: Path, text_column: str = "text") -> Iterator[dict[str, Any]]:
        """Stream dataset records as dictionaries."""
        raise NotImplementedError

    @abstractmethod
    def inspect_columns(self, path: Path, text_column: str = "text") -> list[str]:
        """Inspect and return available column headers without loading all data."""
        raise NotImplementedError


class DelimitedLoader(BaseDatasetLoader):
    """Loader for delimited tabular text (.csv, .tsv, .tab)."""

    def __init__(self, delimiter: str = ",", format_name: str = "CSV"):
        self.delimiter = delimiter
        self.format_name = format_name

    def inspect_columns(self, path: Path, text_column: str = "text") -> list[str]:
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            try:
                headers = next(reader)
            except StopIteration:
                return []
            if not headers:
                return []
            return [h.strip() for h in headers if h.strip()]

    def iter_rows(self, path: Path, text_column: str = "text") -> Iterator[dict[str, Any]]:
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError(f"{self.format_name} needs a header with unique column names.")
            for i, row in enumerate(reader, 2):
                if None in row or any(v is None for v in row.values()):
                    raise ValueError(
                        f"Malformed {self.format_name} row at line {i}; no rows were silently skipped."
                    )
                yield row


class JsonLinesLoader(BaseDatasetLoader):
    """Loader for line-delimited JSON (.jsonl, .ndjson)."""

    def inspect_columns(self, path: Path, text_column: str = "text") -> list[str]:
        with path.open(encoding="utf-8-sig") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        obj = json.loads(line_str)
                        if isinstance(obj, dict):
                            return list(obj.keys())
                    except Exception:
                        return []
        return []

    def iter_rows(self, path: Path, text_column: str = "text") -> Iterator[dict[str, Any]]:
        with path.open(encoding="utf-8-sig") as f:
            for i, line in enumerate(f, 1):
                if not line.strip():
                    raise ValueError(f"Empty JSONL line {i}; no rows were silently skipped.")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL line {i} must be an object.")
                yield row


class JsonStructuredLoader(BaseDatasetLoader):
    """
    Loader for structured JSON files (.json).

    Handles standard shapes:
    1. List of records: [{"text": "...", "label": "..."}, ...]
    2. Envelope dictionary: {"data": [...]} or {"rows": [...]} or {"records": [...]}
    3. Split dictionary: {"train": [...], "test": [...]} (auto-injects split column)
    4. Columnar dictionary: {"text": ["a", "b"], "label": ["0", "1"]}
    """

    ENVELOPE_KEYS = ("data", "rows", "records", "instances", "items", "dataset", "examples")

    def _load_data(self, path: Path) -> Any:
        with path.open(encoding="utf-8-sig") as f:
            return json.load(f)

    def inspect_columns(self, path: Path, text_column: str = "text") -> list[str]:
        data = self._load_data(path)
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                return list(data[0].keys())
            return []
        if isinstance(data, dict):
            # Check envelope
            for key in self.ENVELOPE_KEYS:
                if key in data and isinstance(data[key], list) and data[key] and isinstance(data[key][0], dict):
                    return list(data[key][0].keys())
            # Check split mapping
            if any(k.lower() in ("train", "validation", "test", "val", "dev") for k in data) and any(isinstance(v, list) for v in data.values()):
                cols = set()
                for v in data.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        cols.update(v[0].keys())
                cols.add("split")
                return sorted(cols)
            # Check columnar
            if data and all(isinstance(v, list) for v in data.values()):
                return list(data.keys())
        return []

    def iter_rows(self, path: Path, text_column: str = "text") -> Iterator[dict[str, Any]]:
        data = self._load_data(path)
        if isinstance(data, list):
            for i, item in enumerate(data, 1):
                if not isinstance(item, dict):
                    raise ValueError(f"JSON record {i} must be an object.")
                yield item
            return

        if isinstance(data, dict):
            # 1. Envelope dict
            for key in self.ENVELOPE_KEYS:
                if key in data and isinstance(data[key], list):
                    for i, item in enumerate(data[key], 1):
                        if not isinstance(item, dict):
                            raise ValueError(f"JSON envelope record {i} in '{key}' must be an object.")
                        yield item
                    return

            # 2. Split mapping: {"train": [...], "test": [...]}
            split_keys = [k for k, v in data.items() if isinstance(v, list)]
            if split_keys and any(k.lower() in ("train", "val", "validation", "test", "dev") for k in split_keys):
                total_i = 1
                for split_name, records in data.items():
                    if not isinstance(records, list):
                        continue
                    # Canonicalize split name
                    norm_split = "validation" if split_name.lower() in ("val", "dev") else split_name.lower()
                    for item in records:
                        if not isinstance(item, dict):
                            raise ValueError(f"JSON record {total_i} in split '{split_name}' must be an object.")
                        row = dict(item)
                        row.setdefault("split", norm_split)
                        yield row
                        total_i += 1
                return

            # 3. Columnar dictionary: {"text": [...], "label": [...]}
            if data and all(isinstance(v, list) for v in data.values()):
                lengths = {len(v) for v in data.values()}
                if len(lengths) > 1:
                    raise ValueError("JSON columnar format has mismatched column array lengths.")
                num_rows = lengths.pop()
                cols = list(data.keys())
                for i in range(num_rows):
                    yield {col: data[col][i] for col in cols}
                return

        raise ValueError(
            "Unsupported JSON dataset structure. Expected an array of objects, "
            "an envelope with a records list (e.g. 'data', 'rows'), a split mapping (e.g. 'train', 'test'), "
            "or a columnar object."
        )


class ParquetLoader(BaseDatasetLoader):
    """Loader for Apache Parquet datasets (.parquet)."""

    def _get_pq(self):
        try:
            import pyarrow.parquet as pq
            return pq
        except ImportError:
            raise ImportError(
                "PyArrow is required to read Parquet datasets. "
                "Install with: pip install 'text-quality-auditor[arrow]' or pip install pyarrow"
            )

    def inspect_columns(self, path: Path, text_column: str = "text") -> list[str]:
        pq = self._get_pq()
        with open(path, "rb") as f:
            with pq.ParquetFile(f) as pf:
                return list(pf.schema.names)

    def iter_rows(self, path: Path, text_column: str = "text") -> Iterator[dict[str, Any]]:
        pq = self._get_pq()
        with open(path, "rb") as f:
            with pq.ParquetFile(f) as pf:
                for batch in pf.iter_batches(batch_size=2000):
                    for row in batch.to_pylist():
                        yield row


class ArrowLoader(BaseDatasetLoader):
    """Loader for Apache Arrow / Feather IPC files and streams (.arrow, .feather)."""

    def _get_pyarrow(self):
        try:
            import pyarrow as pa
            import pyarrow.feather as feather
            import pyarrow.ipc as ipc
            return pa, ipc, feather
        except ImportError:
            raise ImportError(
                "PyArrow is required to read Arrow datasets. "
                "Install with: pip install 'text-quality-auditor[arrow]' or pip install pyarrow"
            )

    def inspect_columns(self, path: Path, text_column: str = "text") -> list[str]:
        pa, ipc, feather = self._get_pyarrow()
        with open(path, "rb") as f:
            try:
                with ipc.open_file(f) as reader:
                    return list(reader.schema.names)
            except Exception:
                pass
            f.seek(0)
            try:
                with ipc.open_stream(f) as reader:
                    return list(reader.schema.names)
            except Exception:
                pass
            f.seek(0)
            try:
                table = feather.read_table(f)
                return list(table.schema.names)
            except Exception:
                return []

    def iter_rows(self, path: Path, text_column: str = "text") -> Iterator[dict[str, Any]]:
        pa, ipc, feather = self._get_pyarrow()
        with open(path, "rb") as f:
            # 1. Try Arrow IPC File (Random Access / Feather v2)
            try:
                with ipc.open_file(f) as reader:
                    for i in range(reader.num_record_batches):
                        batch = reader.get_batch(i)
                        for row in batch.to_pylist():
                            yield row
                    return
            except Exception:
                pass

            # 2. Try Arrow IPC Stream
            f.seek(0)
            try:
                with ipc.open_stream(f) as reader:
                    for batch in reader:
                        for row in batch.to_pylist():
                            yield row
                    return
            except Exception:
                pass

            # 3. Try Feather fallback
            f.seek(0)
            table = feather.read_table(f)
            for row in table.to_pylist():
                yield row


class PlainTextLoader(BaseDatasetLoader):
    """Loader for unstructured plain text corpora (.txt)."""

    def inspect_columns(self, path: Path, text_column: str = "text") -> list[str]:
        return [text_column]

    def iter_rows(self, path: Path, text_column: str = "text") -> Iterator[dict[str, Any]]:
        with path.open(encoding="utf-8-sig") as f:
            for line in f:
                yield {text_column: line.rstrip("\r\n")}


# ── Loader Registry & Dispatcher ─────────────────────────────────────────────

LOADER_REGISTRY: dict[str, BaseDatasetLoader] = {
    ".csv": DelimitedLoader(delimiter=",", format_name="CSV"),
    ".tsv": DelimitedLoader(delimiter="\t", format_name="TSV"),
    ".tab": DelimitedLoader(delimiter="\t", format_name="TSV"),
    ".jsonl": JsonLinesLoader(),
    ".ndjson": JsonLinesLoader(),
    ".json": JsonStructuredLoader(),
    ".parquet": ParquetLoader(),
    ".arrow": ArrowLoader(),
    ".feather": ArrowLoader(),
    ".txt": PlainTextLoader(),
}


def get_supported_extensions() -> list[str]:
    """Return all supported file extensions."""
    return sorted(LOADER_REGISTRY.keys())


def sniff_loader(path: Path) -> BaseDatasetLoader:
    """Sniff file magic bytes or content when file extension is absent or ambiguous."""
    try:
        with open(path, "rb") as f:
            magic = f.read(1024)
    except Exception:
        return LOADER_REGISTRY[".txt"]

    if magic.startswith(b"PAR1"):
        return LOADER_REGISTRY[".parquet"]
    if magic.startswith(b"ARROW1"):
        return LOADER_REGISTRY[".arrow"]
    stripped = magic.lstrip()
    if stripped.startswith(b"{") or stripped.startswith(b"["):
        try:
            first_line = stripped.split(b"\n")[0].decode("utf-8")
            json.loads(first_line)
            if b"\n" in stripped:
                return LOADER_REGISTRY[".jsonl"]
            return LOADER_REGISTRY[".json"]
        except Exception:
            return LOADER_REGISTRY[".json"]
    try:
        first_line_str = magic.decode("utf-8-sig", errors="ignore").splitlines()[0]
        if "\t" in first_line_str and "," not in first_line_str:
            return LOADER_REGISTRY[".tsv"]
        if "," in first_line_str:
            return LOADER_REGISTRY[".csv"]
    except Exception:
        pass
    return LOADER_REGISTRY[".txt"]


def get_loader(path: Path) -> BaseDatasetLoader:
    """Resolve the appropriate loader for a given file path."""
    suffix = path.suffix.lower()
    if suffix in LOADER_REGISTRY:
        return LOADER_REGISTRY[suffix]
    return sniff_loader(path)


def inspect_columns(path: Path, text_column: str = "text") -> list[str]:
    """Inspect column names without loading the whole dataset."""
    loader = get_loader(path)
    return loader.inspect_columns(path, text_column=text_column)


def iter_rows(path: Path, text_column: str = "text") -> Iterator[dict[str, Any]]:
    """Iterate rows from any supported textual dataset format."""
    loader = get_loader(path)
    return loader.iter_rows(path, text_column=text_column)
