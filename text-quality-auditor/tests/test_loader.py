import json
import tempfile
import unittest
from pathlib import Path

from textaudit.core import Config, LocalDataset, evaluate
from textaudit.loader import (
    get_loader,
    get_supported_extensions,
    inspect_columns,
    iter_rows,
)

try:
    import pyarrow as pa
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


def minimal_policy(dataset):
    mandatory = dataset.mandatory()
    names = list(mandatory)
    weight = 1.0 / len(names)
    return {
        "version": 1,
        "metrics": [
            {
                "name": n,
                "weight": weight,
                "pass_min": mandatory.get(n, 0.9),
                "gate": n in mandatory,
                "rationale": "Automated test policy",
            }
            for n in names
        ],
        "parameters": {"min_chars": 1, "max_chars": 1000, "min_class_examples": 1},
        "acceptance_score": 90,
        "rationale": "Automated test policy",
    }


class MultiFormatLoaderTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def test_supported_extensions(self):
        exts = get_supported_extensions()
        for expected in [".csv", ".tsv", ".json", ".jsonl", ".parquet", ".arrow", ".txt"]:
            self.assertIn(expected, exts)

    def test_csv_and_tsv_loading(self):
        # CSV with BOM
        csv_path = self.root / "data.csv"
        csv_path.write_text("\ufefftext,label,split\nHello world,greeting,train\nGoodbye,farewell,test\n", encoding="utf-8")
        cols = inspect_columns(csv_path)
        self.assertEqual(cols, ["text", "label", "split"])
        rows = list(iter_rows(csv_path))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["text"], "Hello world")
        self.assertEqual(rows[0]["label"], "greeting")

        # TSV
        tsv_path = self.root / "data.tsv"
        tsv_path.write_text("text\tlabel\tsplit\nGood morning\tgreeting\ttrain\nSee you\tfarewell\ttest\n", encoding="utf-8")
        self.assertEqual(inspect_columns(tsv_path), ["text", "label", "split"])
        rows = list(iter_rows(tsv_path))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["label"], "farewell")

    def test_delimited_malformed_rows(self):
        csv_path = self.root / "bad.csv"
        csv_path.write_text("text,label\nhello\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            list(iter_rows(csv_path))

    def test_jsonlines_loading(self):
        jsonl_path = self.root / "data.jsonl"
        records = [
            {"text": "Sample query 1", "label": "intent_a", "split": "train"},
            {"text": "Sample query 2", "label": "intent_b", "split": "test"},
        ]
        jsonl_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        self.assertEqual(inspect_columns(jsonl_path), ["text", "label", "split"])
        rows = list(iter_rows(jsonl_path))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["text"], "Sample query 1")

    def test_json_list_of_objects(self):
        json_path = self.root / "records.json"
        records = [
            {"text": "Sample query 1", "label": "intent_a", "split": "train"},
            {"text": "Sample query 2", "label": "intent_b", "split": "test"},
        ]
        json_path.write_text(json.dumps(records), encoding="utf-8")
        self.assertEqual(inspect_columns(json_path), ["text", "label", "split"])
        rows = list(iter_rows(json_path))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["text"], "Sample query 1")

    def test_json_envelope_dict(self):
        json_path = self.root / "envelope.json"
        content = {
            "version": "1.0",
            "data": [
                {"text": "Sample query 1", "label": "intent_a", "split": "train"},
                {"text": "Sample query 2", "label": "intent_b", "split": "test"},
            ],
        }
        json_path.write_text(json.dumps(content), encoding="utf-8")
        self.assertEqual(inspect_columns(json_path), ["text", "label", "split"])
        rows = list(iter_rows(json_path))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["text"], "Sample query 2")

    def test_json_split_mapping(self):
        json_path = self.root / "splits.json"
        content = {
            "train": [{"text": "Train sample", "label": "intent_a"}],
            "test": [{"text": "Test sample", "label": "intent_b"}],
        }
        json_path.write_text(json.dumps(content), encoding="utf-8")
        cols = inspect_columns(json_path)
        self.assertIn("text", cols)
        self.assertIn("label", cols)
        self.assertIn("split", cols)
        rows = list(iter_rows(json_path))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["split"], "train")
        self.assertEqual(rows[1]["split"], "test")

    def test_json_columnar(self):
        json_path = self.root / "columnar.json"
        content = {
            "text": ["Row 1", "Row 2"],
            "label": ["A", "B"],
            "split": ["train", "test"],
        }
        json_path.write_text(json.dumps(content), encoding="utf-8")
        self.assertEqual(sorted(inspect_columns(json_path)), ["label", "split", "text"])
        rows = list(iter_rows(json_path))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0], {"text": "Row 1", "label": "A", "split": "train"})
        self.assertEqual(rows[1], {"text": "Row 2", "label": "B", "split": "test"})

    def test_plain_text_corpus(self):
        txt_path = self.root / "corpus.txt"
        txt_path.write_text("First line of text\nSecond line of text\nThird line of text\n", encoding="utf-8")
        self.assertEqual(inspect_columns(txt_path), ["text"])
        rows = list(iter_rows(txt_path))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["text"], "First line of text")
        self.assertEqual(rows[2]["text"], "Third line of text")

    @unittest.skipUnless(HAS_PYARROW, "pyarrow not installed")
    def test_parquet_loading(self):
        parquet_path = self.root / "dataset.parquet"
        t = pa.table({
            "text": ["Parquet query 1", "Parquet query 2"],
            "label": ["greeting", "farewell"],
            "split": ["train", "test"],
        })
        with open(parquet_path, "wb") as f:
            pq.write_table(t, f)

        cols = inspect_columns(parquet_path)
        self.assertEqual(cols, ["text", "label", "split"])
        rows = list(iter_rows(parquet_path))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["text"], "Parquet query 1")
        self.assertEqual(rows[1]["label"], "farewell")

    @unittest.skipUnless(HAS_PYARROW, "pyarrow not installed")
    def test_arrow_ipc_loading(self):
        arrow_path = self.root / "dataset.arrow"
        t = pa.table({
            "text": ["Arrow query 1", "Arrow query 2"],
            "label": ["intent_a", "intent_b"],
            "split": ["train", "test"],
        })
        with open(arrow_path, "wb") as f:
            with ipc.new_file(f, t.schema) as writer:
                writer.write_table(t)

        cols = inspect_columns(arrow_path)
        self.assertEqual(cols, ["text", "label", "split"])
        rows = list(iter_rows(arrow_path))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["text"], "Arrow query 1")

    def test_sniff_extensionless_file(self):
        no_ext = self.root / "dataset_without_extension"
        no_ext.write_text("text,label\nhi,a\nbye,b\n", encoding="utf-8")
        loader = get_loader(no_ext)
        self.assertEqual(loader.inspect_columns(no_ext), ["text", "label"])
        rows = list(iter_rows(no_ext))
        self.assertEqual(len(rows), 2)

    def test_end_to_end_local_dataset_across_formats(self):
        formats_to_test = [
            ("data.csv", "text,label,split\nHello,intent_a,train\nWorld,intent_a,test\n"),
            ("data.tsv", "text\tlabel\tsplit\nHello\tintent_a\ttrain\nWorld\tintent_a\ttest\n"),
            ("data.jsonl", '{"text":"Hello","label":"intent_a","split":"train"}\n{"text":"World","label":"intent_a","split":"test"}\n'),
            ("data.json", json.dumps([
                {"text": "Hello", "label": "intent_a", "split": "train"},
                {"text": "World", "label": "intent_a", "split": "test"},
            ])),
        ]

        if HAS_PYARROW:
            p_pq = self.root / "test.parquet"
            t = pa.table({"text": ["Hello", "World"], "label": ["intent_a", "intent_a"], "split": ["train", "test"]})
            with open(p_pq, "wb") as f:
                pq.write_table(t, f)
            formats_to_test.append(("test.parquet", None))

            p_arr = self.root / "test.arrow"
            with open(p_arr, "wb") as f:
                with ipc.new_file(f, t.schema) as writer:
                    writer.write_table(t)
            formats_to_test.append(("test.arrow", None))

        config = Config(text_column="text", label_column="label", split_column="split")
        for filename, content in formats_to_test:
            p = self.root / filename
            if content is not None:
                p.write_text(content, encoding="utf-8")
            with LocalDataset(p, config) as ds:
                self.assertEqual(ds.count, 2)
                p_policy = minimal_policy(ds)
                report = evaluate(ds, p_policy)
                self.assertEqual(report["decision"], "CHECKS_PASSED_REVIEW_REQUIRED")
                self.assertAlmostEqual(report["quality_index"], 100.0)


if __name__ == "__main__":
    unittest.main()
