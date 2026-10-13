import copy
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from textaudit.core import Config, LocalDataset, evaluate, validate_policy
from textaudit.planner import plan


def policy(dataset, extras=()):
    mandatory = dataset.mandatory()
    names = list(mandatory)+list(extras)
    weight = 1/len(names)
    return {"version":1,"metrics":[{"name":n,"weight":weight,
        "pass_min":mandatory.get(n,0.9),"gate":n in mandatory,"rationale":"Test policy"} for n in names],
        "parameters":{"min_chars":1,"max_chars":1000,"min_class_examples":2},
        "acceptance_score":90,"rationale":"Explicit test policy"}


class AuditorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def dataset(self,records,config=Config()):
        path = self.root/"input.jsonl"
        path.write_text("".join(json.dumps(r)+"\n" for r in records),encoding="utf-8")
        d = LocalDataset(path,config)
        self.addCleanup(d.close)
        return d

    def test_score_and_full_coverage(self):
        d = self.dataset([{"text":"alpha"},{"text":"beta"}])
        result = evaluate(d,policy(d))
        self.assertEqual(result["quality_index"],100)
        self.assertEqual(result["evaluated_weight_fraction"],1)
        self.assertEqual(result["decision"],"CHECKS_PASSED_REVIEW_REQUIRED")

    def test_normalized_split_leakage_blocks_even_high_score(self):
        records = [{"text":f"unique row {i}","split":"test"} for i in range(100)]
        records += [{"text":" Shared   Text ","split":"train"},{"text":"shared text","split":"test"}]
        d = self.dataset(records,Config(split_column="split"))
        result = evaluate(d,policy(d))
        self.assertGreater(result["quality_index"],99)
        self.assertIn("split_disjointness",result["blockers"])
        self.assertEqual(result["decision"],"BLOCKED")

    def test_conflicting_labels(self):
        d = self.dataset([{"text":"same","label":"a"},{"text":"SAME","label":"b"}],Config(label_column="label"))
        r = d.metric("label_consistency",{})
        self.assertEqual(r["score"],0)
        self.assertEqual(r["issue_row_ids"],[1,2])

    def test_missing_splits_withholds_index(self):
        d = self.dataset([{"text":"train only","split":"train"}],Config(split_column="split"))
        result = evaluate(d,policy(d))
        self.assertIsNone(result["quality_index"])
        self.assertLess(result["evaluated_weight_fraction"],1)
        self.assertIn("split_disjointness",result["blockers"])

    def test_empty_dataset_does_not_pass(self):
        d = self.dataset([])
        r = evaluate(d,policy(d))
        self.assertIsNone(r["quality_index"])
        self.assertIn("empty_dataset",r["blockers"])

    def test_non_string_and_empty_text(self):
        d = self.dataset([{"text":123},{"text":"   "},{"text":"valid"}])
        self.assertAlmostEqual(d.metric("text_type",{})["score"],2/3)
        self.assertAlmostEqual(d.metric("text_completeness",{})["score"],1/3)

    def test_unknown_metric_and_omitted_gate_rejected(self):
        d = self.dataset([{"text":"valid"}])
        p = policy(d)
        p["metrics"][0]["name"] = "execute_python"
        with self.assertRaises(ValueError): validate_policy(p,d)
        p = policy(d)
        p["metrics"][0]["gate"] = False
        with self.assertRaises(ValueError): validate_policy(p,d)

    def test_invalid_weights_nan_and_threshold_rejected(self):
        d = self.dataset([{"text":"valid"}])
        for field,value in [("weight",float("nan")),("weight",0.1),("pass_min",0)]:
            p = policy(d)
            p["metrics"][0][field] = value
            with self.assertRaises(ValueError): validate_policy(p,d)

    def test_unsupported_check_not_allowed(self):
        d = self.dataset([{"text":"valid"}])
        p = policy(d)
        p["metrics"].append({"name":"label_validity","weight":0.1,"pass_min":1,"gate":True,"rationale":"unsupported"})
        with self.assertRaises(ValueError): validate_policy(p,d)

    def test_legitimate_phone_is_not_automatically_penalized(self):
        d = self.dataset([{"text":"Contact +44 7123 456789"}])
        self.assertNotIn("privacy_pattern_clearance",d.applicable())
        self.assertEqual(evaluate(d,policy(d))["quality_index"],100)

    def test_privacy_gate_when_required(self):
        d = self.dataset([{"text":"Email person@example.com"}],Config(require_redacted=True))
        self.assertIn("privacy_pattern_clearance",evaluate(d,policy(d))["blockers"])

    def test_group_leakage_even_when_texts_different(self):
        c = Config(split_column="split",group_column="customer")
        d = self.dataset([{"text":"first query","split":"train","customer":"x"},
                          {"text":"different query","split":"test","customer":"x"}],c)
        r=evaluate(d,policy(d))
        self.assertIn("group_disjointness",r["blockers"])
        self.assertNotIn("split_disjointness",r["blockers"])

    def test_absent_expected_class_support_zero(self):
        d = self.dataset([{"text":"x","label":"a"}],Config(label_column="label",allowed_labels=("a","b")))
        self.assertEqual(d.metric("class_support",{"min_class_examples":2})["score"],0)

    def test_imbalance_itself_not_penalized(self):
        d = self.dataset([{"text":str(i),"label":"majority"} for i in range(100)]+
                         [{"text":"minority 1","label":"minority"},{"text":"minority 2","label":"minority"}],Config(label_column="label"))
        self.assertEqual(d.metric("class_support",{"min_class_examples":2})["score"],1)

    def test_malformed_rows_not_silently_skipped(self):
        p=self.root/"bad.csv"; p.write_text("text,label\nhi,a,extra\n")
        with self.assertRaises(ValueError): LocalDataset(p,Config())
        p=self.root/"bad.jsonl"; p.write_text('{"text":"hello"}\n\n')
        with self.assertRaises(ValueError): LocalDataset(p,Config())

    def test_local_planner_repair_cache_and_no_raw_text(self):
        secret="PRIVATE_RAW_DATA_12345"
        d=self.dataset([{"text":secret}])
        p=policy(d)
        p["version"]=2
        p["context"]={"purpose":"log_monitoring","privacy":"retain","description":"Monitor text events","assumptions":[]}
        for m in p["metrics"]:
            m["utility"]={"type":"linear","bad":0,"good":1}
        calls=[]
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args): pass
            def do_POST(self):
                request=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append(request)
                output={} if len(calls)==1 else p
                body=json.dumps({"choices":[{"message":{"content":json.dumps(output)}}]}).encode()
                self.send_response(200); self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
        server=HTTPServer(("127.0.0.1",0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            endpoint=f"http://127.0.0.1:{server.server_port}/v1"
            actual,meta=plan(d,"Classify support messages","test-model",endpoint,self.root/"cache")
            self.assertEqual(actual,p); self.assertEqual(meta["request_count"],2)
            self.assertNotIn(secret,json.dumps(calls))
            actual,meta=plan(d,"Classify support messages","test-model",endpoint,self.root/"cache")
            self.assertEqual(len(calls),2); self.assertEqual(meta["source"],"ai_cache")
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_remote_endpoint_requires_opt_in(self):
        d=self.dataset([{"text":"text"}])
        with self.assertRaises(ValueError):
            plan(d,"business","model","https://remote.example/v1",self.root/"cache")


if __name__ == "__main__": unittest.main()
