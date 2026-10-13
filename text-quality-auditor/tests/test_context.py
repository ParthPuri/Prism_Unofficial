import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from textaudit.core import Config, LocalDataset, evaluate, utility, validate_policy
from scripts.context_demo import example_policies


def monitoring_policy():
    return dict(example_policies())["event_monitoring"]


class ContextTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/"data.jsonl"

    def dataset(self,records,config=Config()):
        self.path.write_text("".join(json.dumps(x)+"\n" for x in records))
        d=LocalDataset(self.path,config); self.addCleanup(d.close); return d

    def test_utility_anchors_and_missing_evidence(self):
        self.assertAlmostEqual(utility(.98,{"bad":.9,"good":1}),.8)
        self.assertAlmostEqual(utility(.98,{"bad":.5,"good":1}),.96)
        self.assertEqual(utility(.8,{"bad":.9,"good":1}),0)
        self.assertEqual(utility(1,{"bad":.9,"good":.99}),1)
        self.assertIsNone(utility(None,{"bad":.9,"good":1}))

    def test_same_data_different_context_scores(self):
        d=self.dataset([{"text":"same message"} for _ in range(10)])
        ps=dict(example_policies())
        monitor=evaluate(d,ps["event_monitoring"])
        retrieval=evaluate(d,ps["retrieval_corpus"])
        self.assertEqual(monitor["quality_index"],100)
        self.assertEqual(retrieval["quality_index"],50)
        self.assertEqual(monitor["input_sha256"],retrieval["input_sha256"])

    def test_utility_affects_score_not_raw_evidence(self):
        d=self.dataset([{"text":"good"}]*99+[{"text":"bad\ufffd"}])
        p=monitoring_policy(); r=evaluate(d,p)
        encoding=next(x for x in r["metrics"] if x["metric"]=="encoding_cleanliness")
        self.assertEqual(encoding["score"],.99)
        self.assertAlmostEqual(encoding["contextual_score"],.9)
        self.assertAlmostEqual(r["quality_index"],98)

    def test_irrelevant_label_checks_rejected(self):
        d=self.dataset([{"text":"event","label":"x"}],Config(label_column="label"))
        p=monitoring_policy(); p["metrics"][0]["name"]="label_completeness"
        with self.assertRaises(ValueError): validate_policy(p,d)

    def test_trusted_purpose_cannot_be_changed_by_ai(self):
        d=self.dataset([{"text":"event"}],Config(expected_purpose="retrieval_corpus"))
        with self.assertRaises(ValueError): validate_policy(monitoring_policy(),d)

    def test_privacy_cannot_be_waived(self):
        d=self.dataset([{"text":"event"}],Config(require_redacted=True))
        with self.assertRaises(ValueError): validate_policy(monitoring_policy(),d)

    def test_utility_bounds_and_direction_enforced(self):
        d=self.dataset([{"text":"event"}])
        for bad,good in [(1,.9),(-1,1),(.5,.5),(float("nan"),1)]:
            p=monitoring_policy(); p["metrics"][0]["utility"].update(bad=bad,good=good)
            with self.assertRaises(ValueError): validate_policy(p,d)

    @unittest.skipUnless(importlib.util.find_spec("great_expectations"),"GX optional dependency not installed")
    def test_gx_executes_real_checks_with_same_evidence(self):
        from textaudit.gx_backend import GXRunner
        d=self.dataset([{"text":"valid"},{"text":"\ufffd"},{"text":123},{"text":" "}])
        p=monitoring_policy()
        baseline=evaluate(d,p); runner=GXRunner(d,p); gx=evaluate(d,p,runner)
        self.assertEqual(gx["quality_index"],baseline["quality_index"])
        self.assertEqual(gx["blockers"],baseline["blockers"])
        self.assertEqual(len(runner.specifications),4)
        for expected,actual in zip(baseline["metrics"],gx["metrics"]):
            self.assertEqual(actual["score"],expected["score"])
            self.assertEqual(actual["executor"],"great_expectations")
            self.assertEqual(actual["gx_success"],actual["status"]=="PASS")
        self.assertFalse(runner.context.variables.analytics_enabled)

    @unittest.skipUnless(importlib.util.find_spec("great_expectations"),"GX optional dependency not installed")
    def test_gx_no_eligible_rows_does_not_pass(self):
        from textaudit.gx_backend import GXRunner
        d=self.dataset([{"text":" "}]); p=monitoring_policy()
        r=evaluate(d,p,GXRunner(d,p))
        self.assertIsNone(r["quality_index"])
        self.assertIn("text_completeness",r["blockers"])

    @unittest.skipUnless(importlib.util.find_spec("great_expectations"),"GX optional dependency not installed")
    def test_gx_label_and_split_validation_match_local_counts(self):
        from textaudit.gx_backend import GXRunner
        c=Config(label_column="label",split_column="split",allowed_labels=("ham","spam"))
        d=self.dataset([{"text":"train one","label":"ham","split":"train"},
                        {"text":"test one","label":"wrong","split":"test"},
                        {"text":"missing label","label":None,"split":"typo"}],c)
        p=dict(example_policies())["spam_training"]
        local=evaluate(d,p); actual=evaluate(d,p,GXRunner(d,p))
        self.assertEqual(local["quality_index"],actual["quality_index"])
        self.assertEqual(local["blockers"],actual["blockers"])
        for left,right in zip(local["metrics"],actual["metrics"]):
            self.assertEqual(left["score"],right["score"])


if __name__=="__main__": unittest.main()
