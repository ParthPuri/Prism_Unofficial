"""Explicit illustrative policies: same public SMS data, three business contexts.

No AI outputs are fabricated. This isolates the effect of checks, utility anchors
and weights. Use the --use-case CLI to obtain a real model-generated policy.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from textaudit.core import Config, LocalDataset, evaluate
from textaudit.cli import markdown


def example_policies():
    base = {"version":2,"parameters":{"min_chars":1,"max_chars":2000,"min_class_examples":20},
            "acceptance_score":90,"rationale":"Explicit demonstration of contextual utility. Anchors and weights are illustrative, not empirically calibrated or AI-generated."}
    specs = {
        "spam_training":("supervised_training","retain",[
            ("text_type",.03,1,True,0,1),("text_completeness",.12,.99,True,.95,1),
            ("label_completeness",.12,.99,True,.95,1),("label_consistency",.15,1,True,.95,1),
            ("label_validity",.08,1,True,.95,1),("split_completeness",.05,1,True,.95,1),
            ("split_disjointness",.25,1,True,.99,1),("encoding_cleanliness",.05,.99,False,.9,1),
            ("class_support",.15,1,False,0,1)],
            "Supervised SMS spam classifier. Preserve legitimate repeated campaigns and phone numbers; require independent train/evaluation splits."),
        "retrieval_corpus":("retrieval_corpus","retain",[
            ("text_type",.05,1,True,0,1),("text_completeness",.15,.99,True,.95,1),
            ("normalized_uniqueness",.5,.99,False,.9,1),("encoding_cleanliness",.1,.99,False,.9,1),
            ("length_validity",.2,.99,False,.95,1)],
            "Structural suitability as a message retrieval corpus. Redundant normalized entries waste retrieval/storage capacity; relevance and factuality are not assessed."),
        "event_monitoring":("log_monitoring","retain",[
            ("text_type",.2,1,True,0,1),("text_completeness",.4,.99,True,.95,1),
            ("encoding_cleanliness",.2,.99,False,.9,1),("length_validity",.2,.99,False,.95,1)],
            "Monitor message-event records. Repeated messages represent real recurring events, so uniqueness is not a quality requirement; labels and training partitions are irrelevant.")}
    for name,(purpose,privacy,metrics,description) in specs.items():
        p=dict(base)
        p["context"]={"purpose":purpose,"privacy":privacy,"description":description,
                      "assumptions":["Public SMS content is permitted for this example.","Only implemented structural quality dimensions are assessed."]}
        p["metrics"]=[{"name":n,"weight":w,"pass_min":t,"gate":g,
                       "utility":{"type":"linear","bad":bad,"good":good},
                       "rationale":description} for n,w,t,g,bad,good in metrics]
        yield name,p


def main():
    from textaudit.gx_backend import GXRunner
    root=Path(__file__).resolve().parents[1]
    dataset_path=root/"data/sms_spam/dataset.csv"
    config=Config(label_column="label",split_column="split",allowed_labels=("ham","spam"))
    comparison=[]
    with LocalDataset(dataset_path,config) as d:
        for name,p in example_policies():
            runner=GXRunner(d,p)
            r=evaluate(d,p,runner)
            r.update(planner={"source":"explicit_context_demo"},backend="gx",gx_expectations=runner.specifications)
            out=root/"reports/context"/name; out.mkdir(parents=True,exist_ok=True)
            (out/"report.json").write_text(json.dumps(r,indent=2))
            (out/"report.md").write_text(markdown(r))
            (out/"policy.json").write_text(json.dumps(p,indent=2))
            summary={"context":name,"quality_index":r["quality_index"],"decision":r["decision"],"input_sha256":r["input_sha256"]}
            comparison.append(summary); print(json.dumps(summary))
    (root/"reports/context/comparison.json").write_text(json.dumps(comparison,indent=2))


if __name__=="__main__": main()
