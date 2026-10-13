"""Download publisher sources, retain hashes and attribution, export canonical CSV.

SMS receives deterministic hash-grouped demonstration splits (not publisher splits).
Banking77 preserves the publisher's train/test split for auditing, including overlaps.
"""
import argparse
import csv
import hashlib
import json
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

UCI = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
BANK = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"


def fetch(url,path):
    if not path.exists():
        with urllib.request.urlopen(url,timeout=60) as response:
            data = response.read(20_000_001)
        if len(data)>20_000_000:
            raise ValueError("Dataset download exceeds 20 MB limit.")
        path.write_bytes(data)
    return {"url":url,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size}


def export(root,which):
    out = root/which
    raw = out/"raw"
    raw.mkdir(parents=True,exist_ok=True)
    sources, records = [], []
    if which == "sms_spam":
        source = raw/"source.zip"
        sources.append(fetch(UCI,source))
        with zipfile.ZipFile(source) as archive:
            source_text = archive.read("SMSSpamCollection").decode("utf-8")
            (raw/"publisher_readme.txt").write_bytes(archive.read("readme"))
        import unicodedata
        for line in source_text.splitlines():
            label,text = line.split("\t",1)
            normalized = " ".join(unicodedata.normalize("NFKC",text).casefold().split())
            bucket = int(hashlib.sha256(normalized.encode()).hexdigest()[:8],16)%100
            split = "train" if bucket < 80 else "validation" if bucket < 90 else "test"
            records.append({"text":text,"label":label,"split":split})
        labels = ["ham","spam"]
        provenance = {"publisher":"UCI Machine Learning Repository","doi":"10.24432/C5CC84",
                      "authors":"Tiago Almeida and Jos Hidalgo","license":"CC BY 4.0",
                      "citation":"Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository.",
                      "publisher_url":"https://archive.ics.uci.edu/dataset/228/sms+spam+collection",
                      "transformation":"Tab-delimited messages converted to CSV; deterministic 80/10/10 approximate splits grouped by normalized text hash. Not stratified or publisher-provided; preserve duplicate records for auditing.",
                      "catalog_count_note":"UCI lists 5,574 instances; actual parsed file row count is recorded below. Do not fabricate rows to match catalog metadata."}
    else:
        for split in ("train","test"):
            path = raw/(split+".csv")
            sources.append(fetch(BANK+path.name,path))
            with path.open(encoding="utf-8",newline="") as f:
                for row in csv.DictReader(f):
                    records.append({"text":row["text"],"label":row["category"],"split":split})
        path = raw/"categories.json"
        sources.append(fetch(BANK+path.name,path))
        labels = json.loads(path.read_text())
        provenance = {"publisher":"PolyAI","license":"CC BY 4.0",
                      "citation":"Casanueva et al. (2020). Efficient Intent Detection with Dual Sentence Encoders. ACL NLP for ConvAI.",
                      "publisher_url":"https://github.com/PolyAI-LDN/task-specific-datasets",
                      "transformation":"category column renamed label; train/test files concatenated with explicit split column; publisher splits and all rows preserved.",
                      "known_limit":"No official validation split in these files; do not tune downstream models on test data."}
    with (out/"dataset.csv").open("w",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=["text","label","split"])
        writer.writeheader(); writer.writerows(records)
    (out/"labels.json").write_text(json.dumps(labels,indent=2),encoding="utf-8")
    provenance.update(sources=sources,parsed_rows=len(records),split_counts=dict(Counter(r["split"] for r in records)),
                      label_count=len(labels),processed_sha256=hashlib.sha256((out/"dataset.csv").read_bytes()).hexdigest())
    (out/"provenance.json").write_text(json.dumps(provenance,indent=2),encoding="utf-8")
    print(which, len(records), "rows")


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--dataset",choices=["sms_spam","banking77","all"],default="all")
    parser.add_argument("--out",type=Path,default=Path("data"))
    args=parser.parse_args()
    for which in (["sms_spam","banking77"] if args.dataset=="all" else [args.dataset]):
        export(args.out,which)
