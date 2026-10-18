import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .core import Config, LocalDataset, evaluate
from .planner import plan
from .providers import resolve_provider


def markdown(report: dict) -> str:
    def cell(value):
        return str(value).replace("|","\\|").replace("\n"," ").replace("\r"," ")
    score = report["quality_index"]
    lines = ["# Text Dataset Quality Report", "",f"Decision: **{report['decision']}**", "",
             f"Quality index: **{score:.2f}/100**" if score is not None else "Quality index: **withheld; not all selected checks could run**",
             f"Evaluated weight: {100*report['evaluated_weight_fraction']:.1f}%", "",
             "The index describes this policy's checks. Final fitness requires human review and downstream validation.", "",
             "| Metric | Raw / 100 | Context utility / 100 | Weight | Raw pass minimum / 100 | Gate | Status |", "|---|---:|---:|---:|---:|---|---|"]
    for r in report["metrics"]:
        s = f"{100*r['score']:.2f}" if r["score"] is not None else "Not evaluated"
        u = f"{100*r['contextual_score']:.2f}" if r["contextual_score"] is not None else "Not evaluated"
        lines.append(f"| {r['metric']} | {s} | {u} | {r['weight']:.3f} | {100*r['pass_min']:.2f} | {r['gate']} | {r['status']} |")
    if report.get("context"):
        c = report["context"]
        lines += ["", "## Interpreted business context", "",f"Purpose: **{c['purpose']}**; privacy: **{c['privacy']}**.","",cell(c["description"]),"","Assumptions:",""]
        lines += ["- "+cell(a) for a in c["assumptions"]] or ["- None declared."]
    lines += ["", "## Blocking checks", "", ", ".join(report["blockers"]) or "None", "",
              "## Policy rationale", "", cell(report["policy"]["rationale"]), "",
              "## Findings and local evidence", ""]
    for r in report["metrics"]:
        lines.append(f"- **{r['metric']}**: {cell(r.get('details') or r.get('reason') or 'Full scan of eligible rows.')} Row IDs: {r.get('issue_row_ids',[])}. {cell(r['rationale'])}")
    lines += ["", "## Limits", ""]+["- "+x for x in report["limits"]]
    lines += ["", "## Reproducibility", "",f"Input SHA-256: `{report['input_sha256']}`",
              f"Policy SHA-256: `{report['policy_sha256']}`",f"Engine: `{report['engine_version']}`",
              f"Planner source: `{report['planner']['source']}`", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="AI-planned, locally computed text dataset quality auditing")
    parser.add_argument("input", type=Path, help="Input dataset path (.csv, .tsv, .json, .jsonl, .parquet, .arrow, .txt)")
    parser.add_argument("--text-column",default="text")
    parser.add_argument("--label-column")
    parser.add_argument("--split-column")
    parser.add_argument("--group-column",help="Enable group isolation when groups must not cross splits")
    parser.add_argument("--allowed-labels",type=Path,help="JSON array of allowed labels")
    parser.add_argument("--require-redacted",action="store_true")
    parser.add_argument("--purpose",choices=["supervised_training","retrieval_corpus","log_monitoring"],
                        help="Optional trusted purpose; prevents the AI from changing assessment scope")
    parser.add_argument("--backend",choices=["local","gx"],default="local")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--policy",type=Path,help="Explicit saved policy; not described as AI-generated")
    source.add_argument("--use-case",help="Business context for an AI policy; do not enter secrets")
    parser.add_argument("--provider",choices=["huggingface","groq","local","compatible"],default="huggingface")
    parser.add_argument("--model",help="Hosted model ID or exact locally installed model name")
    parser.add_argument("--base-url",help="Override local or compatible endpoint")
    parser.add_argument("--allow-remote",action="store_true")
    parser.add_argument("--timeout",type=float,default=120)
    parser.add_argument("--max-tokens",type=int,default=4000)
    parser.add_argument("--cache-dir",type=Path,default=Path(".cache/policies"))
    parser.add_argument("--out",type=Path,default=Path("reports/latest"))
    args = parser.parse_args(argv)
    start = time.perf_counter()
    try:
        inference=None; token=None
        if args.use_case is not None:
            inference=resolve_provider(args.provider,args.model,args.base_url,args.allow_remote)
            token=inference.credentials()  # Fail before scanning the dataset if authentication is missing.
        labels = json.loads(args.allowed_labels.read_text()) if args.allowed_labels else []
        if not isinstance(labels,list):
            raise ValueError("Allowed-label file must contain a JSON array.")
        config = Config(args.text_column,args.label_column,args.split_column,args.group_column,
                        tuple(labels),args.require_redacted,args.purpose)
        with LocalDataset(args.input,config) as dataset:
            if args.policy:
                policy = json.loads(args.policy.read_text())
                provenance = {"source":"explicit_policy_file"}
            else:
                policy,provenance = plan(dataset,args.use_case,inference.model,inference.base_url,
                    args.cache_dir,inference.allow_remote,args.timeout,api_key=token,
                    provider=inference.name,structured_schema=inference.structured_schema,max_tokens=args.max_tokens)
            runner = None
            if args.backend == "gx":
                from .gx_backend import GXRunner
                # Validate the generated contract before creating any GX objects.
                from .core import validate_policy
                validate_policy(policy,dataset)
                runner=GXRunner(dataset,policy)
            report = evaluate(dataset,policy,runner)
            report["backend"]=args.backend
            if runner:
                report["gx_expectations"]=runner.specifications
            report.update(planner=provenance,created_utc=datetime.now(timezone.utc).isoformat(),
                          elapsed_seconds=round(time.perf_counter()-start,3))
        args.out.mkdir(parents=True,exist_ok=True)
        for name,content in [("report.json",json.dumps(report,ensure_ascii=True,indent=2)),
                             ("report.md",markdown(report)),("policy.json",json.dumps(policy,indent=2))]:
            target = args.out/name
            temp = target.with_suffix(target.suffix+".tmp")
            temp.write_text(content,encoding="utf-8")
            temp.replace(target)
        print(json.dumps({"decision":report["decision"],"quality_index":report["quality_index"],
                          "coverage":report["evaluated_weight_fraction"],"output":str(args.out)}))
        return 2 if report["decision"] in {"BLOCKED","INCOMPLETE","REVIEW_REQUIRED"} else 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        # No provider response body or raw dataset text is logged.
        print("Audit failed: "+str(exc),file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
