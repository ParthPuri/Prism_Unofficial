"""Great Expectations validation of locally derived text features.

GX computes violation counts and raw scores for supported checks. Cross-row checks
remain SQLite aggregations. No raw message content enters GX or its stored results.
"""
from .core import CATALOG

SUPPORTED = {
    "text_type":("typ",None),
    "text_completeness":("nonempty",None),
    "encoding_cleanliness":("clean","nonempty=1"),
    "label_completeness":("has_label",None),
    "label_validity":("label_ok","label IS NOT NULL"),
    "split_completeness":("split_ok",None),
    "group_completeness":("has_group",None),
    "privacy_pattern_clearance":("privacy_clear","nonempty=1"),
    "length_validity":("n","nonempty=1"),
}


class GXRunner:
    def __init__(self,dataset,policy):
        try:
            import great_expectations as gx
            import pandas as pd
        except ImportError as exc:
            raise ValueError("Install GX support: python3 -m pip install -e '.[gx]'") from exc
        self.gx,self.pd,self.dataset,self.policy=gx,pd,dataset,policy
        self.specifications=[]
        from great_expectations.data_context.types.base import DataContextConfig, InMemoryStoreBackendDefaults
        self.context=gx.get_context(mode="ephemeral",project_config=DataContextConfig(
            store_backend_defaults=InMemoryStoreBackendDefaults(),analytics_enabled=False,
            progress_bars={"globally":False}))
        self.source=self.context.data_sources.add_pandas(name="local_text_features")
        self.constraints={m["name"]:m for m in policy["metrics"]}
        self.frame=pd.read_sql_query("SELECT i,n,typ,nonempty,clean,label_ok,split_ok,(label IS NOT NULL) AS has_label,(g IS NOT NULL) AS has_group,(1-private) AS privacy_clear FROM r",dataset.db)

    def __call__(self,name,parameters):
        if name not in SUPPORTED:
            result=self.dataset.metric(name,parameters)
            result["executor"]="local_sqlite"
            return result
        column,where=SUPPORTED[name]
        frame=self.frame
        if where=="nonempty=1":
            frame=frame.loc[frame.nonempty==1]
        elif where=="label IS NOT NULL":
            frame=frame.loc[frame.has_label==1]
        total=len(frame)
        if not total:
            return {"metric":name,"score":None,"evaluated_rows":0,"reason":"No eligible rows.","executor":"great_expectations","issue_row_ids":[]}
        constraint=self.constraints[name]
        args={"column":column,"mostly":constraint["pass_min"],
              "severity":"critical" if constraint["gate"] else "warning",
              "meta":{"metric":name,"business_reason":constraint["rationale"]}}
        if name=="length_validity":
            expectation=self.gx.expectations.ExpectColumnValuesToBeBetween(
                min_value=parameters["min_chars"],max_value=parameters["max_chars"],**args)
        else:
            expectation=self.gx.expectations.ExpectColumnValuesToBeInSet(value_set=[1],**args)
        suite=self.gx.ExpectationSuite(name="context_"+name)
        suite.add_expectation(expectation)
        suite=self.context.suites.add(suite)
        asset=self.source.add_dataframe_asset(name=name)
        batch=asset.add_batch_definition_whole_dataframe(name="all_eligible_rows")
        validation=self.context.validation_definitions.add(self.gx.ValidationDefinition(
            name="validate_"+name,data=batch,suite=suite))
        results=validation.run(batch_parameters={"dataframe":frame},result_format={
            "result_format":"SUMMARY","partial_unexpected_count":0,
            "include_unexpected_rows":False,"exclude_unexpected_values":True})
        result=results.results[0]
        if result.exception_info.get("raised_exception"):
            raise ValueError("GX check failed to execute: "+name)
        bad=result.result["unexpected_count"]
        score=(total-bad)/total
        spec=expectation.configuration.to_json_dict()
        self.specifications.append(spec)
        # Only safe counts and configuration are exported; GX raw result objects stay local.
        if name=="length_validity":
            bad_rows=frame.loc[(frame.n<parameters["min_chars"])|(frame.n>parameters["max_chars"]),"i"]
        else:
            bad_rows=frame.loc[frame[column]!=1,"i"]
        return {"metric":name,"score":score,"evaluated_rows":total,"good_rows":total-bad,
                "bad_rows":bad,"issue_row_ids":[int(i) for i in bad_rows.head(10)],
                "details":CATALOG[name],"executor":"great_expectations",
                "gx_success":bool(result.success),"gx_version":self.gx.__version__}
