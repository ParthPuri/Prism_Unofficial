"""Portable structured output contract; cross-field rules remain local validation."""
from .core import CATALOG


def obj(properties):
    return {"type":"object","properties":properties,"required":list(properties),"additionalProperties":False}


POLICY_SCHEMA = obj({
    "version":{"type":"integer","enum":[2]},
    "context":obj({
        "purpose":{"type":"string","enum":["supervised_training","retrieval_corpus","log_monitoring"]},
        "privacy":{"type":"string","enum":["retain","redacted"]},
        "description":{"type":"string"},
        "assumptions":{"type":"array","items":{"type":"string"}},
    }),
    "metrics":{"type":"array","items":obj({
        "name":{"type":"string","enum":list(CATALOG)},
        "weight":{"type":"number"},"pass_min":{"type":"number"},
        "gate":{"type":"boolean"},"rationale":{"type":"string"},
        "utility":obj({"type":{"type":"string","enum":["linear"]},
                       "bad":{"type":"number"},"good":{"type":"number"}}),
    })},
    "parameters":obj({"min_chars":{"type":"integer"},"max_chars":{"type":"integer"},
                      "min_class_examples":{"type":"integer"}}),
    "acceptance_score":{"type":"number"},"rationale":{"type":"string"},
})


def response_format(structured):
    if structured:
        return {"type":"json_schema","json_schema":{"name":"TextQualityPolicy",
                "schema":POLICY_SCHEMA,"strict":True}}
    return {"type":"json_object"}
