"""Confidence based escalation policy."""
from __future__ import annotations


def apply(output: dict, *, threshold: float = .75):
    if not 0 <= threshold <= 1: raise ValueError("threshold outside [0,1]")
    escalated=[]; bands={}
    for qid, answer in output["answers"].items():
        confidence=(max(answer["probabilities"].values()) if answer["type"] != "boolean" else max(answer["probability"],1-answer["probability"]))
        escalated.append(confidence < threshold)
        bands[qid]="auto" if confidence>=.90 else "audit" if confidence>=threshold else "escalate"
    return {"automatic":not any(escalated),"bands":bands,"escalate":{qid:flag for qid,flag in zip(output["answers"],escalated)},
            "audit_questions":[qid for qid,band in bands.items() if band=="audit"],
            "low_confidence_questions":[qid for qid,band in bands.items() if band=="escalate"]}
