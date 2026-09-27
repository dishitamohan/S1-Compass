"""Metrics for ragged multiclass decisions."""
from __future__ import annotations
import math


def evaluate(labels, probabilities, option_ids, bins: int = 15, question_ids=None):
    if not (len(labels) == len(probabilities) == len(option_ids)) or not labels:
        raise ValueError("metrics require equally sized nonempty inputs")
    question_ids=question_ids or [""]*len(labels)
    if len(question_ids)!=len(labels): raise ValueError("question_ids length mismatch")
    correct=0; brier=0.0; nll=0.0; abstain=0; confidences=[]; hits=[]; classes={}; primitive={}
    for y, p, ids, qid in zip(labels, probabilities, option_ids, question_ids):
        if len(p) != len(ids) or not ids: raise ValueError("ragged row mismatch")
        pred=max(range(len(p)), key=lambda j:(p[j], tuple(-ord(c) for c in ids[j])))
        hit=int(pred==y); correct+=hit
        brier += sum((p[j]-(1.0 if j==y else 0.0))**2 for j in range(len(p)))
        nll -= math.log(max(p[y], 1e-300)); conf=max(p); abstain += conf < .75
        confidences.append(conf); hits.append(hit)
        for j, oid in enumerate(ids):
            key=(qid,oid)
            tp,fp,fn=classes.get(key,[0,0,0])
            classes[key]=[tp+(pred==j and y==j),fp+(pred==j and y!=j),fn+(pred!=j and y==j)]
        primitive.setdefault(qid,[]).append((y,p,ids))
    ece=0.0
    for b in range(bins):
        ix=[i for i,c in enumerate(confidences) if (b/bins <= c < (b+1)/bins) or (b==bins-1 and c==1)]
        if ix: ece += len(ix)/len(labels)*abs(sum(hits[i] for i in ix)/len(ix)-sum(confidences[i] for i in ix)/len(ix))
    f1=[]
    for tp,fp,fn in classes.values(): f1.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
    per_primitive={}
    for qid,rows in primitive.items():
        per_primitive[qid]={"count":len(rows),"accuracy":sum(max(range(len(p)),key=lambda j:(p[j],tuple(-ord(c) for c in ids[j])))==y for y,p,ids in rows)/len(rows)}
    uniform_brier=sum(1.0-1.0/len(ids) for ids in option_ids)/len(labels)
    uniform_nll=sum(math.log(len(ids)) for ids in option_ids)/len(labels)
    return {"count":len(labels),"accuracy":correct/len(labels),"macro_f1":sum(f1)/len(f1),"brier":brier/len(labels),"uniform_brier":uniform_brier,"log_loss":nll/len(labels),"uniform_log_loss":uniform_nll,"ece":ece,"abstention_rate":abstain/len(labels),"per_primitive":per_primitive}
