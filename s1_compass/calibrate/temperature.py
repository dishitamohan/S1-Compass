"""Deterministic inverse-temperature calibration with a hash-bound sidecar."""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
from s1_compass.data.schema import strict_dumps


def _ece(logits, labels, option_ids, beta: float, bins: int = 15) -> float:
    confidence=[]; hits=[]
    for row,y in zip(logits,labels):
        z=[x*beta for x in row]; m=max(z); exp=[math.exp(v-m) for v in z]; p=[v/sum(exp) for v in exp]
        winner=max(range(len(p)),key=lambda j:(p[j],tuple(-ord(c) for c in option_ids[len(confidence)][j])))
        confidence.append(p[winner]); hits.append(float(winner==y))
    ece=0.0
    for b in range(bins):
        ix=[i for i,c in enumerate(confidence) if b/bins<=c<(b+1)/bins or (b==bins-1 and c==1.0)]
        if ix: ece+=len(ix)/len(labels)*abs(sum(hits[i] for i in ix)/len(ix)-sum(confidence[i] for i in ix)/len(ix))
    return ece


def fit(logits, labels, option_ids, *, model_sha: str, fit_on: str, output_path: str | Path | None = None):
    if not logits or len(logits)!=len(labels) or len(labels)!=len(option_ids):
        raise ValueError("calibration requires equally sized nonempty data")
    if len(model_sha)!=64 or len(fit_on)!=len("calibration:")+64:
        raise ValueError("model_sha and calibration fit_on must be SHA-256 identities")
    if any(len(row)!=len(ids) or not 0<=y<len(row) for row,ids,y in zip(logits,option_ids,labels)):
        raise ValueError("ragged calibration row or invalid label")
    def nll(beta):
        total=0.0
        for row,y in zip(logits,labels):
            z=[v*beta for v in row]; m=max(z)
            total += m+math.log(sum(math.exp(v-m) for v in z))-z[y]
        return total/len(labels)
    before=nll(1.0)
    lo,hi=.05,20.0; ratio=(math.sqrt(5)-1)/2
    c=hi-ratio*(hi-lo); d=lo+ratio*(hi-lo)
    for _ in range(128):
        if nll(c)<=nll(d): hi,d=d,c; c=hi-ratio*(hi-lo)
        else: lo,c=c,d; d=lo+ratio*(hi-lo)
    candidates=[.05,1.0,20.0,(lo+hi)/2]
    best=min(nll(x) for x in candidates)
    beta=min((x for x in candidates if nll(x)<=best+1e-12),key=lambda x:(abs(math.log(x)),x))
    if nll(beta)>before+1e-12: beta=1.0
    temperature=1.0/beta
    record={"T":temperature,"nll_before":before,"nll_after":nll(beta),
            "ece_before":_ece(logits,labels,option_ids,1.0),"ece_after":_ece(logits,labels,option_ids,beta),
            "fit_on":fit_on,"model_sha":model_sha}
    body=(strict_dumps(record)+"\n").encode("utf-8")
    version=hashlib.sha256(body).hexdigest()
    if output_path is not None:
        path=Path(output_path); path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_bytes(body); tmp.replace(path)
    return {**record,"version":version}
