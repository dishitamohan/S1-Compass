"""Evaluation gates and a small HTTP latency benchmark."""
from __future__ import annotations
import concurrent.futures, time, urllib.request, json
from urllib.error import HTTPError
from s1_compass.eval.metrics import evaluate


def option_order_delta(original, permuted, original_ids, permuted_ids):
    maximum=0.0; disagreements=0
    for a,b,ia,ib in zip(original,permuted,original_ids,permuted_ids):
        ap=dict(zip(ia,a)); bp=dict(zip(ib,b))
        maximum=max(maximum,max(abs(ap[k]-bp[k]) for k in ap))
        if max(ap,key=lambda k:(ap[k],tuple(-ord(c) for c in k))) != max(bp,key=lambda k:(bp[k],tuple(-ord(c) for c in k))): disagreements+=1
    return {"option_order_delta":maximum,"argmax_disagreements":disagreements}


def benchmark_url(url: str, request: dict, repeats: int=30):
    payload=json.dumps(request).encode(); samples=[]
    for _ in range(repeats):
        req=urllib.request.Request(url,data=payload,headers={"Content-Type":"application/json"},method="POST")
        start=time.perf_counter()
        try:
            with urllib.request.urlopen(req,timeout=2) as response: response.read()
        except HTTPError as response:
            response.read()
            if response.code not in (409,422,504): raise
        samples.append((time.perf_counter()-start)*1000)
    ordered=sorted(samples)
    return {"requests":repeats,"p95_latency_ms":ordered[max(0,int(.95*len(ordered)+.999)-1)],"mean_latency_ms":sum(samples)/len(samples)}


def run(labels, probabilities, option_ids, *, thresholds=None, question_ids=None):
    if not labels:
        return {"metrics":None,"thresholds":thresholds or {},"checks":{},"promotable":False,"error":"No answerable evaluation rows."}
    metrics=evaluate(labels,probabilities,option_ids,question_ids=question_ids)
    limits={"accuracy":.85,"macro_f1":.80,"brier":.25,"ece":.05,**(thresholds or {})}
    checks={key:(metrics[key]>=limit if key in ("accuracy","macro_f1") else metrics[key]<=limit) for key,limit in limits.items()}
    checks["brier_below_uniform"]=metrics["brier"]<metrics["uniform_brier"]
    checks["nll_below_uniform"]=metrics["log_loss"]<metrics["uniform_log_loss"]
    return {"metrics":metrics,"thresholds":limits,"checks":checks,"promotable":all(checks.values())}


def reliability_svg(probabilities, labels, *, bins: int=15, path=None):
    """Write a standalone reliability plot SVG using the benchmark's top-label bins."""
    confidence=[max(row) for row in probabilities]
    hits=[int(max(range(len(row)),key=row.__getitem__)==label) for row,label in zip(probabilities,labels)]
    points=[]
    for b in range(bins):
        ix=[i for i,c in enumerate(confidence) if b/bins<=c<(b+1)/bins or (b==bins-1 and c==1)]
        if ix: points.append((sum(confidence[i] for i in ix)/len(ix),sum(hits[i] for i in ix)/len(ix)))
    coords=" ".join(f"{40+450*x:.2f},{490-450*y:.2f}" for x,y in points)
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="540" height="540" viewBox="0 0 540 540"><rect width="540" height="540" fill="white"/><path d="M40 490 L490 40" stroke="#999" stroke-dasharray="5,5"/><path d="M40 490 H490 V40" fill="none" stroke="#222"/><polyline points="{coords}" fill="none" stroke="#1769aa" stroke-width="3"/></svg>\n'
    if path is not None:
        from pathlib import Path
        Path(path).write_text(svg,encoding="utf-8")
    return svg
