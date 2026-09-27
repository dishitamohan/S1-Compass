"""Validated one-worker HTTP JSON API."""
from __future__ import annotations
import hashlib, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit
from s1_compass.data.schema import ValidationError, validate_request, validate_output
from s1_compass.serve.policy import apply


def post_decide(service: Any, body: Any, deadline: float | None = None, *, confidence_floor: float = .75):
    started=time.perf_counter()
    try:
        request=validate_request(body)
        if deadline is not None and time.monotonic() >= deadline:
            return 504,{"error":{"code":"TIMEOUT","message":"Request deadline exceeded.","deadline_ms":500,"fallback":"caller_escalation"}}
        response=service(request,deadline)
        if deadline is not None and time.monotonic()>deadline:
            return 504,{"error":{"code":"TIMEOUT","message":"Request deadline exceeded.","deadline_ms":500,"fallback":"caller_escalation"}}
        response["latency_ms"]=int((time.perf_counter()-started)*1000)
        validate_output(response,request)
        policy=apply(response, threshold=confidence_floor)
        if policy["low_confidence_questions"]:
            return 409,{"error":{"code":"LOW_CONFIDENCE","message":"One or more answers require caller escalation.","question_ids":policy["low_confidence_questions"],"confidence_floor":confidence_floor,"fallback":"caller_escalation"},"decision":response,"policy":policy}
        return 200,response
    except ValidationError as exc:
        return 422,{"error":{"code":"INVALID_SCHEMA","message":exc.message,"path":exc.path}}
    except TimeoutError:
        return 504,{"error":{"code":"TIMEOUT","message":"Request deadline exceeded.","deadline_ms":500,"fallback":"caller_escalation"}}


def create_server(service: Callable, host="127.0.0.1", port=0, allowed_origins=None):
    worker=threading.BoundedSemaphore(1)
    service_lock=threading.RLock()
    active_service={"value":service}
    policy_config={"confidence_floor":.75}
    allowed_origins=set(allowed_origins or ("http://127.0.0.1:8001","http://localhost:8001"))
    def origin_is_allowed(origin):
        if origin in allowed_origins:
            return True
        parsed=urlsplit(origin or "")
        return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    class Handler(BaseHTTPRequestHandler):
        def _cors_headers(self):
            origin=self.headers.get("Origin")
            if origin_is_allowed(origin):
                self.send_header("Access-Control-Allow-Origin",origin)
                self.send_header("Vary","Origin")
                self.send_header("Access-Control-Allow-Methods","GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers","Content-Type")

        def do_OPTIONS(self):
            origin=self.headers.get("Origin")
            self.send_response(204 if origin_is_allowed(origin) else 403)
            self._cors_headers()
            self.send_header("Content-Length","0")
            self.end_headers()

        def _json(self,status,payload):
            encoded=json.dumps(payload,separators=(",",":"),allow_nan=False).encode()
            self.send_response(status); self.send_header("Content-Type","application/json"); self._cors_headers()
            self.send_header("Content-Length",str(len(encoded))); self.end_headers(); self.wfile.write(encoded)

        def do_GET(self):
            if self.path=="/v1/policy":
                self._json(200,dict(policy_config)); return
            if self.path!="/v1/models": self.send_error(404); return
            with service_lock:
                current=active_service["value"]
                self._json(200,getattr(current,"model_registry",{"active_model_id":getattr(current,"model_id","modernbert-base"),"models":[{"model_id":getattr(current,"model_id","modernbert-base"),"display_name":getattr(current,"display_name","ModernBERT base"),"available":True,"status":"ready","benchmark_candidate":getattr(current,"benchmark_candidate",True)}]}))

        def do_POST(self):
            if self.path=="/v1/policy":
                try: size=int(self.headers.get("Content-Length","0")); body=json.loads(self.rfile.read(size))
                except Exception: body={}
                floor=body.get("confidence_floor") if isinstance(body,dict) and set(body)=={"confidence_floor"} else None
                if isinstance(floor,bool) or not isinstance(floor,(int,float)) or not .5<=float(floor)<=.99:
                    self._json(422,{"error":{"code":"INVALID_SCHEMA","message":"confidence_floor must be a number from 0.50 through 0.99."}}); return
                policy_config["confidence_floor"]=float(floor)
                self._json(200,dict(policy_config)); return
            if self.path=="/v1/models/active":
                try: size=int(self.headers.get("Content-Length","0")); body=json.loads(self.rfile.read(size))
                except Exception: body={}
                if not isinstance(body,dict) or set(body)!={"model_id"}:
                    self._json(422,{"error":{"code":"INVALID_SCHEMA","message":"Expected model_id."}}); return
                try:
                    with service_lock:
                        active_service["value"].activate(body["model_id"])
                        self._json(200,active_service["value"].model_registry)
                except KeyError:
                    self._json(404,{"error":{"code":"MODEL_UNAVAILABLE","message":"No trained candidate is installed for that model."}})
                except RuntimeError as exc:
                    self._json(503,{"error":{"code":"MODEL_LOADING","message":str(exc)}})
                except Exception as exc:
                    self._json(422,{"error":{"code":"ACTIVATION_FAILED","message":str(exc)}})
                return
            if self.path!="/v1/decide": self.send_error(404); return
            try: size=int(self.headers.get("Content-Length","0"))
            except ValueError: size=0
            if size<0 or size>1024*1024:
                status,payload=422,{"error":{"code":"INVALID_SCHEMA","message":"Request body exceeds 1 MiB.","path":"/"}}
            else:
                try: body=json.loads(self.rfile.read(size))
                except Exception: body=None
                if not worker.acquire(blocking=False):
                    status,payload=504,{"error":{"code":"TIMEOUT","message":"Decision worker is busy.","deadline_ms":500,"fallback":"caller_escalation"}}
                else:
                    try:
                        if not service_lock.acquire(blocking=False):
                            status,payload=503,{"error":{"code":"MODEL_LOADING","message":"Model activation is in progress; retry shortly."}}
                        else:
                            try: status,payload=post_decide(active_service["value"],body,time.monotonic()+.5,confidence_floor=policy_config["confidence_floor"])
                            finally: service_lock.release()
                    finally: worker.release()
            headers={}
            if status==200:
                policy=apply(payload, threshold=policy_config["confidence_floor"])
                headers["X-S1-Compass-Policy"]="audit" if policy["audit_questions"] else "auto"
                if policy["audit_questions"]: headers["X-S1-Compass-Audit-Questions"]=json.dumps(policy["audit_questions"],separators=(",",":"))
            encoded=json.dumps(payload,separators=(",",":"),allow_nan=False).encode()
            self.send_response(status); self.send_header("Content-Type","application/json")
            self._cors_headers()
            for key,value in headers.items(): self.send_header(key,value)
            self.send_header("Content-Length",str(len(encoded))); self.end_headers(); self.wfile.write(encoded)
        def log_message(self,format,*args):
            origin=self.headers.get("Origin","-")
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {self.address_string()} origin={origin!r} {format % args}",flush=True)
    return ThreadingHTTPServer((host,port),Handler)


def load_release(path: str | Path, snapshot_path: str | Path, *, benchmark_mode: bool=False, base_model: str = "modernbert-base"):
    """Load a hash-checked candidate bundle; production requires passed gates."""
    from s1_compass.model.backbone import load, mx
    from s1_compass.model.heads import Head
    from s1_compass.train.loop import Model
    from s1_compass.data.tokenize import load_tokenizer, encode
    from s1_compass.model.scorer import decide
    from s1_compass import public_model_id
    from s1_compass.models.registry import get_base_model
    from s1_compass.models.snapshot import verify_snapshot
    spec=get_base_model(base_model)
    snapshot_lock=None
    modern_lock_path=Path(snapshot_path)/"s1-compass-snapshot.lock.json"
    if modern_lock_path.is_file():
        snapshot_lock=verify_snapshot(spec,Path(snapshot_path))
    root=Path(path); report=json.loads((root/"evaluation.json").read_text())
    manifest_path=root/"release.json"
    if manifest_path.is_file():
        release_manifest=json.loads(manifest_path.read_text())
        if release_manifest.get("base_model_id")!=base_model:
            raise ValueError("Candidate release is bound to a different base model.")
        if release_manifest.get("checkpoint_sha256")!=hashlib.sha256((root/"candidate.npz").read_bytes()).hexdigest():
            raise ValueError("Candidate checkpoint hash does not match release manifest.")
        if release_manifest.get("calibration_version")!=hashlib.sha256((root/"calibration.json").read_bytes()).hexdigest():
            raise ValueError("Calibration hash does not match release manifest.")
        if release_manifest.get("preprocessing_version")!=spec.preprocessing_version:
            raise ValueError("Candidate preprocessing version does not match this base-model loader.")
        if snapshot_lock is None:
            raise ValueError("New candidates require a hash-locked base snapshot.")
        lock_hash=hashlib.sha256(modern_lock_path.read_bytes()).hexdigest()
        if release_manifest.get("snapshot_lock_sha256")!=lock_hash or release_manifest.get("base_revision")!=snapshot_lock["revision"]:
            raise ValueError("Candidate snapshot binding does not match the selected snapshot.")
        tokenizer_hash=snapshot_lock.get("tokenizer_sha256") or next(item["sha256"] for item in snapshot_lock["files"] if item["path"]=="tokenizer.json")
        if release_manifest.get("tokenizer_sha256")!=tokenizer_hash:
            raise ValueError("Candidate tokenizer binding does not match the selected snapshot.")
        if release_manifest.get("evaluation_report_sha256")!=hashlib.sha256((root/"evaluation.json").read_bytes()).hexdigest():
            raise ValueError("Evaluation report hash does not match release manifest.")
    elif base_model!="modernbert-base":
        raise ValueError("Legacy candidates are readable only as ModernBERT releases.")
    if (not report.get("promotable") or not report.get("production_data_available")) and not benchmark_mode:
        raise ValueError("Evaluation gates failed; candidate cannot serve as a release.")
    weights=root/"candidate.npz"; calibration_path=root/"calibration.json"
    if not weights.is_file() or not calibration_path.is_file(): raise ValueError("Candidate bundle is incomplete.")
    if hashlib.sha256(weights.read_bytes()).hexdigest()!=report.get("checkpoint_sha256"):
        raise ValueError("Checkpoint hash does not match evaluation report.")
    calibration=json.loads(calibration_path.read_text())
    calibration_version=hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    if calibration_version!=report.get("calibration_version",calibration_version):
        raise ValueError("Calibration sidecar hash does not match evaluation report.")
    calibration["version"]=calibration_version
    snapshot=Path(snapshot_path); architecture="causal" if base_model=="qwen3-0.6b" else "encoder"
    if architecture=="causal":
        from s1_compass.model.backbone import load_base_model
        encoder=load_base_model(snapshot,base_model,require_parity=True,training_mode="lora")
    else:
        encoder=load(snapshot,require_parity=True)
    hidden_size=int(json.loads((snapshot/"config.json").read_text())["hidden_size"])
    head=Head(hidden_size); model=Model(encoder,head)
    tensors=mx.load(str(weights))
    if architecture=="causal":
        model.load_weights(list(tensors.items()),strict=False)
    else:
        encoder_weights={k[len("encoder."):]:v for k,v in tensors.items() if k.startswith("encoder.")}
        head_weights={k[len("head."):]:v for k,v in tensors.items() if k.startswith("head.")}
        encoder.load_weights(list(encoder_weights.items()),strict=True); head.load_weights(list(head_weights.items()),strict=True)
    model.eval()
    tokenizer=load_tokenizer(snapshot); model_id=public_model_id(base_model,calibration["model_sha"])
    def service(request,deadline):
        rows=[{"id":q["id"],"state_text":request["state_text"],"question":{"id":q["id"],"type":q["type"],"instructions":q["instructions"]},"options":q["options"],"label":"ABSTAIN","annotator_confidence":0.0,"source":"human_review"} for q in request["questions"]]
        batch=encode(rows,False,17,tokenizer=tokenizer,architecture=architecture)
        return decide(model,request,batch,calibration,model_id)
    service.model_id=base_model
    service.display_name=spec.display_name
    service.benchmark_candidate=not report.get("production_data_available",False)
    models=[{"model_id":base_model,"display_name":spec.display_name,"available":True,"status":"ready" if not service.benchmark_candidate else "benchmark candidate","benchmark_candidate":service.benchmark_candidate}]
    qwen_snapshot=Path(__import__("s1_compass.models.registry",fromlist=["get_base_model"]).get_base_model("qwen3-0.6b").snapshot_path)
    if base_model!="qwen3-0.6b" and qwen_snapshot.exists() and (qwen_snapshot/"s1-compass-snapshot.lock.json").is_file():
        models.append({"model_id":"qwen3-0.6b","display_name":"Qwen3 0.6B","available":False,"status":"Snapshot installed; no trained candidate","benchmark_candidate":True,"unavailable_reason":"A Qwen-trained release and parity-checked native MLX backbone are required."})
    service.model_registry={"active_model_id":base_model,"models":models}
    service.base_revision=snapshot_lock["revision"] if snapshot_lock else None
    service.candidate_hash=hashlib.sha256(weights.read_bytes()).hexdigest()
    def activate(model_id):
        if model_id!=base_model: raise KeyError(model_id)
        return service.model_registry
    service.activate=activate
    return service


class CandidateRouter:
    """Explicit single-resident candidate registry with serialized activation and rollback."""
    def __init__(self, candidates: dict[str, tuple[str | Path, str | Path]], active_model_id: str, *, benchmark_mode=False):
        if active_model_id not in candidates:
            raise ValueError("The active model must be listed in the explicit candidate registry.")
        self.candidates={key:(Path(value[0]),Path(value[1])) for key,value in candidates.items()}
        self.benchmark_mode=benchmark_mode
        self._active_id=active_model_id
        self._service=load_release(*self.candidates[active_model_id],benchmark_mode=benchmark_mode,base_model=active_model_id)

    @property
    def model_registry(self):
        models=[]
        for model_id,(release,snapshot) in sorted(self.candidates.items()):
            try:
                report=json.loads((release/"evaluation.json").read_text())
                checkpoint=release/"candidate.npz"; calibration=release/"calibration.json"
                manifest_path=release/"release.json"
                if not checkpoint.is_file() or not calibration.is_file():
                    raise ValueError("Candidate files are incomplete.")
                if manifest_path.is_file():
                    manifest=json.loads(manifest_path.read_text())
                    lock_file=snapshot/"s1-compass-snapshot.lock.json"
                    if manifest.get("base_model_id")!=model_id or not lock_file.is_file():
                        raise ValueError("Candidate model binding is incomplete.")
                    if manifest.get("checkpoint_sha256")!=report.get("checkpoint_sha256") or manifest.get("calibration_version")!=report.get("calibration_version"):
                        raise ValueError("Candidate artifact hashes disagree with evaluation metadata.")
                    if manifest.get("evaluation_report_sha256")!=hashlib.sha256((release/"evaluation.json").read_bytes()).hexdigest():
                        raise ValueError("Candidate evaluation hash mismatch.")
                elif model_id != "modernbert-base":
                    raise ValueError("Qwen candidates require a release manifest.")
                elif hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=report.get("checkpoint_sha256"):
                    raise ValueError("Legacy ModernBERT checkpoint hash mismatch.")
                promotable=report.get("production_data_available") and report.get("promotable")
                status="ready" if promotable else "benchmark candidate"
                available=bool(promotable or self.benchmark_mode)
                reason=None if available else "Candidate failed production gates; start the server with --benchmark-mode to use it locally."
            except Exception as exc:
                available=False; status="unavailable"; reason=str(exc)
            from s1_compass.models.registry import get_base_model
            spec=get_base_model(model_id)
            item={"model_id":model_id,"display_name":spec.display_name,"architecture":spec.architecture,
                  "available":available,"status":status,"benchmark_candidate":status!="ready"}
            if reason: item["unavailable_reason"]=reason
            models.append(item)
        from s1_compass.models.registry import get_base_model
        absent_qwen="qwen3-0.6b" not in self.candidates
        qwen_snapshot=get_base_model("qwen3-0.6b").snapshot_path
        if absent_qwen and (qwen_snapshot/"s1-compass-snapshot.lock.json").is_file():
            models.append({"model_id":"qwen3-0.6b","display_name":"Qwen3 0.6B","architecture":"causal",
                           "available":False,"status":"Snapshot installed; no release registered","benchmark_candidate":True,
                           "unavailable_reason":"Register a Qwen-trained candidate after parity, training, and calibration."})
        return {"active_model_id":self._active_id,"models":models}

    def activate(self, model_id: str):
        if model_id not in self.candidates: raise KeyError(model_id)
        if model_id==self._active_id: return self.model_registry
        previous_id=self._active_id
        self._service=None
        import gc
        gc.collect()
        try:
            from s1_compass.model.backbone import mx
            if mx is not None: mx.clear_cache()
        except Exception:
            pass
        try:
            self._service=load_release(*self.candidates[model_id],benchmark_mode=self.benchmark_mode,base_model=model_id)
            self._active_id=model_id
            print(f"Activated model={model_id} revision={self._service.base_revision} candidate_sha256={self._service.candidate_hash}",flush=True)
            return self.model_registry
        except Exception as activation_error:
            try:
                self._service=load_release(*self.candidates[previous_id],benchmark_mode=self.benchmark_mode,base_model=previous_id)
                self._active_id=previous_id
            except Exception as rollback_error:
                self._service=None
                raise RuntimeError(f"Activation failed ({activation_error}); rollback failed ({rollback_error}).") from activation_error
            raise RuntimeError(f"Activation failed; restored {previous_id}: {activation_error}") from activation_error

    def __call__(self, request, deadline):
        if self._service is None: raise RuntimeError("Model service is loading; retry shortly.")
        return self._service(request,deadline)
