"""Start the local S1 Compass decision API."""
from __future__ import annotations
import argparse
from s1_compass.serve.api import create_server, load_release, CandidateRouter
from s1_compass.models.registry import get_base_model


def main():
    parser=argparse.ArgumentParser(description="Serve a local S1 Compass model candidate.")
    parser.add_argument("--release",default="artifacts/smoke/run",help="candidate or promoted release directory")
    parser.add_argument("--candidate",action="append",default=[],help="explicit installed candidate MODEL_ID=RELEASE_DIR; repeat for multiple models")
    parser.add_argument("--snapshot",default=None,help="hash-locked snapshot (defaults to the selected base-model path)")
    parser.add_argument("--base-model",choices=("modernbert-base","qwen3-0.6b"),default="modernbert-base")
    parser.add_argument("--host",default="127.0.0.1")
    parser.add_argument("--port",type=int,default=8000)
    parser.add_argument("--benchmark-mode",action="store_true",help="allow serving an unevaluated local candidate")
    parser.add_argument("--client-origin",action="append",default=None,help="allowed browser origin; repeat to allow multiple local browser origins")
    args=parser.parse_args()
    spec=get_base_model(args.base_model)
    candidate_specs={}
    for value in args.candidate:
        if "=" not in value: parser.error("--candidate must be MODEL_ID=RELEASE_DIR")
        model_id,release_path=value.split("=",1)
        if model_id in candidate_specs: parser.error(f"duplicate candidate {model_id}")
        candidate_specs[model_id]=(release_path,get_base_model(model_id).snapshot_path)
    if candidate_specs:
        if args.base_model not in candidate_specs: parser.error("--base-model must be included among --candidate entries")
        service=CandidateRouter(candidate_specs,args.base_model,benchmark_mode=args.benchmark_mode)
    else:
        service=load_release(args.release,args.snapshot or spec.snapshot_path,benchmark_mode=args.benchmark_mode,base_model=args.base_model)
    origins=set(args.client_origin or {"http://127.0.0.1:8001","http://localhost:8001"})
    server=create_server(service,args.host,args.port,allowed_origins=origins)
    active=getattr(service,"_service",service)
    print(f"S1 Compass API listening on http://{args.host}:{server.server_port}/v1/decide active_model={args.base_model} revision={getattr(active,'base_revision','unknown')} candidate_sha256={getattr(active,'candidate_hash','unknown')}",flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()


if __name__=="__main__": main()
