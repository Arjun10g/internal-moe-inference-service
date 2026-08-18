#!/usr/bin/env sh
set -eu

repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
model_dir=$(mktemp -d "${TMPDIR:-/tmp}/tiny-model.XXXXXX")
network_name="llm-smoke-$$"
service_name="llm-smoke-service-$$"

cleanup() {
  docker rm -f "$service_name" >/dev/null 2>&1 || true
  docker network rm "$network_name" >/dev/null 2>&1 || true
  find "$model_dir" -type f -delete 2>/dev/null || true
  find "$model_dir" -depth -type d -empty -delete 2>/dev/null || true
}
trap cleanup EXIT INT TERM

cd "$repo_dir"
.venv/bin/python scripts/create_tiny_test_model.py "$model_dir"
docker build --target runtime -t llm-inference-service:smoke .
docker network create --internal "$network_name" >/dev/null
docker run -d --name "$service_name" --network "$network_name" \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=128m \
  --cap-drop ALL --security-opt no-new-privileges:true \
  -v "$model_dir:/models/model:ro" \
  -e MODEL_SOURCE=/models/model -e MODEL_DTYPE=float32 -e DEVICE=cpu \
  -e MODEL_MAX_CONTEXT=128 -e MAX_PROMPT_TOKENS=96 -e MODEL_MAX_NEW_TOKENS=16 \
  -e MAX_BATCH_TOKENS=128 -e API_KEY=container-smoke-key -e ENVIRONMENT=test \
  llm-inference-service:smoke >/dev/null

docker run --rm --network "$network_name" --entrypoint python llm-inference-service:smoke -c \
  "import json,time,urllib.request; base='http://$service_name:8000'; h={'Authorization':'Bearer container-smoke-key','Content-Type':'application/json'}; deadline=time.time()+120
while True:
 try:
  r=urllib.request.urlopen(base+'/health/ready',timeout=2)
  if r.status==200: break
 except Exception:
  if time.time()>deadline: raise
  time.sleep(.5)
p=json.dumps({'messages':[{'role':'user','content':'The sky is'}],'max_tokens':4}).encode(); q=urllib.request.Request(base+'/v1/chat/completions',data=p,headers=h); a=json.load(urllib.request.urlopen(q,timeout=60)); assert a['usage']['completion_tokens']>0; b=json.load(urllib.request.urlopen(q,timeout=60)); m=urllib.request.Request(base+'/metrics',headers=h); t=urllib.request.urlopen(m,timeout=5).read().decode(); assert 'model_load_total 1.0' in t; print(json.dumps({'passed':True,'completion_tokens':a['usage']['completion_tokens'],'model_load_total':1,'outbound_network_disabled':True}))"
