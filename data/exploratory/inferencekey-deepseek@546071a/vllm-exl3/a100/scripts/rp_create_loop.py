# Wait for capacity: try to create the 2x RTX PRO 6000 pod every 45 s, rotating datacenters, for up to MAX_WAIT_S.
# No pod exists while waiting. A pod that comes up with too little RAM or too high a price is deleted at once.
# On success writes the pod id to OUT (default ./pod_id) and exits 0.
import os, json, sys, time, urllib.request, urllib.error
KEY=os.environ["RUNPOD_API_KEY"]; UA="Mozilla/5.0 inferencekey-deepseek-bench"
PUBKEY=open(os.path.expanduser("~/.ssh/runpod.pub")).read().strip()
OUT=os.environ.get("OUT","pod_id"); MAX_WAIT_S=int(os.environ.get("MAX_WAIT_S","14400"))
MIN_MEM=int(os.environ.get("MIN_MEM_GB","250")); MAX_COST=float(os.environ.get("MAX_COST","4.40"))
IMAGE="vllm/vllm-openai:nightly-36768d1bfd39094681cdbc8cb37d4b31c0729c89"
START=("apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server >/dev/null && "
       "mkdir -p /root/.ssh /run/sshd && echo \"$PUBLIC_KEY\" >> /root/.ssh/authorized_keys && chmod 700 /root/.ssh && "
       "chmod 600 /root/.ssh/authorized_keys && /usr/sbin/sshd && sleep infinity")
def call(method, url, body=None):
    req=urllib.request.Request(url, data=json.dumps(body).encode() if body else None, method=method)
    for k,v in {"Authorization":"Bearer "+KEY,"User-Agent":UA,"Content-Type":"application/json","Accept":"application/json"}.items(): req.add_header(k,v)
    try:
        r=urllib.request.urlopen(req,timeout=90); return r.status, json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e: return e.code, e.read().decode()[:300]
    except Exception as e: return 0, repr(e)[:300]
DCS=["EUR-IS-1","EU-NL-1","CA-MTL-3"]
t0=time.time(); i=0
while time.time()-t0 < MAX_WAIT_S:
    dc=DCS[i%len(DCS)]; i+=1
    body={"name":"ik-vllmexl3-a100class-2xpro6000","cloudType":"SECURE","computeType":"GPU",
          "gpuTypeIds":["NVIDIA RTX PRO 6000 Blackwell Server Edition","NVIDIA RTX PRO 6000 Blackwell Workstation Edition"],
          "gpuTypePriority":"custom","gpuCount":2,"minRAMPerGPU":125,"minVCPUPerGPU":16,"containerDiskInGb":450,
          "imageName":IMAGE,"dataCenterIds":[dc],"ports":["22/tcp","8000/http"],
          "env":{"PUBLIC_KEY":PUBKEY},"dockerEntrypoint":["bash","-c"],"dockerStartCmd":[START]}
    st,d=call("POST","https://rest.runpod.io/v1/pods",body)
    ts=time.strftime("%H:%M:%S")
    if st in (200,201) and isinstance(d,dict) and d.get("id"):
        pid=d["id"]; mem=d.get("memoryInGb"); cost=d.get("costPerHr")
        print(f"{ts} CREATED {pid} dc={dc} cost={cost} memoryInGb={mem} vcpu={d.get('vcpuCount')} gpu={d.get('machine',{}).get('gpuTypeId') if isinstance(d.get('machine'),dict) else ''}", flush=True)
        if (mem is not None and float(mem) < MIN_MEM) or (cost is not None and float(cost) > MAX_COST):
            print(f"{ts} GATE FAIL, deleting {pid}:", call("DELETE",f"https://rest.runpod.io/v1/pods/{pid}")[0], flush=True)
            time.sleep(45); continue
        open(OUT,"w").write(pid+"\n"); json.dump(d,open(OUT+".json","w"),indent=1); sys.exit(0)
    print(f"{ts} {dc} -> {st} {str(d)[:140]}", flush=True)
    time.sleep(45)
print("NO CAPACITY within window"); sys.exit(3)
