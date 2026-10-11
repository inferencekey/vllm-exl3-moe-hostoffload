# Price and stock of RTX PRO 6000 types for gpuCount=2 per datacenter (read-only GraphQL).
import os, json, urllib.request
KEY=os.environ["RUNPOD_API_KEY"]; UA="Mozilla/5.0 inferencekey-deepseek-bench"
def q(query):
    r=urllib.request.Request("https://api.runpod.io/graphql",data=json.dumps({"query":query}).encode(),method="POST")
    for k,v in {"Authorization":"Bearer "+KEY,"User-Agent":UA,"Content-Type":"application/json"}.items(): r.add_header(k,v)
    return json.loads(urllib.request.urlopen(r,timeout=60).read().decode())
for gid in ["NVIDIA RTX PRO 6000 Blackwell Server Edition","NVIDIA RTX PRO 6000 Blackwell Workstation Edition"]:
    for dc in ["EUR-IS-1","EU-NL-1","CA-MTL-3","US-KS-2","EU-RO-1","EU-SE-1","US-TX-3"]:
        d=q('query { gpuTypes(input:{id:"%s"}) { id securePrice lowestPrice(input:{gpuCount:2, dataCenterId:"%s", secureCloud:true}) { stockStatus uninterruptablePrice minMemory minVcpu maxUnreservedGpuCount } } }'%(gid,dc))
        for g in d.get("data",{}).get("gpuTypes",[]) or []:
            lp=g.get("lowestPrice") or {}
            if lp.get("stockStatus"): print(gid[-17:],dc,lp)
