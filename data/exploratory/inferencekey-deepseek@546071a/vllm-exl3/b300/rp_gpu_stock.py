import os, json, urllib.request
KEY=os.environ["RUNPOD_API_KEY"]; UA="Mozilla/5.0 inferencekey-deepseek-bench"
q='''query { gpuTypes { id displayName memoryInGb secureCloud communityCloud securePrice communityPrice
  lowestPrice(input:{gpuCount:1}) { stockStatus minimumBidPrice uninterruptablePrice } } }'''
r=urllib.request.Request("https://api.runpod.io/graphql",data=json.dumps({"query":q}).encode(),method="POST")
for k,v in {"Authorization":"Bearer "+KEY,"User-Agent":UA,"Content-Type":"application/json"}.items(): r.add_header(k,v)
d=json.loads(urllib.request.urlopen(r,timeout=60).read().decode())
for g in d["data"]["gpuTypes"]:
    if (g.get("memoryInGb") or 0) >= 80:
        lp=g.get("lowestPrice") or {}
        print(f'{g["id"]:35s} {g["memoryInGb"]:4} GB secure={g["securePrice"]} stock={lp.get("stockStatus")}')
