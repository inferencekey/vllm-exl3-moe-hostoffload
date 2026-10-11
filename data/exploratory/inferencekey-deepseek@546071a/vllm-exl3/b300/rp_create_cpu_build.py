"""CPU pod with the same vLLM image to prebuild the ExLlamaV3 + vllm-exl3 wheels for sm_103 (no GPU needed)."""
import os, json, urllib.request, urllib.error, sys
sys.argv=[sys.argv[0],"any"]
src=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"rp_create_b300.py")).read().split("body=")[0]
exec(src)
body={"name":"ik-vllmexl3-wheelbuild","cloudType":"SECURE","computeType":"CPU","cpuFlavorIds":["cpu5c","cpu3c"],"vcpuCount":16,
  "imageName":"vllm/vllm-openai:deepseekv41-flash-0909","containerDiskInGb":60,"ports":["22/tcp"],
  "env":{"PUBLIC_KEY":PUBKEY},"dockerEntrypoint":["bash","-c"],"dockerStartCmd":[start]}
r=urllib.request.Request("https://rest.runpod.io/v1/pods",data=json.dumps(body).encode(),method="POST")
for k,v in {"Authorization":"Bearer "+KEY,"User-Agent":UA,"Content-Type":"application/json","Accept":"application/json"}.items(): r.add_header(k,v)
try:
    d=json.loads(urllib.request.urlopen(r,timeout=90).read().decode())
    print(f"CREATE 201 id={d.get('id')} ram={d.get('memoryInGb')} vcpu={d.get('vcpuCount')} cost={d.get('costPerHr')}")
except urllib.error.HTTPError as e: print("FAIL",e.code,e.read().decode()[:300])
