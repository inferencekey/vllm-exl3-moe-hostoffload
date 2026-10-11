import os, json, urllib.request, sys
KEY=os.environ["RUNPOD_API_KEY"]; UA="Mozilla/5.0 inferencekey-deepseek-bench"
q='query { pod(input:{podId:"%s"}) { id name desiredStatus lastStatusChange imageName uptimeSeconds runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort type } } machine { gpuDisplayName location } } }'%sys.argv[1]
r=urllib.request.Request("https://api.runpod.io/graphql",data=json.dumps({"query":q}).encode(),method="POST")
for k,v in {"Authorization":"Bearer "+KEY,"User-Agent":UA,"Content-Type":"application/json"}.items(): r.add_header(k,v)
print(json.dumps(json.loads(urllib.request.urlopen(r,timeout=60).read().decode()),indent=1))
