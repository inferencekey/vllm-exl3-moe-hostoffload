"""Create one B300 pod for the vllm-exl3 test. Usage: rp_create_b300.py <dc|any> [image]
Prints CREATE 201 ... JSON on success, FAIL <code> <msg> otherwise. Never uses a network volume."""
import os, json, urllib.request, urllib.error, sys
KEY=os.environ["RUNPOD_API_KEY"]; UA="Mozilla/5.0 inferencekey-deepseek-bench"
PUBKEY=open(os.path.expanduser("~/.ssh/runpod.pub")).read().strip()
dc=sys.argv[1]; image=sys.argv[2] if len(sys.argv)>2 else "vllm/vllm-openai:deepseekv41-flash-0909"
start=("export DEBIAN_FRONTEND=noninteractive; export -p | grep -v PUBLIC_KEY > /root/.pod_env; "
       "apt-get update && apt-get install -y openssh-server git build-essential ninja-build tmux curl rsync && "
       "mkdir -p /root/.ssh /run/sshd && echo \"$PUBLIC_KEY\" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys && "
       "echo \". /root/.pod_env\" >> /root/.bashrc && /usr/sbin/sshd; sleep infinity")
body={"name":"ik-vllmexl3-b300","cloudType":"SECURE","computeType":"GPU","gpuTypeIds":["NVIDIA B300 SXM6 AC"],"gpuCount":1,
  "imageName":image,"containerDiskInGb":600,"minRAMPerGPU":240,"allowedCudaVersions":["13.0"],"minVCPUPerGPU":16,
  "ports":["22/tcp","8000/http"],"env":{"PUBLIC_KEY":PUBKEY,"HF_HUB_ENABLE_HF_TRANSFER":"1"},
  "dockerEntrypoint":["bash","-c"],"dockerStartCmd":[start]}
if dc!="any": body["dataCenterIds"]=[dc]
def call(m,u,b=None):
    r=urllib.request.Request(u,data=json.dumps(b).encode() if b else None,method=m)
    for k,v in {"Authorization":"Bearer "+KEY,"User-Agent":UA,"Content-Type":"application/json","Accept":"application/json"}.items(): r.add_header(k,v)
    x=urllib.request.urlopen(r,timeout=90); t=x.read().decode(); return json.loads(t) if t.strip() else {}
try:
    d=call("POST","https://rest.runpod.io/v1/pods",body)
    print(f"CREATE 201 id={d.get('id')} ram={d.get('memoryInGb')} vcpu={d.get('vcpuCount')} cost={d.get('costPerHr')} dc={dc}")
    print(json.dumps(d)[:3000])
except urllib.error.HTTPError as e: print("FAIL",e.code,e.read().decode()[:300].replace("\n"," "))
