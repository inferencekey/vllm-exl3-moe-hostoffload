#!/usr/bin/env bash
# On the pod: download (background) + install from prebuilt wheels, then serve once the download is complete.
. /root/.pod_env 2>/dev/null || true
cd /root
tmux new -d -s dl "bash /root/pod_download.sh > /root/download.log 2>&1"
bash /root/pod_install.sh > /root/install.log 2>&1
until grep -q DOWNLOAD_DONE /root/download.log; do sleep 10; done
MODE=${MODE:-eager} bash /root/pod_serve.sh > /root/serve.log 2>&1
