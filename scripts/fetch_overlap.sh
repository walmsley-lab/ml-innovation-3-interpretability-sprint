#!/usr/bin/env bash
set -uo pipefail
mkdir -p artifacts/layer2_overlap/units
for VM in ${DSI_WORKERS:?set DSI_WORKERS}; do
  gcloud compute ssh "$VM" --zone "${DSI_ZONE:-us-west1-a}" --ssh-flag="-o ConnectTimeout=15" \
    --command 'cd ~/work && tar czf ~/ovl.tgz artifacts/layer2_overlap 2>/dev/null' >/dev/null 2>&1 || continue
  gcloud compute scp "$VM:~/ovl.tgz" "/tmp/ovl_$VM.tgz" --zone "${DSI_ZONE:-us-west1-a}" >/dev/null 2>&1 || continue
  rm -rf "/tmp/ox_$VM" && mkdir -p "/tmp/ox_$VM"
  tar xzf "/tmp/ovl_$VM.tgz" -C "/tmp/ox_$VM" 2>/dev/null
  cp -n "/tmp/ox_$VM"/artifacts/layer2_overlap/units/*.json artifacts/layer2_overlap/units/ 2>/dev/null
done
echo "overlap units: $(ls artifacts/layer2_overlap/units 2>/dev/null | wc -l)/15"
