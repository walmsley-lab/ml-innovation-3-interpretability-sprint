#!/usr/bin/env bash
set -uo pipefail
ZONE=us-west1-a
mkdir -p artifacts/layer2_scout/units
for VM in dsi-cpu-bench dsi-cpu-w2; do
  gcloud compute ssh "$VM" --zone "$ZONE" --ssh-flag="-o ConnectTimeout=15" \
    --command 'cd ~/work && tar czf ~/scout_out.tgz artifacts/layer2_scout 2>/dev/null' >/dev/null 2>&1 || continue
  gcloud compute scp "$VM:~/scout_out.tgz" "/tmp/scout_$VM.tgz" --zone "$ZONE" >/dev/null 2>&1 || continue
  rm -rf "/tmp/sx_$VM" && mkdir -p "/tmp/sx_$VM"
  tar xzf "/tmp/scout_$VM.tgz" -C "/tmp/sx_$VM" 2>/dev/null
  cp -n "/tmp/sx_$VM"/artifacts/layer2_scout/units/*.json artifacts/layer2_scout/units/ 2>/dev/null
done
echo "scout units: $(ls artifacts/layer2_scout/units 2>/dev/null | wc -l)/15"
