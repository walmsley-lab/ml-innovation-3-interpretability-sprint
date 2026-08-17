#!/usr/bin/env bash
# Merge development artifacts from both workers into one local pool.
# Units are content-identical across machines when both computed the same
# shard, so a plain merge is safe; the aggregator re-validates every unit.
set -uo pipefail
ZONE=us-west1-a
mkdir -p artifacts/wikitext_transfer/units artifacts/wikitext_transfer/controls
for VM in ${DSI_WORKERS:?set DSI_WORKERS}; do
  gcloud compute ssh "$VM" --zone "${DSI_ZONE:-us-west1-a}" --ssh-flag="-o ConnectTimeout=15" \
    --command 'cd ~/work && tar czf ~/wt.tgz artifacts/wikitext_transfer 2>/dev/null' >/dev/null 2>&1 || continue
  gcloud compute scp "$VM:~/wt.tgz" "/tmp/wt_$VM.tgz" --zone "${DSI_ZONE:-us-west1-a}" >/dev/null 2>&1 || continue
  rm -rf "/tmp/x_$VM" && mkdir -p "/tmp/x_$VM"
  tar xzf "/tmp/wt_$VM.tgz" -C "/tmp/x_$VM" 2>/dev/null
  cp -n "/tmp/x_$VM"/artifacts/wikitext_transfer/units/*.json artifacts/wikitext_transfer/units/ 2>/dev/null
  cp -n "/tmp/x_$VM"/artifacts/wikitext_transfer/controls/*.json artifacts/wikitext_transfer/controls/ 2>/dev/null
done
echo "units:    $(ls artifacts/wikitext_transfer/units 2>/dev/null | wc -l)/78"
echo "controls: $(ls artifacts/wikitext_transfer/controls 2>/dev/null | wc -l)/21"
