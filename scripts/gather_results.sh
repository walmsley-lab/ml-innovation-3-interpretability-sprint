#!/usr/bin/env bash
# Collect distributed artifacts into the local tree for analysis.
# Units are content-named by (state, corpus) so cross-machine merge is safe:
# a duplicate is byte-identical work, not a conflict.
set -u
GC=/home/patrickwalmsley/google-cloud-sdk/bin/gcloud
ZONE=us-west1-a
VMS="dsi-cpu-bench dsi-cpu-w2 dsi-w3 pdp-gpu"
DIRS="vsd_matrix mediator_validation mediator_discovery b2_factorial_h1 b2_factorial_h2 h23"
for VM in $VMS; do
  for a in 1 2 3; do
    if $GC compute ssh "$VM" --zone $ZONE --ssh-flag="-o ConnectTimeout=20" --command \
      'H=$HOME; [ -d /home/dsi/work ] && H=/home/dsi; cd $H/work && tar czf /tmp/gather.tgz $(for d in '"$DIRS"'; do [ -d artifacts/$d/units ] && echo artifacts/$d/units; done) 2>/dev/null; echo ok' 2>/dev/null | grep -q ok; then
      $GC compute ssh "$VM" --zone $ZONE --ssh-flag="-o ConnectTimeout=20" --command \
        'H=$HOME; [ -d /home/dsi/work ] && H=/home/dsi; sudo cp /tmp/gather.tgz ~/g.tgz 2>/dev/null || cp /tmp/gather.tgz ~/g.tgz; sudo chown $(whoami) ~/g.tgz 2>/dev/null; true' >/dev/null 2>&1
      if $GC compute scp "$VM:~/g.tgz" "/tmp/g_$VM.tgz" --zone $ZONE --quiet >/dev/null 2>&1; then
        rm -rf "/tmp/gx_$VM"; mkdir -p "/tmp/gx_$VM"
        tar xzf "/tmp/g_$VM.tgz" -C "/tmp/gx_$VM" 2>/dev/null
        for d in $DIRS; do
          if [ -d "/tmp/gx_$VM/artifacts/$d/units" ]; then
            mkdir -p "artifacts/$d/units"
            cp -n "/tmp/gx_$VM/artifacts/$d/units"/*.json "artifacts/$d/units/" 2>/dev/null
          fi
        done
        echo "$VM gathered"; break
      fi
    fi
  done
done
for d in $DIRS; do
  n=$(ls artifacts/$d/units/*.json 2>/dev/null | wc -l)
  [ "$n" -gt 0 ] && echo "  $d: $n units"
done
