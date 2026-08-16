#!/usr/bin/env bash
# Pull the VM matrices. The VM's complementary run lands in
# artifacts/natural_transfer_vm/, deliberately NOT the local
# artifacts/natural_transfer/: one is the same-environment comparator for H3,
# the other is the cross-environment reproducibility check, and merging them
# would destroy both readings.
set -euo pipefail
ZONE=us-west1-a
VM=dsi-cpu-bench
mkdir -p artifacts/natural_common_control artifacts/natural_transfer_vm

# Stage 3 may not have created its directory yet; tar only what exists so a
# mid-wave fetch is still useful.
gcloud compute ssh "$VM" --zone "$ZONE" --command \
  'cd ~/work && D=""; for d in artifacts/natural_common_control artifacts/natural_transfer; do [ -d "$d" ] && D="$D $d"; done; tar czf ~/h3_results.tgz $D'
gcloud compute scp "$VM:~/h3_results.tgz" /tmp/h3_results.tgz --zone "$ZONE"
rm -rf /tmp/h3_extract && mkdir -p /tmp/h3_extract
tar xzf /tmp/h3_results.tgz -C /tmp/h3_extract
cp -r /tmp/h3_extract/artifacts/natural_common_control/. artifacts/natural_common_control/
[ -d /tmp/h3_extract/artifacts/natural_transfer ] && \
  cp -r /tmp/h3_extract/artifacts/natural_transfer/. artifacts/natural_transfer_vm/ || true
echo "H3 units:   $(ls artifacts/natural_common_control/units 2>/dev/null | wc -l)"
echo "H3 controls:$(ls artifacts/natural_common_control/controls 2>/dev/null | wc -l)"
echo "VM comp:    $(ls artifacts/natural_transfer_vm/units 2>/dev/null | wc -l)"
echo "local comp: $(ls artifacts/natural_transfer/units 2>/dev/null | wc -l)"
