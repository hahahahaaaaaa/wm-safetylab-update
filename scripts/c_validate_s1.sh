#!/usr/bin/env bash
set -euo pipefail
mkdir -p .workspace/results
LOG=".workspace/results/s1-c-validation-$(date -u +%Y%m%dT%H%M%SZ).log"
IDS=(a-obs-corruption a-obs-noise-vector a-camera-jitter a-patch-attack a-frame-guard a-input-monitor a-input-monitor-vector)
{
 echo "S1 C validation started: $(date -u +%FT%TZ)"
 for id in "${IDS[@]}"; do echo; echo "===== $id ====="; ./dev.sh plugin-test "$id"; done
 echo "S1 C validation finished: $(date -u +%FT%TZ)"
} 2>&1 | tee "$LOG"
echo "Validation log: $LOG"
