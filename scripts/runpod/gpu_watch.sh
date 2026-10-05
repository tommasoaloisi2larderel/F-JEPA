#!/usr/bin/env bash
set -euo pipefail

watch -n 2 'nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv && echo && ps -eo pid,etime,pcpu,pmem,cmd | grep -E "python train.py|PID" | grep -v grep'
