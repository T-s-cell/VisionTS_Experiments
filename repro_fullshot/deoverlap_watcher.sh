#!/usr/bin/env bash
# theta de-overlap watcher (user-approved plan 2026-09-28).
# Fully autonomous endgame, no Claude session required:
#   1. touch PAUSE_w0 when w0 starts its last Traffic job (Traffic_336_seed2022)
#   2. touch PAUSE_w1 when w1 starts its last Traffic job (Traffic_336_seed2021)
#      -> original workers sleep forever at that boundary (PAUSE checked before
#         every job; Electricity and beyond are eta's territory now)
#   3. launch t720 continuation worker 0 once PAUSE_w0 set AND Traffic_336_seed2022 SUCCESS
#   4. launch t720 continuation worker 1 once PAUSE_w1 set AND Traffic_336_seed2021 SUCCESS
#      -> t720 script has its own VRAM gate (MIN_FREE_MB=22700) and its own
#         sentinel namespace (PAUSE720_*), resume-skips any already-done job
# Exits after all four steps. Only touches files / launches the approved
# t720 scripts. Log: pause_watcher.log
set -u
REPO="$HOME/VisionTS_Experiments/repro_fullshot"
LOG="$REPO/pause_watcher.log"
cd "$REPO" || exit 1

say() { echo "$(date '+%F %T') $*" >> "$LOG"; }
T720=run_all_3090_t720.sh
S0=0; S1=0; L0=0; L1=0
say "watcher start (pid $$)"

while [ $S0 -eq 0 ] || [ $S1 -eq 0 ] || [ $L0 -eq 0 ] || [ $L1 -eq 0 ]; do
  if [ $S0 -eq 0 ] && grep -q "start Traffic_336_seed2022 (gpu 0)" queue_status_theta_w0.txt; then
    touch PAUSE_w0; S0=1; say "PAUSE_w0 set (w0 entered Traffic_336_seed2022)"
  fi
  if [ $S1 -eq 0 ] && grep -q "start Traffic_336_seed2021 (gpu 1)" queue_status_theta_w1.txt; then
    touch PAUSE_w1; S1=1; say "PAUSE_w1 set (w1 entered Traffic_336_seed2021)"
  fi
  if [ $L0 -eq 0 ] && [ -f PAUSE_w0 ] && [ -f "$HOME/VisionTS_Experiments/long_term_tsf/save_fullshot/Traffic_336_seed2022/SUCCESS.txt" ]; then
    nohup bash "$T720" 0 > logs_theta/w0_t720_nohup.log 2>&1 &
    L0=1; say "t720 worker0 launched (pid $!)"
  fi
  if [ $L1 -eq 0 ] && [ -f PAUSE_w1 ] && [ -f "$HOME/VisionTS_Experiments/long_term_tsf/save_fullshot/Traffic_336_seed2021/SUCCESS.txt" ]; then
    nohup bash "$T720" 1 > logs_theta/w1_t720_nohup.log 2>&1 &
    L1=1; say "t720 worker1 launched (pid $!)"
  fi
  sleep 120
done
say "all sentinels set and t720 workers launched, watcher exits"
