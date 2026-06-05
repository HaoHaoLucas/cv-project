#!/usr/bin/env bash
# 每分钟写一行状态，供监视轮询
LOG=/tmp/pipeline_poll.log
ZIP=/root/cv-project/data/coco/train2014.zip
EXP=13510573713
while true; do
  ts=$(date -Iseconds)
  sz=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
  pct=$(python3 -c "print(f'{$sz/$EXP*100:.2f}')" 2>/dev/null || echo 0)
  rem=$(python3 -c "print(f'{($EXP-$sz)/1024/1024:.0f}')" 2>/dev/null || echo ?)
  aria2=$(pgrep -f 'aria2c.*train2014' >/dev/null && echo yes || echo no)
  vg=$(pgrep -af 'run_vg_eval|sweep_vg' 2>/dev/null | head -1 | sed 's/.*python3/python3/' | cut -c1-80)
  nimg=$(find /root/cv-project/data/coco/train2014 -maxdepth 1 -name '*.jpg' 2>/dev/null | wc -l)
  done_f=$([ -f /tmp/pipeline_all_done.flag ] && echo DONE || echo -)
  dl=$(grep -oE 'DL:[^ ]+' /tmp/train2014_aria2.log 2>/dev/null | tail -1)
  echo "$ts | zip=${pct}% rem=${rem}MiB aria2=$aria2 imgs=$nimg $dl $done_f ${vg:-}" >> "$LOG"
  test -f /tmp/pipeline_all_done.flag && break
  sleep 60
done
