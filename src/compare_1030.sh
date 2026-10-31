#!/bin/bash
# 今の環境（10/30）で、固定動作とモデルの成功率を測り直すスクリプト。
# Gazebo を起動した状態で（テレオペは止めて）実行する:
#   bash /workspace/pai_final/src/compare_1030.sh
#
# 1. 固定動作   ±2〜5cm 各10回        → results/replay_1030/
# 2. 固定動作   ±5cm   30回           → results/replay_1030_n30/
# 3. モデル     ±5cm   30回           → results/model_1030_n30/
# 全体で 45分〜1時間。経過は logs/compare_1030.log にも残る。

SRC=/workspace/pai_final/src
RES=/workspace/pai_final/results
LOG=/workspace/pai_final/logs/compare_1030.log
mkdir -p $RES/replay_1030 $RES/replay_1030_n30 $RES/model_1030_n30 /workspace/pai_final/logs

{
echo "[$(date '+%H:%M')] 1/3 固定動作 ±2〜5cm 各10回"
cd $RES/replay_1030
python3 -u $SRC/evaluate_replay.py --delta 0.02 0.03 0.04 0.05 --trials 10

echo "[$(date '+%H:%M')] 2/3 固定動作 ±5cm 30回"
cd $RES/replay_1030_n30
python3 -u $SRC/evaluate_replay.py --delta 0.05 --trials 30

echo "[$(date '+%H:%M')] 3/3 モデル ±5cm 30回"
cd $RES/model_1030_n30
python3 -u $SRC/run_policy.py --model /workspace/pai_final/models/bc_auto_v1.pt --delta 0.05 --trials 30

echo "[$(date '+%H:%M')] すべて完了"
} 2>&1 | tee $LOG
