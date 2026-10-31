#!/bin/bash
# 学習 → Gazebo 起動 → 評価 を、無人でまとめて実行するスクリプト。
#
# 使い方（Gazebo は閉じた状態で、Ubuntu のターミナルから）:
#   cd /workspace/pai_final/src
#   bash run_all.sh
#
# 経過はすべて /workspace/pai_final/logs/ に保存される:
#   train.log   学習の経過
#   gazebo.log  Gazebo の出力
#   eval.log    評価の結果（最後に「まとめ」が出る）

DATA=/workspace/pai_final/dataset_auto
MODEL=/workspace/pai_final/models/bc_auto_v1.pt
LOGS=/workspace/pai_final/logs
SRC=/workspace/pai_final/src
mkdir -p "$LOGS" /workspace/pai_final/models /workspace/pai_final/results

echo "[$(date '+%H:%M')] 1/3 学習を開始します（Gazebo は閉じておくこと）"
cd "$SRC"
python3 -u train_bc.py --data_dir "$DATA" --out "$MODEL" --val_episodes 10 > "$LOGS/train.log" 2>&1
if [ ! -f "$MODEL" ]; then
    echo "[ERROR] 学習に失敗しました。$LOGS/train.log を確認してください"
    exit 1
fi
echo "[$(date '+%H:%M')] 学習が終わりました"

echo "[$(date '+%H:%M')] 2/3 Gazebo を起動します"
source ~/airobot_ws/install/setup.bash
ros2 launch crane_plus_gazebo crane_plus_with_table.launch.py > "$LOGS/gazebo.log" 2>&1 &
GAZEBO_PID=$!
sleep 90   # Gazebo とコントローラの起動を待つ

echo "[$(date '+%H:%M')] 3/3 評価を開始します（ずれ幅 ±2〜5cm、各10回）"
cd /workspace/pai_final/results
python3 -u "$SRC/run_policy.py" --model "$MODEL" --delta 0.02 0.03 0.04 0.05 --trials 10 \
    > "$LOGS/eval.log" 2>&1
echo "[$(date '+%H:%M')] 評価が終わりました。結果は $LOGS/eval.log の最後にあります"
tail -8 "$LOGS/eval.log"

# 終わったら Gazebo を止める
kill $GAZEBO_PID 2>/dev/null
pkill -f "ign gazebo" 2>/dev/null
echo "[$(date '+%H:%M')] すべて完了"
