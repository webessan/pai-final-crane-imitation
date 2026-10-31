#!/usr/bin/env python3
"""キューブの位置をランダムにずらしながら action_publisher.py を何度も実行し、
成功率を自動で測るスクリプト。

使い方（Gazebo を起動した状態で）:
    python3 evaluate_replay.py --delta 0.02 --trials 10
    python3 evaluate_replay.py --delta 0.04 0.05 0.06 --trials 10   # 複数まとめて

    --delta   ずれ幅 [m]（例: 0.02 = ±2cm）。スペース区切りで複数指定できる
    --trials  試行回数（既定 10）
    --seed    乱数の種（同じ値にすると毎回同じ位置の並びになる。既定 0）
    --action  実行する動作プログラムのパス
    --hold    持ち上げた状態で静止する時間 [秒]（既定 2.0）

各試行の流れ: キューブをずらして置く → 掴んで持ち上げる → 判定
              → 持ち上げたまま --hold 秒静止 → キューブを離す
成功判定: 動作が終わった時点で、キューブが元の高さより 30cm 以上高い
          （＝一番上まで持ち上がっている）なら成功。
結果は画面に表示し、ずれ幅ごとに results_delta<ずれ幅>.csv に保存する。
複数のずれ幅をまとめて実行したときは summary.csv に成功率の一覧も追記する。
"""
import argparse
import csv
import os
import random
import re
import shutil
import subprocess
import time

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

# キューブの元の位置（Gazebo 起動直後に実測した値）
BASE_X = 0.0
BASE_Y = -0.1
BASE_Z = 1.0035
CUBE_NAME = "wood_cube_5cm"
LIFT_THRESHOLD = 0.30  # 元の高さより30cm以上（=一番上付近）なら成功 [m]

DEFAULT_ACTION = os.path.expanduser(
    "~/pai_ws/src/physicalai_pubsub/physicalai_pubsub/action_publisher.py")

CMD_BIN = shutil.which("ign") or shutil.which("gz")
MSG_PREFIX = "ignition.msgs" if CMD_BIN and "ign" in CMD_BIN else "gz.msgs"


class Resetter(Node):
    """アームを初期姿勢に戻し、グリッパを開くためのノード"""

    def __init__(self):
        super().__init__('evaluate_replay_resetter')
        self.arm_pub = self.create_publisher(
            JointTrajectory, 'crane_plus_arm_controller/joint_trajectory', 10)
        self.gripper_pub = self.create_publisher(
            JointTrajectory, 'crane_plus_gripper_controller/joint_trajectory', 10)
        time.sleep(1.0)  # 最初の指令の取りこぼし防止

    def _send(self, pub, names, positions, sec=2.0):
        msg = JointTrajectory()
        msg.joint_names = names
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start = Duration(sec=int(sec), nanosec=int((sec % 1) * 1e9))
        msg.points.append(pt)
        pub.publish(msg)

    def reset_robot(self):
        # グリッパを開いてキューブを離す → アームを初期姿勢へ
        self._send(self.gripper_pub, ['crane_plus_joint_hand'], [-0.5236])
        self._send(self.arm_pub,
                   ['crane_plus_joint1', 'crane_plus_joint2',
                    'crane_plus_joint3', 'crane_plus_joint4'],
                   [0.0, 0.0, 0.0, 0.0])


def set_cube_pose(x, y, z):
    req = (f'name: "{CUBE_NAME}", position: {{x: {x}, y: {y}, z: {z}}}, '
           f'orientation: {{x: 0, y: 0, z: 0, w: 1}}')
    res = subprocess.run(
        [CMD_BIN, "service", "-s", "/world/default/set_pose",
         "--reqtype", f"{MSG_PREFIX}.Pose", "--reptype", f"{MSG_PREFIX}.Boolean",
         "--timeout", "1000", "--req", req],
        capture_output=True, text=True)
    return "true" in res.stdout.lower()


def get_cube_z():
    """Gazebo からキューブの現在の高さ z を読み取る（読めなければ None）"""
    try:
        res = subprocess.run(
            [CMD_BIN, "topic", "-e", "-t", "/world/default/pose/info", "-n", "1"],
            capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return None
    text = res.stdout
    i = text.find(f'name: "{CUBE_NAME}"')
    if i < 0:
        return None
    m = re.search(r'position\s*\{[^}]*?z:\s*([-0-9.eE+]+)', text[i:])
    return float(m.group(1)) if m else None


def run_one_delta(resetter, delta, args):
    """1つのずれ幅について args.trials 回試し、成功回数を返す"""
    # ずれ幅ごとに乱数の種をそろえる（どのずれ幅でも同じ「ずれ方の並び」になる）
    random.seed(args.seed)
    csv_path = f"results_delta{delta:.3f}.csv"
    rows = []
    successes = 0

    print(f"\n===== ずれ幅 ±{delta*100:.1f}cm を {args.trials} 回 =====")
    for t in range(1, args.trials + 1):
        # 1. キューブをランダムにずらして置く
        dx = random.uniform(-delta, delta)
        dy = random.uniform(-delta, delta)
        set_cube_pose(round(BASE_X + dx, 4), round(BASE_Y + dy, 4), BASE_Z)
        time.sleep(1.0)

        # 2. 動作プログラムを実行（終わるまで待つ。万一止まったら60秒で打ち切り）
        try:
            subprocess.run(["python3", args.action],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=60)
        except subprocess.TimeoutExpired:
            print("  [WARN] 動作プログラムが60秒で終わらなかったので打ち切りました")

        # 3. キューブの高さで成功判定
        z = get_cube_z()
        ok = z is not None and z > BASE_Z + LIFT_THRESHOLD
        successes += ok
        print(f"[{t:2d}/{args.trials}] ずれ x {dx*100:+.1f}cm, y {dy*100:+.1f}cm"
              f" -> {'成功' if ok else '失敗'} (cube z = {z})")
        rows.append([t, round(dx, 4), round(dy, 4), z, int(ok)])

        # 4. 持ち上げた状態で静止 → キューブを離して初期状態に戻す
        time.sleep(args.hold)
        resetter.reset_robot()
        time.sleep(3.0)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "dx", "dy", "cube_z", "success"])
        w.writerows(rows)

    print(f"ずれ幅 ±{delta*100:.1f}cm: 成功 {successes}/{args.trials}"
          f" ({100*successes/args.trials:.0f}%)  → {csv_path} に保存")
    return successes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, nargs='+', default=[0.01])
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--action', default=DEFAULT_ACTION)
    parser.add_argument('--hold', type=float, default=2.0)
    args = parser.parse_args()

    if not CMD_BIN:
        print("[ERROR] 'ign' または 'gz' コマンドが見つかりません。")
        return

    rclpy.init()
    resetter = Resetter()

    # 開始前にロボットを初期状態にする（グリッパを開く）
    resetter.reset_robot()
    time.sleep(3.0)

    summary = []
    for delta in args.delta:
        s = run_one_delta(resetter, delta, args)
        summary.append((delta, s))
        # 1つ終わるごとに summary.csv に追記（途中で止まっても結果が残る）
        new_file = not os.path.exists("summary.csv")
        with open("summary.csv", "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["delta_m", "trials", "successes", "success_rate"])
            w.writerow([delta, args.trials, s, round(s / args.trials, 3)])

    # 最後に一覧を表示
    print("\n===== まとめ =====")
    for delta, s in summary:
        print(f"  ±{delta*100:4.1f}cm : {s:2d}/{args.trials}"
              f" ({100*s/args.trials:3.0f}%)")
    print("（summary.csv に追記済み）")

    resetter.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()