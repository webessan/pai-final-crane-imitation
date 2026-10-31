#!/usr/bin/env python3
"""学習したモデルでロボットを動かし、成功率を測るプログラム。

各試行の流れ:
  1. ロボットを初期姿勢・グリッパ全開に戻す
  2. キューブをランダムにずらして置く（evaluate_replay.py と同じ乱数 → 同じ位置）
  3. 定位置（キューブの元の位置の真上 9cm）までプログラムで動かす（auto_demo.py と同じ）
  4. ここからモデルが操作: カメラ画像と関節角度を見て、0.1秒ごとに関節を動かす
     モデルが「閉じる」と判断したらグリッパを閉じる
  5. プログラムで持ち上げ（初期姿勢へ）、キューブの高さで成功判定
  6. 持ち上げたまま少し静止 → 離す

使い方（Gazebo を起動した状態で。テレオペは止めておくこと）:
    python3 run_policy.py --model /workspace/pai_final/models/bc_v2.pt --delta 0.05 --trials 10

    --delta     キューブのずれ幅 [m]（スペース区切りで複数指定可）
    --trials    試行回数（既定 10）
    --seed      乱数の種（既定 0 = evaluate_replay.py と同じ位置の並び）
    --max_time  モデルが操作する最大時間 [秒]（既定 40。超えたら失敗）
    --verbose   モデル操作中、1秒ごとに「動かす量」と「閉じる確率」を表示
結果: 画面表示と policy_results_delta<ずれ幅>.csv、policy_summary.csv
"""
import argparse
import csv
import os
import random
import re
import shutil
import subprocess
import threading
import time

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from bc_model import BCPolicy
from crane_ik import PRESET_Q, preset_steps

ARM = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint3', 'crane_plus_joint4']
HAND = 'crane_plus_joint_hand'
JOINTS = ARM + [HAND]
LIMITS = [(-2.618, 2.618), (-1.824, 1.783), (-2.556, 2.454), (-1.840, 1.768)]
GRIPPER_OPEN, GRIPPER_CLOSE = -0.65, 0.35   # auto_demo.py と同じ開き方

BASE_X, BASE_Y, BASE_Z = 0.0, -0.1, 1.0035
CUBE_NAME = "wood_cube_5cm"
LIFT_THRESHOLD = 0.30
RATE_HZ = 10.0
SHRINK = 4
CLOSE_CONFIRM = 3      # 「閉じる」が何回続いたら本当に閉じるか

CMD_BIN = shutil.which("ign") or shutil.which("gz")
MSG_PREFIX = "ignition.msgs" if CMD_BIN and "ign" in CMD_BIN else "gz.msgs"


# ---------- Gazebo の操作（evaluate_replay.py と同じ） ----------
def set_cube_pose(x, y, z):
    req = (f'name: "{CUBE_NAME}", position: {{x: {x}, y: {y}, z: {z}}}, '
           f'orientation: {{x: 0, y: 0, z: 0, w: 1}}')
    res = subprocess.run(
        [CMD_BIN, "service", "-s", "/world/default/set_pose",
         "--reqtype", f"{MSG_PREFIX}.Pose", "--reptype", f"{MSG_PREFIX}.Boolean",
         "--timeout", "1000", "--req", req], capture_output=True, text=True)
    return "true" in res.stdout.lower()


def get_cube_z():
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


# ---------- ROS ノード ----------
class PolicyNode(Node):
    def __init__(self):
        super().__init__('bc_policy_runner')
        self.lock = threading.Lock()
        self.image = None
        self.joint = {}
        self.create_subscription(Image, '/camera/image_raw', self.on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self.on_joint, 10)
        self.arm_pub = self.create_publisher(
            JointTrajectory, 'crane_plus_arm_controller/joint_trajectory', 10)
        self.grip_pub = self.create_publisher(
            JointTrajectory, 'crane_plus_gripper_controller/joint_trajectory', 10)

    def on_image(self, msg):
        # record_teleop.py と同じ前処理（RGB にそろえて 1/4 に縮小）
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding.lower().startswith('bgr'):
            arr = arr[:, :, ::-1]
        with self.lock:
            self.image = arr[::SHRINK, ::SHRINK].copy()

    def on_joint(self, msg):
        with self.lock:
            for n, p in zip(msg.name, msg.position):
                self.joint[n] = p

    def observe(self):
        with self.lock:
            if self.image is None or not all(j in self.joint for j in JOINTS):
                return None, None
            return self.image.copy(), np.array([self.joint[j] for j in JOINTS], np.float32)

    def _traj(self, names, pos, sec):
        msg = JointTrajectory()
        msg.joint_names = names
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in pos]
        pt.time_from_start = Duration(sec=int(sec), nanosec=int((sec % 1) * 1e9))
        msg.points.append(pt)
        return msg

    def move_arm(self, pos, sec):
        self.arm_pub.publish(self._traj(ARM, pos, sec))

    def move_gripper(self, pos, sec=1.0):
        self.grip_pub.publish(self._traj([HAND], [pos], sec))


# ---------- 1回の試行 ----------
def run_trial(node, model, ck, dx, dy, args):
    # 1. 初期姿勢・グリッパ全開
    node.move_gripper(GRIPPER_OPEN)
    node.move_arm([0, 0, 0, 0], 2.0)
    time.sleep(3.0)

    # 2. キューブを置く
    set_cube_pose(round(BASE_X + dx, 4), round(BASE_Y + dy, 4), BASE_Z)
    time.sleep(1.0)

    # 3. 定位置（キューブの元の位置の真上 9cm）へ（auto_demo.py と同じ動かし方）
    pose = list(PRESET_Q)
    for q in preset_steps():
        node.move_arm(q, 2.5)
        time.sleep(3.0)
    # 実物が定位置に追いつくまで待つ（最大5秒）
    # （デモでも、定位置に着いてから記録を始めていたので、それに合わせる）
    t_wait = time.time()
    while time.time() - t_wait < 5.0:
        _, jp = node.observe()
        if jp is not None and max(abs(jp[k] - pose[k]) for k in range(4)) < 0.02:
            break
        time.sleep(0.1)

    # 4. モデルが操作
    j_mean = torch.tensor(ck['j_mean']); j_std = torch.tensor(ck['j_std'])
    d_std = np.asarray(ck['d_std'])
    # command 方式: テレオペと同じく「少し先の指令」を短い時間で出し続ける
    # future 方式（v1）: 0.5秒後の角度を目標に出す
    if ck.get('target', 'future') == 'command':
        move_sec = 0.15
    else:
        move_sec = ck['horizon'] / RATE_HZ
    step = 0
    close_count = 0
    closed = False
    t_start = time.time()
    while time.time() - t_start < args.max_time:
        t0 = time.time()
        img, jp = node.observe()
        if img is None:
            time.sleep(0.05); continue
        with torch.no_grad():
            x_img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            x_j = ((torch.from_numpy(jp) - j_mean) / j_std).float().unsqueeze(0)
            pd, pg = model(x_img, x_j)
        delta = pd[0].numpy() * d_std
        p_close = torch.sigmoid(pg[0]).item()

        target = [min(max(jp[i] + delta[i], LIMITS[i][0]), LIMITS[i][1]) for i in range(4)]
        node.move_arm(target, move_sec)

        if args.verbose and step % 10 == 0:   # 1秒ごとにモデルの判断を表示
            print(f"    {time.time() - t_start:4.1f}秒  動かす量 "
                  + " ".join(f"j{k + 1}:{delta[k]:+.3f}" for k in range(4))
                  + f"  閉じる確率 {p_close:.2f}")
        step += 1

        close_count = close_count + 1 if p_close > 0.5 else 0
        if close_count >= CLOSE_CONFIRM:
            closed = True
            break
        time.sleep(max(0.0, 1.0 / RATE_HZ - (time.time() - t0)))
    used = time.time() - t_start

    # 5. 閉じて持ち上げ → 判定
    if closed:
        node.move_arm([float(v) for v in node.observe()[1][:4]], 0.5)  # その場で止める
        node.move_gripper(GRIPPER_CLOSE)
        time.sleep(1.5)
        node.move_arm([0, 0, 0, 0], 2.0)
        time.sleep(3.0)
    z = get_cube_z()
    ok = closed and z is not None and z > BASE_Z + LIFT_THRESHOLD

    # 6. 静止 → 離す
    time.sleep(args.hold)
    node.move_gripper(GRIPPER_OPEN)
    node.move_arm([0, 0, 0, 0], 2.0)
    time.sleep(2.0)
    return ok, closed, used, z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--delta', type=float, nargs='+', default=[0.05])
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--max_time', type=float, default=40.0)
    ap.add_argument('--hold', type=float, default=2.0)
    ap.add_argument('--verbose', action='store_true',
                    help='モデル操作中、1秒ごとに判断の中身を表示する')
    args = ap.parse_args()

    ck = torch.load(args.model, map_location='cpu', weights_only=False)
    model = BCPolicy()
    model.load_state_dict(ck['model'])
    model.eval()
    print(f"モデル: {args.model}（学習に使ったエピソード {len(ck['train_episodes'])}本）")

    rclpy.init()
    node = PolicyNode()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    for _ in range(100):
        if node.observe()[0] is not None:
            break
        time.sleep(0.1)
    if node.observe()[0] is None:
        print("[ERROR] 画像または関節角度が届きません"); rclpy.shutdown(); return

    name = os.path.splitext(os.path.basename(args.model))[0]
    summary = []
    for delta in args.delta:
        random.seed(args.seed)   # evaluate_replay.py と同じ位置の並びにする
        rows, succ = [], 0
        print(f"\n===== ずれ幅 ±{delta*100:.1f}cm を {args.trials} 回（モデル: {name}） =====")
        for t in range(1, args.trials + 1):
            dx = random.uniform(-delta, delta)
            dy = random.uniform(-delta, delta)
            ok, closed, used, z = run_trial(node, model, ck, dx, dy, args)
            succ += ok
            state = '成功' if ok else ('失敗（掴めず）' if closed else '失敗（閉じる判断なし）')
            print(f"[{t:2d}/{args.trials}] ずれ x {dx*100:+.1f}cm, y {dy*100:+.1f}cm"
                  f" -> {state}（モデル操作 {used:.1f}秒, cube z = {z}）")
            rows.append([t, round(dx, 4), round(dy, 4), z, int(closed), round(used, 1), int(ok)])
        path = f"policy_results_{name}_delta{delta:.3f}.csv"
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['trial', 'dx', 'dy', 'cube_z', 'closed', 'policy_sec', 'success'])
            w.writerows(rows)
        print(f"ずれ幅 ±{delta*100:.1f}cm: 成功 {succ}/{args.trials}  → {path}")
        summary.append((delta, succ))
        new = not os.path.exists("policy_summary.csv")
        with open("policy_summary.csv", "a", newline='') as f:
            w = csv.writer(f)
            if new:
                w.writerow(['model', 'delta_m', 'trials', 'successes', 'success_rate'])
            w.writerow([name, delta, args.trials, succ, round(succ / args.trials, 3)])

    print("\n===== まとめ =====")
    for delta, s in summary:
        print(f"  ±{delta*100:4.1f}cm : {s:2d}/{args.trials} ({100*s/args.trials:3.0f}%)")
    rclpy.shutdown()


if __name__ == '__main__':
    main()
