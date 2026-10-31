#!/usr/bin/env python3
"""逆運動学を使って、キューブを掴むデモを自動で作って記録するプログラム。

各回の流れ:
  1. 初期姿勢・グリッパ全開に戻し、キューブをランダムな位置に置く
  2. 定位置（キューブの元の位置の真上、高さ 9cm）まで動かす（ここは記録しない）
  3. ここから記録開始。手先を
       (a) 同じ高さのまま、キューブの真上へ水平移動
           （届かない位置のときは、先に真下へ少し下げてから水平移動）
       (b) 真下へ下ろす
     という道すじで動かす（上がる動きはしない）。各点は逆運動学で関節角度に変換する
  4. グリッパを閉じ、1秒待って記録終了
  5. 持ち上げて成功判定。成功した回だけ保存する（record_teleop.py と同じ形式）

使い方（Gazebo を起動した状態で。テレオペや run_policy.py は止めておくこと）:
    python3 auto_demo.py --n 100 --outdir /workspace/pai_final/dataset_auto

    --n         集める成功デモの数（既定 100）
    --delta     キューブのずれ幅 [m]（既定 0.05）
    --seed      乱数の種（既定 1000。評価用の seed=0 とは別の位置になる）
    --speed     手先を動かす速さ [m/秒]（既定 0.04）
    --test N    保存せずに N 回だけ試して、成功率を表示する
"""
import argparse
import math
import os
import random
import re
import shutil
import subprocess
import threading
import time

import numpy as np
import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from crane_ik import (fk, ik, PRESET_X, PRESET_Y, PRESET_Z, PRESET_PHI,
                      PRESET_Q, GRASP_Z, preset_steps)

ARM = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint3', 'crane_plus_joint4']
HAND = 'crane_plus_joint_hand'
JOINTS = ARM + [HAND]
GRIPPER_OPEN, GRIPPER_CLOSE = -0.65, 0.35   # 開くときは可動範囲(-0.709)近くまで大きく開く

BASE_X, BASE_Y, BASE_Z = 0.0, -0.1, 1.0035   # キューブの元の位置（Gazebo の座標）
ROBOT_Z = 1.015                              # ロボットの土台の高さ（Gazebo の座標）
CUBE_NAME = "wood_cube_5cm"
HOVER_MIN = 0.05         # 水平移動するときの、掴む点の高さの下限（これより低いとキューブに当たる）
PHI_CANDIDATES = (3.0, 2.9, 2.8, 2.7, 2.6, 2.5)   # 手先の角度の候補（3.0 がほぼ真下）
LIFT_THRESHOLD = 0.30
RATE_HZ = 10.0
SHRINK = 4

CMD_BIN = shutil.which("ign") or shutil.which("gz")
MSG_PREFIX = "ignition.msgs" if CMD_BIN and "ign" in CMD_BIN else "gz.msgs"


def set_cube_pose(x, y, z):
    req = (f'name: "{CUBE_NAME}", position: {{x: {x}, y: {y}, z: {z}}}, '
           f'orientation: {{x: 0, y: 0, z: 0, w: 1}}')
    res = subprocess.run(
        [CMD_BIN, "service", "-s", "/world/default/set_pose",
         "--reqtype", f"{MSG_PREFIX}.Pose", "--reptype", f"{MSG_PREFIX}.Boolean",
         "--timeout", "1000", "--req", req], capture_output=True, text=True)
    return "true" in res.stdout.lower()


def get_cube_pose():
    """キューブの現在位置 (x, y, z)（Gazebo の座標）。読めなければ None"""
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
    m = re.search(r'position\s*\{([^}]*)\}', text[i:])
    if not m:
        return None
    vals = {}
    for k in ('x', 'y', 'z'):
        mm = re.search(rf'\b{k}:\s*([-0-9.eE+]+)', m.group(1))
        vals[k] = float(mm.group(1)) if mm else 0.0   # 0 のときは省略されることがある
    return vals['x'], vals['y'], vals['z']


class DemoNode(Node):
    def __init__(self):
        super().__init__('auto_demo')
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
        # 記録用
        self.recording = False
        self.buf = None
        self.cmd = [0.0, 0.0, 0.0, 0.0, GRIPPER_OPEN]   # いま出している指令
        self.create_timer(1.0 / RATE_HZ, self.on_timer)

    def on_image(self, msg):
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

    def on_timer(self):
        # record_teleop.py と同じく、0.1秒ごとに 画像・関節角度・指令 を記録
        if not self.recording:
            return
        img, jp = self.observe()
        if img is None:
            return
        with self.lock:
            self.buf['images'].append(img)
            self.buf['joint_pos'].append(jp)
            self.buf['action'].append(list(self.cmd))
            self.buf['time'].append(time.time() - self.t0)

    def start_rec(self):
        with self.lock:
            self.buf = {'images': [], 'joint_pos': [], 'action': [], 'time': []}
            self.t0 = time.time()
            self.recording = True

    def stop_rec(self):
        with self.lock:
            self.recording = False
            return self.buf

    def _traj(self, names, pos, sec):
        msg = JointTrajectory()
        msg.joint_names = names
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in pos]
        pt.time_from_start = Duration(sec=int(sec), nanosec=int((sec % 1) * 1e9))
        msg.points.append(pt)
        return msg

    def move_arm(self, q, sec):
        with self.lock:
            self.cmd[:4] = [float(v) for v in q]
            grip = self.cmd[4]
        self.arm_pub.publish(self._traj(ARM, q, sec))
        # グリッパの指令も毎回出し直す（最初の1回が取りこぼされても開くように）
        self.grip_pub.publish(self._traj([HAND], [grip], 0.3))

    def set_cmd(self, q):
        """記録する指令だけを更新する（ロボットには送らない）"""
        with self.lock:
            self.cmd[:4] = [float(v) for v in q]

    def send_path(self, qs, dt):
        """関節角度の列を、dt 秒おきの時刻付きの道すじとして1回で送る"""
        msg = JointTrajectory()
        msg.joint_names = ARM
        for k, q in enumerate(qs, start=1):
            pt = JointTrajectoryPoint()
            pt.positions = [float(v) for v in q]
            t = k * dt
            pt.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
            msg.points.append(pt)
        with self.lock:
            grip = self.cmd[4]
        self.arm_pub.publish(msg)
        self.grip_pub.publish(self._traj([HAND], [grip], 0.3))

    def move_gripper(self, pos, sec=1.0):
        with self.lock:
            self.cmd[4] = float(pos)
        self.grip_pub.publish(self._traj([HAND], [pos], sec))


def line_ok(a, b, n=20):
    """手先を a から b へ直線で動かすとき、途中の点もすべて逆運動学で解けるか"""
    return all(ik(*[s + (g - s) * k / n for s, g in zip(a, b)]) is not None
               for k in range(n + 1))


def plan_route(cx, cy):
    """定位置 (PRESET) から、キューブ (cx, cy) を掴むまでの道すじを決める。
    動きは「下がる」と「水平移動」だけ（上がる動きはしない）:
      定位置 → (必要なら) 真下に h まで下げる → 高さ h のまま水平移動 → 真下に下ろして掴む
    h はできるだけ高く（定位置の高さのままが理想）。
    掴むときの手先の角度 φg は、なるべく真下（3.0）に近いもの。
    戻り値: (h, φg)、どうしても届かないときは None"""
    p0 = (PRESET_X, PRESET_Y, PRESET_Z, PRESET_PHI)
    h = PRESET_Z
    while h >= HOVER_MIN - 1e-9:
        lowered = (PRESET_X, PRESET_Y, h, PRESET_PHI)
        above = (cx, cy, h, PRESET_PHI)
        if line_ok(p0, lowered) and line_ok(lowered, above):
            for phi_g in PHI_CANDIDATES:
                if (ik(cx, cy, GRASP_Z, phi_g) is not None
                        and line_ok(above, (cx, cy, GRASP_Z, phi_g))):
                    return h, phi_g
        h -= 0.005
    return None


def move_cartesian(node, start, goal, speed):
    """手先を start から goal へ直線的に動かす。
    道すじ上の各点（0.1秒ごと）を逆運動学で関節角度にし、
    時刻付きの道すじとして「まとめて1回」ロボットに送る。
    制御器が道すじ全体を知っているので、1点ずつ送るより遅れが小さく、
    途中で下がりすぎたりしにくい。最後に、実物が goal に追いつくまで待つ。
    start, goal = (x, y, z, phi)"""
    dist = math.dist(start[:3], goal[:3]) + 0.02 * abs(goal[3] - start[3])
    n = max(3, int(dist / speed * RATE_HZ))
    qs = []
    for k in range(1, n + 1):
        a = k / n
        q = ik(*[s + (g - s) * a for s, g in zip(start, goal)])
        if q is None:
            return False
        qs.append(q)
    dt = 1.0 / RATE_HZ
    node.send_path(qs, dt)
    for q in qs:               # 記録用に「いま向かっている点」を指令として残す
        node.set_cmd(q)
        time.sleep(dt)
    wait_settle(node, qs[-1], timeout=3.0)
    return True


def wait_settle(node, q, timeout=3.0, tol=0.02):
    t = time.time()
    while time.time() - t < timeout:
        _, jp = node.observe()
        if jp is not None and max(abs(jp[k] - q[k]) for k in range(4)) < tol:
            return True
        node.move_arm(q, 0.15)   # 記録中も指令を出し続ける
        time.sleep(1.0 / RATE_HZ)
    return False


def run_episode(node, dx, dy, args):
    """1回分。戻り値: (成功したか, 記録データ, 理由)"""
    # 1. 初期姿勢・グリッパ全開 → キューブを置く
    node.move_gripper(GRIPPER_OPEN)
    node.move_arm([0, 0, 0, 0], 2.0)
    time.sleep(3.0)
    set_cube_pose(round(BASE_X + dx, 4), round(BASE_Y + dy, 4), BASE_Z)
    time.sleep(1.0)

    # 2. 定位置（キューブの元の位置の真上 9cm）へ（記録しない）
    #    j1 を回してから、残りの関節をまとめて動かす（途中で 9cm より下がらない）
    for q in preset_steps():
        node.move_arm(q, 2.5)
        time.sleep(3.0)
    wait_settle(node, PRESET_Q, timeout=5.0)

    # キューブの実際の位置（ロボットの土台から見た座標）
    cube = get_cube_pose()
    if cube is None:
        return False, None, 'キューブの位置が読めない'
    cx, cy = cube[0], cube[1]
    route = plan_route(cx, cy)
    if route is None:
        return False, None, '逆運動学で届かない位置'
    h, phi_g = route

    # 3. 記録開始 →（必要なら少し下げる）→ 水平移動 → 真下に下ろす（上がる動きはしない）
    node.start_rec()
    start = (PRESET_X, PRESET_Y, PRESET_Z, PRESET_PHI)
    lowered = (PRESET_X, PRESET_Y, h, PRESET_PHI)
    above = (cx, cy, h, PRESET_PHI)
    down = (cx, cy, GRASP_Z, phi_g)
    ok = True
    if h < PRESET_Z - 1e-9:
        ok = move_cartesian(node, start, lowered, args.speed)
    ok = (ok and move_cartesian(node, lowered, above, args.speed)
          and move_cartesian(node, above, down, args.speed))
    if not ok:
        node.stop_rec()
        return False, None, '途中で逆運動学が解けない'

    # 4. 閉じて1秒待つ → 記録終了
    node.move_gripper(GRIPPER_CLOSE)
    time.sleep(1.0)
    buf = node.stop_rec()

    # 5. 持ち上げて判定 → 離す
    time.sleep(0.5)
    node.move_arm([0, 0, 0, 0], 2.0)
    time.sleep(3.0)
    pos = get_cube_pose()
    lifted = pos is not None and pos[2] > BASE_Z + LIFT_THRESHOLD
    node.move_gripper(GRIPPER_OPEN)
    time.sleep(1.0)
    return lifted, buf, ('成功' if lifted else '持ち上がらなかった')


def next_episode_path(outdir):
    os.makedirs(outdir, exist_ok=True)
    i = 0
    while os.path.exists(os.path.join(outdir, f"episode_{i:03d}.npz")):
        i += 1
    return os.path.join(outdir, f"episode_{i:03d}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--delta', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=1000)
    ap.add_argument('--speed', type=float, default=0.04)
    ap.add_argument('--outdir', default='/workspace/pai_final/dataset_auto')
    ap.add_argument('--test', type=int, default=0)
    args = ap.parse_args()

    if not CMD_BIN:
        print("[ERROR] 'ign' または 'gz' コマンドが見つかりません。"); return

    rclpy.init()
    node = DemoNode()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    for _ in range(100):
        if node.observe()[0] is not None:
            break
        time.sleep(0.1)
    if node.observe()[0] is None:
        print("[ERROR] 画像または関節角度が届きません"); rclpy.shutdown(); return

    rng = random.Random(args.seed)
    goal = args.test if args.test > 0 else args.n
    saved = tries = 0
    while (saved if args.test == 0 else tries) < goal:
        tries += 1
        dx = rng.uniform(-args.delta, args.delta)
        dy = rng.uniform(-args.delta, args.delta)
        ok, buf, why = run_episode(node, dx, dy, args)
        n = len(buf['images']) if buf else 0
        msg = f"[{tries:3d}] ずれ x {dx*100:+.1f}cm, y {dy*100:+.1f}cm -> {why}（{n}フレーム）"
        if ok and args.test == 0:
            path = next_episode_path(args.outdir)
            np.savez_compressed(
                path,
                images=np.array(buf['images'], dtype=np.uint8),
                joint_pos=np.array(buf['joint_pos'], dtype=np.float32),
                action=np.array(buf['action'], dtype=np.float32),
                time=np.array(buf['time'], dtype=np.float64),
                cube_offset=np.array([dx, dy], dtype=np.float32),
            )
            saved += 1
            msg += f" → 保存 {path}（{saved}/{args.n}）"
        elif ok:
            saved += 1
        print(msg, flush=True)

    print(f"\n試行 {tries} 回中 成功 {saved} 回（{100 * saved / tries:.0f}%）")
    # 後片付け（受信の処理を止めてから終了する）
    executor.shutdown()
    spin_thread.join(timeout=2.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
