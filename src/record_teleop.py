#!/usr/bin/env python3
"""テレオペ中の「カメラ画像」「関節角度」「操作指令」を同時に記録するプログラム。

使い方:
  ターミナルA: Gazebo を起動
  ターミナルB: テレオペを起動
      python3 my_teleop.py
  ターミナルC: このプログラムを起動
      python3 record_teleop.py --outdir /workspace/pai_final/dataset

  1. ターミナルBで X → O を押し、ロボットを初期姿勢・グリッパ全開にする
     （my_teleop.py を起動した直後なら不要）
  2. ターミナルCで Enter → キューブがランダムな位置に置かれる（まだ記録しない）
  3. ターミナルBで R を4回押してプリセット姿勢まで動かす
  4. ターミナルCで Enter → 記録開始
  5. ターミナルBで W/S で微調整し、C で掴む
  6. 1秒待ってターミナルCで Enter → 記録停止
  7. ターミナルBで X を押して持ち上がるか確認し、ターミナルCで y / n を答える
  8. 1 に戻ってくり返す。終了は ターミナルCで q → Enter

  ※ プリセット（R）の動きは毎回同じなので記録しない。
     学習するのは「プリセット姿勢から微調整して掴むまで」。

保存先: dataset/episode_000.npz, episode_001.npz, ...
  images     (T, 120, 160, 3) uint8   カメラ画像（640x480 を 1/4 に縮小）
  joint_pos  (T, 5) float32           実際の関節角度 [j1, j2, j3, j4, hand]
  action     (T, 5) float32           その時点のテレオペの指令値 [j1, j2, j3, j4, hand]
  time       (T,)   float64           記録開始からの秒数
  cube_offset (2,)  float32           キューブの元の位置からのずれ [dx, dy] (m)
"""
import argparse
import os
import random
import shutil
import subprocess
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory

# 関節の並び順（保存するデータはすべてこの順にそろえる）
JOINTS = ['crane_plus_joint1', 'crane_plus_joint2',
          'crane_plus_joint3', 'crane_plus_joint4', 'crane_plus_joint_hand']

# キューブの元の位置（evaluate_replay.py と同じ）
BASE_X, BASE_Y, BASE_Z = 0.0, -0.1, 1.0035
CUBE_NAME = "wood_cube_5cm"

RATE_HZ = 10.0      # 記録の頻度（1秒に10回）
SHRINK = 4          # 画像の縮小率（640x480 → 160x120）

CMD_BIN = shutil.which("ign") or shutil.which("gz")
MSG_PREFIX = "ignition.msgs" if CMD_BIN and "ign" in CMD_BIN else "gz.msgs"


def set_cube_pose(x, y, z):
    req = (f'name: "{CUBE_NAME}", position: {{x: {x}, y: {y}, z: {z}}}, '
           f'orientation: {{x: 0, y: 0, z: 0, w: 1}}')
    res = subprocess.run(
        [CMD_BIN, "service", "-s", "/world/default/set_pose",
         "--reqtype", f"{MSG_PREFIX}.Pose", "--reptype", f"{MSG_PREFIX}.Boolean",
         "--timeout", "1000", "--req", req],
        capture_output=True, text=True)
    return "true" in res.stdout.lower()


class Recorder(Node):
    def __init__(self):
        super().__init__('teleop_recorder')
        self.lock = threading.Lock()
        self.latest_image = None                 # 最新のカメラ画像 (120,160,3)
        self.joint_pos = {}                      # 関節名 → 実際の角度
        self.action = {j: 0.0 for j in JOINTS}   # 関節名 → テレオペの指令値
        self.action['crane_plus_joint_hand'] = -0.5236
        self.recording = False
        self.buf = None

        # カメラ画像は sensor_data の QoS で受け取る（備忘録の注意点）
        self.create_subscription(Image, '/camera/image_raw', self.on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(JointState, '/joint_states', self.on_joint, 10)
        # テレオペが出している指令を横から受け取る（名前で対応させるので並び順は気にしない）
        self.create_subscription(JointTrajectory,
                                 '/crane_plus_arm_controller/joint_trajectory',
                                 self.on_command, 10)
        self.create_subscription(JointTrajectory,
                                 '/crane_plus_gripper_controller/joint_trajectory',
                                 self.on_command, 10)
        self.create_timer(1.0 / RATE_HZ, self.on_timer)

    def on_image(self, msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        if msg.encoding.lower().startswith('bgr'):
            arr = arr[:, :, ::-1]
        small = arr[::SHRINK, ::SHRINK].copy()
        with self.lock:
            self.latest_image = small

    def on_joint(self, msg):
        with self.lock:
            for n, p in zip(msg.name, msg.position):
                self.joint_pos[n] = p

    def on_command(self, msg):
        if not msg.points:
            return
        with self.lock:
            for n, p in zip(msg.joint_names, msg.points[0].positions):
                if n in self.action:
                    self.action[n] = p

    def ready(self):
        with self.lock:
            return (self.latest_image is not None
                    and all(j in self.joint_pos for j in JOINTS))

    def start(self):
        with self.lock:
            self.buf = {'images': [], 'joint_pos': [], 'action': [], 'time': []}
            self.t0 = time.time()
            self.recording = True

    def stop(self):
        with self.lock:
            self.recording = False
            return self.buf

    def on_timer(self):
        with self.lock:
            if not self.recording or self.latest_image is None:
                return
            if not all(j in self.joint_pos for j in JOINTS):
                return
            self.buf['images'].append(self.latest_image)
            self.buf['joint_pos'].append([self.joint_pos[j] for j in JOINTS])
            self.buf['action'].append([self.action[j] for j in JOINTS])
            self.buf['time'].append(time.time() - self.t0)


def next_episode_path(outdir):
    os.makedirs(outdir, exist_ok=True)
    i = 0
    while os.path.exists(os.path.join(outdir, f"episode_{i:03d}.npz")):
        i += 1
    return os.path.join(outdir, f"episode_{i:03d}.npz")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delta', type=float, default=0.05,
                        help='キューブのずれ幅 [m]（0 でずらさない）')
    parser.add_argument('--outdir', default='dataset')
    args = parser.parse_args()

    if not CMD_BIN:
        print("[ERROR] 'ign' または 'gz' コマンドが見つかりません。")
        return

    rclpy.init()
    node = Recorder()
    # ROS の受信処理は裏のスレッドで回し、表ではキーボード入力を待つ
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("カメラ画像と関節角度の受信を待っています...")
    for _ in range(100):
        if node.ready():
            break
        time.sleep(0.1)
    if not node.ready():
        print("[ERROR] 画像または関節角度が届きません。Gazebo は起動していますか？")
        rclpy.shutdown()
        return
    print("受信OK。")

    # 学習用のずれ方は毎回ちがう乱数にする
    # （評価用の evaluate_replay.py は seed=0 なので、それとは別の位置になる）
    rng = random.Random()

    while True:
        cmd = input("\n[準備] テレオペで X → O を押して初期姿勢にしたら Enter（終了は q）: ")
        if cmd.strip().lower() == 'q':
            break

        dx = rng.uniform(-args.delta, args.delta)
        dy = rng.uniform(-args.delta, args.delta)
        set_cube_pose(round(BASE_X + dx, 4), round(BASE_Y + dy, 4), BASE_Z)
        time.sleep(1.0)  # キューブが落ち着くのを待つ
        print(f"キューブを置きました（ずれ x {dx*100:+.1f}cm, y {dy*100:+.1f}cm）")

        input("[プリセット] テレオペで R を4回押し、動き終わったら Enter で記録開始: ")
        node.start()
        input("[記録中] テレオペで C で掴み、1秒待ったら Enter: ")
        buf = node.stop()
        n = len(buf['images'])
        print(f"記録したフレーム数: {n}（約 {n / RATE_HZ:.1f} 秒）")

        ans = input("（テレオペで X を押して持ち上がるか確認してから）保存しますか？ (y/n): ")
        if ans.strip().lower() == 'y' and n > 0:
            path = next_episode_path(args.outdir)
            np.savez_compressed(
                path,
                images=np.array(buf['images'], dtype=np.uint8),
                joint_pos=np.array(buf['joint_pos'], dtype=np.float32),
                action=np.array(buf['action'], dtype=np.float32),
                time=np.array(buf['time'], dtype=np.float64),
                cube_offset=np.array([dx, dy], dtype=np.float32),
            )
            print(f"保存しました → {path}")
        else:
            print("保存しませんでした。")

    rclpy.shutdown()


if __name__ == '__main__':
    main()