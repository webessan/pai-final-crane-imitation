#!/usr/bin/env python3
"""クレーンゲーム型テレオペ: グリッパを下向きに保ったまま、手先を前後・左右・上下に動かす。

逆運動学（crane_ik.py）で、手先の位置を関節角度に変換して指令する。
記録プログラム（record_teleop.py）とそのまま組み合わせられる（記録されるのは関節角度）。

使い方:
  python3 crane_teleop.py

キー操作（Gazebo で「左にカメラ、右にロボット」が見える視点のとき）:
  W / S : 手先を 奥 / 手前 へ
  A / D : 手先を 左（ロボットから遠ざかる）/ 右（ロボットに近づく）へ
  R / F : 手先を 上 / 下 へ
  Z     : 1回の移動量を切り替え（5mm ⇔ 1cm）
  P     : 定位置（キューブの元の位置の真上 9cm）へ自動で移動。グリッパも開く
  O / C : グリッパを 開く / 閉じる
  X     : 初期姿勢（真上）に戻し、グリッパを閉じる（掴んでいれば持ち上がる）
  ESC   : 終了

画面の表示:
  手先 x, y : キューブの「元の位置」から見た手先のずれ [cm]
              （record_teleop.py が表示する「キューブのずれ」と同じ向き。
                この2つの数字をそろえると、手先がキューブの真上に来る）
  高さ      : 掴む高さから見た手先の高さ [cm]（0 で掴む高さ）
"""
import math
import select
import sys
import termios
import threading
import tty

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from crane_ik import fk, ik, preset_steps

ARM = ['crane_plus_joint1', 'crane_plus_joint2', 'crane_plus_joint3', 'crane_plus_joint4']
HAND = 'crane_plus_joint_hand'
GRIPPER_OPEN, GRIPPER_CLOSE = -0.5236, 0.35

CUBE_X, CUBE_Y = 0.0, -0.10   # キューブの元の位置（土台から見た座標）
GRASP_Z = 0.013               # 掴む高さ
PHI_PREF = 3.0                # 手先の角度（ほぼ真下）。届かないときは近い角度で探す
PHI_TRY = (3.0, 2.9, 3.1, 2.8, 2.7, 2.6)

JOINT_STEP = 0.05   # 1回(0.05秒)に指令を進める最大量 [rad]
MAX_LEAD = 0.3      # 指令が実際の角度より先に行ける最大量 [rad]


class CraneTeleop(Node):
    def __init__(self):
        super().__init__('crane_teleop')
        self.arm_pub = self.create_publisher(
            JointTrajectory, '/crane_plus_arm_controller/joint_trajectory', 10)
        self.grip_pub = self.create_publisher(
            JointTrajectory, '/crane_plus_gripper_controller/joint_trajectory', 10)
        self.create_subscription(JointState, '/joint_states', self.on_joint, 10)

        self.lock = threading.Lock()
        self.actual = None                 # 実際の関節角度
        self.target = [0.0, 0.0, 0.0, 0.0]  # いま出している指令
        self.goal = [0.0, 0.0, 0.0, 0.0]    # 行き先（指令はここへ少しずつ近づく）
        self.gripper = GRIPPER_OPEN
        self.tip = None                    # 手先の目標 (x, y, z, phi)
        self.step = 0.005
        self.sequence = []                 # P のときの「1関節ずつ」の予定
        self.msg = ''
        self.running = True

        print(__doc__.split('キー操作')[1].split('画面の表示')[0])
        threading.Thread(target=self.keyboard_loop, daemon=True).start()
        self.create_timer(0.05, self.on_timer)   # 20Hz

    # ---------- 受信 ----------
    def on_joint(self, msg):
        pos = dict(zip(msg.name, msg.position))
        if all(j in pos for j in ARM):
            with self.lock:
                first = self.actual is None
                self.actual = [pos[j] for j in ARM]
                if first:   # 起動直後は、今の姿勢から始める
                    self.target = self.actual[:]
                    self.goal = self.actual[:]

    # ---------- 手先を動かす ----------
    def sync_tip(self):
        """今の行き先（関節角度）から、手先の目標を計算し直す"""
        x, y, z, phi = fk(self.goal)
        self.tip = (x, y, z, phi)

    def move_tip(self, dx=0.0, dy=0.0, dz=0.0):
        with self.lock:
            if self.sequence:
                self.msg = 'プリセットの動作中です'
                return
            if self.tip is None:
                self.sync_tip()
            x, y, z, phi = self.tip
            if phi < 2.0:   # 腕が上を向いている（初期姿勢など）ときは手先操作できない
                self.tip = None
                self.msg = '先に P でプリセット姿勢にしてください'
                return
            nx, ny, nz = x + dx, y + dy, z + dz
            # 今の角度に近い順に試して、解ける手先の角度を探す
            for p in sorted(PHI_TRY, key=lambda v: abs(v - phi)):
                q = ik(nx, ny, nz, p)
                if q is not None:
                    self.tip = (nx, ny, nz, p)
                    self.goal = q
                    self.msg = ''
                    return
            self.msg = 'そこには届きません'

    # ---------- 指令を出す（20Hz） ----------
    def on_timer(self):
        with self.lock:
            if self.actual is None:
                return
            # P の定位置への移動: 経由点に着いたら次へ
            if self.sequence:
                self.goal = list(self.sequence[0])
                if max(abs(self.actual[k] - self.goal[k]) for k in range(4)) < 0.03:
                    self.sequence.pop(0)
                    if not self.sequence:
                        self.sync_tip()
                        self.msg = 'プリセット完了'
            # 指令を行き先へ少しずつ近づける（実物より MAX_LEAD 以上は先に行かない）
            for k in range(4):
                d = self.goal[k] - self.target[k]
                new = self.target[k] + max(-JOINT_STEP, min(JOINT_STEP, d))
                lo, hi = self.actual[k] - MAX_LEAD, self.actual[k] + MAX_LEAD
                self.target[k] = min(max(new, lo), hi)
            target = self.target[:]
            grip = self.gripper
        self.arm_pub.publish(self._msg(ARM, target))
        self.grip_pub.publish(self._msg([HAND], [grip]))

    def _msg(self, names, pos):
        jt = JointTrajectory()
        jt.joint_names = names
        pt = JointTrajectoryPoint()
        pt.positions = [float(p) for p in pos]
        pt.time_from_start.nanosec = int(0.1 * 1e9)
        jt.points.append(pt)
        return jt

    # ---------- 表示 ----------
    def show(self):
        with self.lock:
            g = 'OPEN ' if self.gripper == GRIPPER_OPEN else 'CLOSE'
            if self.tip is not None and not self.sequence:
                x, y, z, _ = self.tip
                pos = (f"手先 x {(x - CUBE_X) * 100:+5.1f}cm  y {(y - CUBE_Y) * 100:+5.1f}cm  "
                       f"高さ {(z - GRASP_Z) * 100:+5.1f}cm")
            else:
                pos = "手先 ---（P または X の動作中）"
            line = f"\r{pos}   hand:{g}  移動量:{self.step * 1000:.0f}mm  {self.msg}"
        sys.stdout.write(line + " " * 10)
        sys.stdout.flush()

    # ---------- キーボード ----------
    def keyboard_loop(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:
            self.show()
            while self.running:
                if not select.select([sys.stdin], [], [], 0.1)[0]:
                    self.show()   # P / X の進み具合を表示に反映
                    continue
                key = sys.stdin.read(1)
                k = key.lower()
                s = self.step
                if k == 'w':
                    self.move_tip(dx=-s)
                elif k == 's':
                    self.move_tip(dx=+s)
                elif k == 'a':
                    self.move_tip(dy=-s)
                elif k == 'd':
                    self.move_tip(dy=+s)
                elif k == 'r':
                    self.move_tip(dz=+s)
                elif k == 'f':
                    self.move_tip(dz=-s)
                elif k == 'z':
                    self.step = 0.01 if self.step < 0.01 else 0.005
                elif k == 'o':
                    self.gripper = GRIPPER_OPEN
                elif k == 'c':
                    self.gripper = GRIPPER_CLOSE
                elif k == 'p':
                    with self.lock:
                        self.gripper = GRIPPER_OPEN
                        self.sequence = preset_steps()
                        self.tip = None
                        self.msg = 'プリセットへ移動中'
                elif k == 'x':
                    with self.lock:
                        self.sequence = []
                        self.goal = [0.0, 0.0, 0.0, 0.0]
                        self.tip = None
                        self.gripper = GRIPPER_CLOSE
                        self.msg = '初期姿勢へ'
                elif ord(key) == 27:
                    self.running = False
                    break
                self.show()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
            print()


def main():
    rclpy.init()
    node = CraneTeleop()
    try:
        while rclpy.ok() and node.running:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.running = False
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
